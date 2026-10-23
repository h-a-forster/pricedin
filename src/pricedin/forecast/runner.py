"""Run one forecaster over every (question, horizon) snapshot and append ``Forecast`` rows.

One task per snapshot: the snapshot's ``as_of`` is the forecast date, so the model is asked
exactly when the market price was taken. Runs are

- resumable: rows already in ``out_path`` with a probability are skipped; failed rows
  (``prob`` None, ``error`` set) stay in the file and the task is retried on the next run.
  Readers should use ``load_forecasts``, which keeps the best row per (question, horizon);
- shardable: ``shard=(i, n)`` keeps questions whose stable hash is ``i`` mod ``n``, so all
  horizons of a question land in the same shard and shards can run on different machines;
- cached: replies are cached on disk keyed by prompt and ``question|horizon|condition|sample``,
  so a re-run (or a run with a different ``out_path``) costs nothing.

In the ``news`` condition a question without a news context (or whose context fetch failed) is
still forecast, with a "no headlines found" note in the prompt: dropping it would make the news
sample differ from the base sample. ``Forecast.usage["news_items"]`` records how many headlines
the prompt showed, and the summary counts the missing contexts, so they can be filtered later.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pricedin.forecast.cache import DEFAULT_DIR, CachedProvider, ResponseCache
from pricedin.forecast.parse import extract_rationale, parse_probability
from pricedin.forecast.prompt import (
    CONDITIONS,
    build_prompt,
    close_time,
    prompt_hash,
    select_headlines,
)
from pricedin.forecast.providers import Provider, ProviderError, make_provider
from pricedin.forecast.providers.base import Completion
from pricedin.schema import (
    HORIZONS,
    Forecast,
    NewsContext,
    NewsItem,
    Question,
    Snapshot,
    parse_time,
    read_jsonl,
    to_dict,
)

# Room for adaptive thinking on the API; the claude CLI ignores it.
DEFAULT_MAX_TOKENS = 8192

Progress = Callable[[int, int, Forecast], None]


def shard_of(question_id: str, n: int) -> int:
    """Stable shard index of a question (sha256, independent of PYTHONHASHSEED)."""
    return int(hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:16], 16) % n


def load_forecasts(path: str | Path) -> dict[tuple[str, str], Forecast]:
    """Rows of a forecast file keyed by (question_id, horizon): the last row with a probability,
    else the last failed row."""
    out: dict[tuple[str, str], Forecast] = {}
    p = Path(path)
    if not p.exists():
        return out
    for f in read_jsonl(p, Forecast):
        key = (f.question_id, f.horizon)
        if f.prob is not None or key not in out or out[key].prob is None:
            out[key] = f
    return out


class _Task:
    __slots__ = ("question", "snapshot", "news", "news_missing")

    def __init__(
        self, question: Question, snapshot: Snapshot, news: list[NewsItem], news_missing: bool
    ):
        self.question = question
        self.snapshot = snapshot
        self.news = news
        self.news_missing = news_missing


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for k, v in usage.items():
        total[k] = total.get(k, 0) + int(v)


def run_forecasts(
    questions: Iterable[Question],
    snapshots: Iterable[Snapshot],
    news: Iterable[NewsContext] | None = None,
    *,
    provider_spec: str,
    condition: str = "base",
    out_path: str | Path,
    workers: int = 4,
    shard: tuple[int, int] | None = None,
    limit: int | None = None,
    cache_dir: str | Path | None = DEFAULT_DIR,
    retries: int = 1,
    sample: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    provider: Provider | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Forecast every (question, horizon) that has a snapshot and append rows to ``out_path``.

    ``provider_spec`` names the forecaster (``claude-cli:claude-opus-5-5@low``, see
    ``make_provider``) and is stored as ``Forecast.forecaster``; pass ``provider`` to use an
    already-built provider under that label (tests). ``retries`` is the number of extra
    attempts per task after an unparseable reply or a non-fatal provider error (transient API
    errors are already retried inside the provider). ``sample`` distinguishes repeated samples
    of the same forecaster in the cache key. ``limit`` caps the number of tasks run this time.
    ``cache_dir=None`` disables the cache. ``progress(done, total, forecast)`` is called after
    each row is written.

    A fatal provider error (not logged in, unknown model, missing key) stops the run: the
    failing row is written, queued tasks are skipped, and ``summary["aborted"]`` holds the
    message. Returns a summary dict (counts, ``cost_usd`` of new calls, token totals).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if shard is not None and not (shard[1] >= 1 and 0 <= shard[0] < shard[1]):
        raise ValueError(f"shard must be (i, n) with 0 <= i < n, got {shard!r}")
    out = Path(out_path)

    existing = load_forecasts(out)
    for f in existing.values():
        if f.forecaster != provider_spec or f.condition != condition:
            raise ValueError(
                f"{out} holds forecasts from {f.forecaster!r}/{f.condition!r}; "
                f"use a separate file for {provider_spec!r}/{condition!r}"
            )
    done = {k for k, f in existing.items() if f.prob is not None}

    qmap = {q.id: q for q in questions}
    nmap = {(c.question_id, c.horizon): c for c in (news or ())}
    horizon_rank = {h: i for i, h in enumerate(HORIZONS)}

    tasks: list[_Task] = []
    n_skipped = n_invalid = n_eligible = 0
    snaps = sorted(
        (s for s in snapshots if s.question_id in qmap),
        key=lambda s: (s.question_id, horizon_rank.get(s.horizon, len(horizon_rank)), s.horizon),
    )
    for s in snaps:
        if shard is not None and shard_of(s.question_id, shard[1]) != shard[0]:
            continue
        q = qmap[s.question_id]
        if parse_time(s.as_of) >= (close_time(q) or q.close):
            n_invalid += 1
            continue
        n_eligible += 1
        if (s.question_id, s.horizon) in done:
            n_skipped += 1
            continue
        items: list[NewsItem] = []
        missing = False
        if condition == "news":
            ctx = nmap.get((s.question_id, s.horizon))
            missing = ctx is None or ctx.error is not None
            if ctx is not None:
                items = select_headlines(ctx.items, parse_time(s.as_of))
        tasks.append(_Task(q, s, items, missing))
    if limit is not None:
        tasks = tasks[: max(0, limit)]

    inner = provider if provider is not None else make_provider(provider_spec)
    cached = CachedProvider(inner, ResponseCache(cache_dir)) if cache_dir is not None else None

    lock = threading.Lock()
    stop = threading.Event()
    summary: dict[str, Any] = {
        "out_path": str(out),
        "forecaster": provider_spec,
        "condition": condition,
        "eligible": n_eligible,
        "skipped_existing": n_skipped,
        "invalid_as_of": n_invalid,
        "tasks": len(tasks),
        "done": 0,
        "failed": 0,
        "cached": 0,
        "not_run": 0,
        "news_missing": 0,
        "cost_usd": 0.0,
        "usage": {},
        "aborted": None,
    }

    def call(system: str, user: str, salt: str) -> Completion:
        if cached is not None:
            return cached.complete(system, user, max_tokens=max_tokens, salt=salt)
        return inner.complete(system, user, max_tokens=max_tokens)

    def run_one(task: _Task) -> Forecast | None:
        if stop.is_set():
            return None
        q, s = task.question, task.snapshot
        system, user, version = build_prompt(q, s.as_of, condition, task.news)
        base_salt = f"{q.id}|{s.horizon}|{condition}|{sample}"
        usage: dict[str, int] = {}
        cost: float | None = None
        new_cost = 0.0
        latency = 0.0
        prob: float | None = None
        text = ""
        model = harness = ""
        error: str | None = None
        all_cached = True
        for attempt in range(1 + max(0, retries)):
            salt = base_salt if attempt == 0 else f"{base_salt}|retry{attempt}"
            try:
                comp = call(system, user, salt)
            except ProviderError as exc:
                error = str(exc)
                all_cached = False
                if exc.fatal:
                    stop.set()
                    with lock:
                        summary["aborted"] = summary["aborted"] or error
                    break
                continue
            except Exception as exc:  # a bug or unexpected failure: record it, keep going
                error = f"{type(exc).__name__}: {exc}"
                all_cached = False
                continue
            _add_usage(usage, comp.usage)
            if comp.cost_usd is not None:
                cost = (cost or 0.0) + comp.cost_usd
                if not comp.cached:
                    new_cost += comp.cost_usd
            all_cached = all_cached and comp.cached
            latency += comp.latency_s
            text = comp.text
            model, harness = comp.model, comp.harness
            prob = parse_probability(text)
            if prob is not None:
                error = None
                break
            error = "unparseable reply: no final PROBABILITY line"
        if condition == "news":
            usage["news_items"] = len(task.news)
        f = Forecast(
            question_id=q.id,
            horizon=s.horizon,
            as_of=s.as_of,
            forecaster=provider_spec,
            condition=condition,
            prob=prob,
            prompt_version=version,
            prompt_hash=prompt_hash(system, user),
            model=model,
            harness=harness,
            rationale=extract_rationale(text),
            usage=usage,
            cost_usd=cost,
            latency_s=round(latency, 3),
            error=error,
        )
        with lock:
            fh.write(json.dumps(to_dict(f), ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            summary["done" if prob is not None else "failed"] += 1
            summary["cached"] += int(all_cached and prob is not None)
            summary["news_missing"] += int(task.news_missing)
            summary["cost_usd"] += new_cost
            _add_usage(summary["usage"], {k: v for k, v in usage.items() if k != "news_items"})
            n = summary["done"] + summary["failed"]
        if progress is not None:
            progress(n, len(tasks), f)
        return f

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as fh:
        if workers <= 1:
            for t in tasks:
                run_one(t)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run_one, tasks))
    summary["not_run"] = len(tasks) - summary["done"] - summary["failed"]
    summary["cost_usd"] = round(summary["cost_usd"], 6)
    return summary
