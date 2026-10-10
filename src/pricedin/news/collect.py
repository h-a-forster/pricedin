"""Build ``news.jsonl``: one ``NewsContext`` of relevant, pre-``as_of`` items per snapshot.

Candidates come from the enabled sources (Wikipedia Current events by default, GDELT optional),
are ranked against the question with BM25 and cut to the top ``k``. The run is resumable: rows
already in ``out_path`` are skipped, and results are appended event by event. A source failure
becomes ``NewsContext.error`` on the affected rows instead of stopping the run. Leakage is a
hard failure: every row is checked (``seen_at <= as_of`` for each item) right before writing.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pricedin.http import HttpClient
from pricedin.news import gdelt
from pricedin.news.relevance import rank_news
from pricedin.news.wikipedia import CurrentEvents
from pricedin.schema import (
    NewsContext,
    NewsItem,
    Question,
    Snapshot,
    parse_time,
    read_jsonl,
    write_jsonl,
)

SOURCES = ("wikipedia", "gdelt")


class LeakageError(AssertionError):
    """A news item is dated after the snapshot it was attached to."""


def check_no_leakage(ctx: NewsContext) -> None:
    """Raise ``LeakageError`` unless every item has ``seen_at <= as_of``."""
    as_of = parse_time(ctx.as_of)
    for it in ctx.items:
        try:
            seen = parse_time(it.seen_at)
        except (TypeError, ValueError) as exc:
            raise LeakageError(
                f"{ctx.question_id}/{ctx.horizon}: bad seen_at {it.seen_at!r}"
            ) from exc
        if seen > as_of:
            raise LeakageError(
                f"{ctx.question_id}/{ctx.horizon}: item seen {it.seen_at} after as_of {ctx.as_of}: "
                f"{it.title[:80]!r}"
            )


def _existing(path: Path, retry_errors: bool) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    rows = list(read_jsonl(path))
    if retry_errors and any(r.get("error") for r in rows):
        keep = [r for r in rows if not r.get("error")]
        tmp = path.with_name(path.name + ".tmp")
        write_jsonl(tmp, keep)
        os.replace(tmp, path)
        rows = keep
    return {(r["question_id"], r["horizon"]) for r in rows}


def collect_news(
    questions: Iterable[Question],
    snapshots: Iterable[Snapshot],
    *,
    sources: Sequence[str] = ("wikipedia",),
    out_path: str | Path,
    cache_dir: str | Path | None,
    k: int = 25,
    window_days: int = 14,
    client: HttpClient | None = None,
    retry_errors: bool = False,
) -> dict[str, Any]:
    """Write one ``NewsContext`` per (question, horizon) snapshot to ``out_path``.

    ``sources`` is any of ``("wikipedia", "gdelt")``. Existing rows are skipped (errored rows
    are retried and replaced when ``retry_errors``). ``client`` defaults to a polite
    ``HttpClient`` caching under ``cache_dir``. Returns counts, request stats and timing.
    """
    unknown = set(sources) - set(SOURCES)
    if unknown or not sources:
        raise ValueError(f"unknown news sources {sorted(unknown)}; choose from {SOURCES}")
    t0 = time.monotonic()
    out = Path(out_path)
    done = _existing(out, retry_errors)
    qs = {q.id: q for q in questions}

    by_event: dict[str, list[Snapshot]] = defaultdict(list)
    summary: dict[str, Any] = {
        "rows_written": 0,
        "skipped_existing": 0,
        "skipped_unknown_question": 0,
        "rows_with_error": 0,
        "empty_rows": 0,
        "items": 0,
    }
    for s in snapshots:
        if (s.question_id, s.horizon) in done:
            summary["skipped_existing"] += 1
        elif s.question_id not in qs:
            summary["skipped_unknown_question"] += 1
        else:
            done.add((s.question_id, s.horizon))  # also dedupes repeated snapshots
            by_event[qs[s.question_id].event_id].append(s)

    own_client = client is None
    http = client or HttpClient(cache_dir=cache_dir)
    wiki = CurrentEvents(http, cache_dir) if "wikipedia" in sources else None
    try:
        for event_id in sorted(by_event):
            snaps = sorted(by_event[event_id], key=lambda s: (s.question_id, s.horizon))
            as_ofs = sorted({parse_time(s.as_of) for s in snaps})
            pool: dict[datetime, list[NewsItem]] = defaultdict(list)
            errors: list[str] = []
            gdelt_query = ""
            if wiki is not None:
                try:
                    for t, items in wiki.items_as_of(as_ofs, days=window_days).items():
                        pool[t].extend(items)
                except Exception as exc:  # recorded per row; the run goes on
                    errors.append(f"wikipedia: {type(exc).__name__}: {exc}")
            if "gdelt" in sources:
                lead = qs[min(s.question_id for s in snaps)]
                try:
                    gdelt_query, got = gdelt.items_for_event(
                        http, lead.title, lead.rules, as_ofs, days=window_days
                    )
                    for t, items in got.items():
                        pool[t].extend(items)
                except Exception as exc:
                    errors.append(f"gdelt: {type(exc).__name__}: {exc}")

            rows = []
            for s in snaps:
                q = qs[s.question_id]
                t = parse_time(s.as_of)
                items = rank_news(q.title, q.rules, pool.get(t, []), k=k)
                query = q.title + (f" || gdelt: {gdelt_query}" if gdelt_query else "")
                ctx = NewsContext(
                    question_id=s.question_id,
                    horizon=s.horizon,
                    as_of=s.as_of,
                    query=query,
                    items=items,
                    source="+".join(sources),
                    error="; ".join(errors) or None,
                )
                check_no_leakage(ctx)
                rows.append(ctx)
                summary["items"] += len(items)
                summary["empty_rows"] += not items
                summary["rows_with_error"] += bool(errors)
            summary["rows_written"] += write_jsonl(out, rows, append=True)
    finally:
        if own_client:
            http.close()
    summary["requests"] = http.stats.get("requests", 0)
    summary["cache_hits"] = http.stats.get("cache_hits", 0)
    summary["retries"] = http.stats.get("retries", 0)
    summary["seconds"] = round(time.monotonic() - t0, 2)
    return summary
