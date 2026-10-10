"""Command-line interface.

pricedin collect   STUDY      list eligible questions on each venue and draw the sample
pricedin snapshots STUDY      market state at each horizon before close
pricedin news      STUDY      dated news published before each snapshot
pricedin forecast  STUDY -f SPEC -c CONDITION
pricedin merge     STUDY      merge sharded forecast files
pricedin evaluate  STUDY      pool, score and test; P&L after fees
pricedin report    STUDY      Markdown report and figures from the evaluation
pricedin power                power analysis on synthetic data
pricedin live      STUDY      prospective collector for open markets (run on a schedule)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

from pricedin import __version__
from pricedin.schema import (
    Forecast,
    NewsContext,
    Question,
    Snapshot,
    parse_time,
    read_jsonl,
    to_iso,
    write_jsonl,
)
from pricedin.universe import Study, load_study, sample

CACHE_DIR = Path(".pricedin-cache")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _client(cache: bool = True):
    from pricedin.http import HttpClient

    return HttpClient(
        cache_dir=CACHE_DIR / "http" if cache else None, max_retries=10, backoff_max=60.0
    )


def _venue(name: str):
    if name == "kalshi":
        from pricedin.venues import kalshi

        return kalshi
    if name == "polymarket":
        from pricedin.venues import polymarket

        return polymarket
    raise SystemExit(f"unknown venue {name!r}")


def _questions(study: Study) -> list[Question]:
    if not study.questions_path.exists():
        raise SystemExit(f"{study.questions_path} not found; run `pricedin collect` first")
    return list(read_jsonl(study.questions_path, Question))


def _snapshots(study: Study) -> list[Snapshot]:
    if not study.snapshots_path.exists():
        raise SystemExit(f"{study.snapshots_path} not found; run `pricedin snapshots` first")
    return list(read_jsonl(study.snapshots_path, Snapshot))


def forecast_slug(spec: str) -> str:
    """``claude-cli:claude-opus-5-5@low`` -> ``claude-cli_claude-opus-5-5_low``."""
    return spec.replace(":", "_").replace("@", "_").replace("/", "_")


def forecast_path(study: Study, spec: str, condition: str, shard: str | None = None) -> Path:
    name = f"{forecast_slug(spec)}__{condition}"
    if shard:
        i, n = shard.split("/")
        return study.forecasts_dir / "shards" / f"{name}.{i}-of-{n}.jsonl"
    return study.forecasts_dir / f"{name}.jsonl.gz"


def load_all_forecasts(study: Study) -> list[Forecast]:
    out: list[Forecast] = []
    for p in sorted(study.forecasts_dir.glob("*.jsonl.gz")):
        out.extend(read_jsonl(p, Forecast))
    return out


# --- commands ---------------------------------------------------------------------------


def cmd_collect(args: argparse.Namespace) -> int:
    study = load_study(args.study)
    client = _client()
    venues = args.venue or study.venues
    candidates: list[Question] = []
    if args.reuse and study.candidates_path.exists():
        candidates = [q for q in read_jsonl(study.candidates_path, Question) if q.venue in venues]
        _log(f"reusing {len(candidates)} candidates from {study.candidates_path}")
        if "polymarket" in venues:  # re-apply eligibility, so rules added since the scan count
            n0 = len(candidates)
            pm_eligible = _venue("polymarket").eligible
            candidates = [
                q
                for q in candidates
                if q.venue != "polymarket"
                or not pm_eligible(
                    q,
                    created_after=study.created_after,
                    closed_before=study.closed_before,
                    min_open_days=study.min_open_days,
                    min_volume=study.min_volume.get("polymarket", 0.0),
                )
            ]
            _log(f"[polymarket] {n0 - len(candidates)} saved candidates no longer eligible")
    else:
        if study.candidates_path.exists():  # keep other venues' candidates from earlier runs
            candidates = [
                q for q in read_jsonl(study.candidates_path, Question) if q.venue not in venues
            ]
        for v in venues:
            mod = _venue(v)
            t0 = time.monotonic()
            n0 = len(candidates)
            for q in mod.list_candidates(
                client,
                created_after=study.created_after,
                closed_before=study.closed_before,
                min_open_days=study.min_open_days,
                min_volume=study.min_volume.get(v, 0.0),
                progress=lambda *msg, v=v: _log(f"[{v}] " + " ".join(map(str, msg))),
            ):
                candidates.append(q)
            _log(f"[{v}] {len(candidates) - n0} candidates in {time.monotonic() - t0:.0f}s")
        write_jsonl(study.candidates_path, candidates)
    chosen, summary = sample(
        candidates, per_event=study.per_event, per_cell=study.per_cell, seed=study.seed
    )
    write_jsonl(study.questions_path, chosen)
    (study.data_dir / "sample.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _log(f"sampled {summary['sampled']} of {summary['candidates']} candidates")
    for c in summary["cells"]:
        _log(f"  {c['venue']:<11} {c['category']:<10} {c['sampled']:>4} / {c['available']}")
    return 0


def cmd_snapshots(args: argparse.Namespace) -> int:
    study = load_study(args.study)
    questions = _questions(study)
    client = _client()
    path = study.snapshots_path
    done = set()
    if path.exists():
        done = {(s.question_id, s.horizon) for s in read_jsonl(path, Snapshot)}
    tasks = [(q, h) for q in questions for h in study.horizons if (q.id, h) not in done]
    missing_path = study.data_dir / "snapshots_missing.json"
    missing: dict[str, str] = {}
    if missing_path.exists():
        missing = json.loads(missing_path.read_text(encoding="utf-8"))
    if not args.retry_missing:
        tasks = [(q, h) for q, h in tasks if f"{q.id}|{h}" not in missing]
    _log(f"{len(tasks)} snapshots to fetch ({len(done)} already present)")

    def one(task: tuple[Question, str]) -> tuple[Question, str, Snapshot | None, str]:
        q, h = task
        try:
            return q, h, _venue(q.venue).snapshot(client, q, h), ""
        except Exception as exc:  # recorded and retried with --retry-missing
            return q, h, None, f"{type(exc).__name__}: {exc}"

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (q, h, snap, err) in enumerate(pool.map(one, tasks), 1):
            if snap is not None:
                write_jsonl(path, [snap], append=True)
                missing.pop(f"{q.id}|{h}", None)
                n_ok += 1
            else:
                missing[f"{q.id}|{h}"] = err or _missing_reason(q, h)
            if i % 100 == 0:
                _log(f"  {i}/{len(tasks)} ({n_ok} ok)")
    missing_path.write_text(json.dumps(missing, indent=1, sort_keys=True) + "\n", "utf-8")
    _log(f"{n_ok} snapshots written, {len(missing)} missing (see {missing_path.name})")
    return 0


def _missing_reason(q: Question, horizon: str) -> str:
    from pricedin.schema import HORIZONS

    actual = q.extra.get("actual_close")
    if actual and q.close - HORIZONS[horizon] >= parse_time(actual) - timedelta(seconds=60):
        return "market closed before the snapshot time"
    return "no observation within max_stale"


def cmd_news(args: argparse.Namespace) -> int:
    from pricedin.news import collect_news

    study = load_study(args.study)
    summary = collect_news(
        _questions(study),
        _snapshots(study),
        sources=tuple(args.source or ["wikipedia"]),
        out_path=study.news_path,
        cache_dir=CACHE_DIR / "news",
        k=args.k,
        window_days=args.window_days,
        retry_errors=args.retry_errors,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from pricedin.forecast.runner import run_forecasts

    study = load_study(args.study)
    questions = _questions(study)
    if args.venue:
        questions = [q for q in questions if q.venue in args.venue]
    snapshots = [s for s in _snapshots(study) if s.horizon in (args.horizon or study.horizons)]
    news: list[NewsContext] | None = None
    if args.condition == "news":
        if not study.news_path.exists():
            raise SystemExit(f"{study.news_path} not found; run `pricedin news` first")
        news = list(read_jsonl(study.news_path, NewsContext))
        # The news condition only runs where retrieval found something; elsewhere its prompt
        # would differ from the base prompt by a "no headlines found" note alone.
        with_items = {(c.question_id, c.horizon) for c in news if c.items}
        snapshots = [s for s in snapshots if (s.question_id, s.horizon) in with_items]
    shard = None
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        shard = (i, n)
    out = (
        Path(args.out)
        if args.out
        else forecast_path(study, args.forecaster, args.condition, args.shard)
    )
    if out.suffix == ".gz":
        # Runs append row by row; write plain JSONL and compress with `pricedin merge`.
        out = out.with_suffix("")
    t0 = time.monotonic()

    def progress(done: int, total: int, fc: Forecast) -> None:
        if done % 25 == 0 or done == total:
            _log(f"  {done}/{total}  {time.monotonic() - t0:.0f}s")

    summary = run_forecasts(
        questions,
        snapshots,
        news,
        provider_spec=args.forecaster,
        condition=args.condition,
        out_path=out,
        workers=args.workers,
        shard=shard,
        limit=args.limit,
        cache_dir=CACHE_DIR / "llm",
        progress=progress,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary.get("aborted") else 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Combine plain and sharded forecast files into one ``.jsonl.gz`` per forecaster."""
    from pricedin.forecast.runner import load_forecasts

    study = load_study(args.study)
    fdir = study.forecasts_dir
    groups: dict[str, list[Path]] = {}
    for p in sorted([*fdir.glob("*.jsonl"), *fdir.glob("shards/*.jsonl")]):
        name = p.name.split(".")[0]
        groups.setdefault(name, []).append(p)
    for name, paths in groups.items():
        final = fdir / f"{name}.jsonl.gz"
        best: dict[tuple[str, str], Forecast] = {}
        if final.exists():
            best.update(load_forecasts(final))
        for p in paths:
            for key, fc in load_forecasts(p).items():
                if fc.prob is not None or key not in best:
                    best[key] = fc
        rows = sorted(best.values(), key=lambda f: (f.question_id, f.horizon))
        write_jsonl(final, rows)
        ok = sum(f.prob is not None for f in rows)
        _log(f"{final.name}: {len(rows)} rows ({ok} with a probability) from {len(paths)} files")
        if args.delete:
            for p in paths:
                p.unlink()
    return 0


def _forecast_keys(forecasts: Sequence[Forecast]) -> list[str]:
    from pricedin.eval.panel import forecaster_key

    return sorted({forecaster_key(f.forecaster, f.condition) for f in forecasts})


def cmd_evaluate(args: argparse.Namespace) -> int:
    from pricedin.eval.evaluate import evaluate
    from pricedin.eval.panel import build_panel

    study = load_study(args.study)
    questions = _questions(study)
    snapshots = _snapshots(study)
    forecasts = load_all_forecasts(study)
    if not forecasts:
        raise SystemExit(f"no forecasts under {study.forecasts_dir}")
    a = study.analysis
    eps = float(a.get("eps", 0.01))
    B = args.bootstrap or int(a.get("bootstrap", 2000))
    keys = args.key or _forecast_keys(forecasts)
    out_dir = Path(args.out) if args.out else study.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split = study_split(study, questions, snapshots)
    _log(f"split at {split}")
    results: dict[str, Any] = {
        "study": study.name,
        "version": __version__,
        "split_at": split,
        "keys": {},
    }
    # Each forecaster is evaluated on the rows it covers; the joint panel (all keys present)
    # feeds the multi-LLM pool.
    for k in keys:
        panel = build_panel(questions, snapshots, forecasts, [k], horizons=study.horizons, eps=eps)
        _log(f"{k}: {len(panel)} rows")
        res = evaluate(
            panel,
            [k],
            split=split,
            eps=eps,
            B=B,
            seed=int(a.get("seed", 0)),
            alpha=float(a.get("alpha", 0.05)),
        )
        res["subsets"] = _subsets(panel, k, questions, split, a, eps, B)
        res["pnl"] = _pnl(panel, k, questions, snapshots, split, a, eps, B)
        results["keys"][k] = res
    results["joint"] = _joint(questions, snapshots, forecasts, keys, study, split, eps, B)

    from pricedin.eval.hypotheses import cluster_sensitivity, primary_family

    results["primary_family"] = primary_family(results)
    results["cluster_sensitivity"] = cluster_sensitivity(results)
    (out_dir / "evaluation.json").write_text(
        json.dumps(results, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    _log(f"wrote {out_dir / 'evaluation.json'}")
    return 0


def _joint(questions, snapshots, forecasts, keys, study, split, eps, B) -> dict[str, Any]:
    """Multi-forecast pools on rows where every member has a forecast.

    - ``condition:<c>``: every forecaster in one condition (do models add to each other?)
    - ``forecaster:<f>``: one forecaster across conditions (does news add beyond base?)
    """
    from pricedin.eval.evaluate import evaluate
    from pricedin.eval.panel import build_panel

    groups: dict[str, list[str]] = {}
    for k in keys:
        forecaster, condition = k.rsplit("|", 1)
        groups.setdefault(f"condition:{condition}", []).append(k)
        groups.setdefault(f"forecaster:{forecaster}", []).append(k)
    a = study.analysis
    out: dict[str, Any] = {}
    for name, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        panel = build_panel(
            questions, snapshots, forecasts, members, horizons=study.horizons, eps=eps
        )
        _log(f"joint {name}: {len(members)} forecasts, {len(panel)} rows")
        try:
            out[name] = evaluate(
                panel,
                members,
                split=split,
                eps=eps,
                B=B,
                seed=int(a.get("seed", 0)),
                alpha=float(a.get("alpha", 0.05)),
                models=["pool", "pool_multi"],
                include_walk_forward=False,
                include_breakdowns=False,
                include_reliability=False,
                include_cluster_sensitivity=False,
            )
        except Exception as exc:  # recorded rather than failing the evaluation
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _subsets(panel, key, questions, split, a, eps, B) -> dict[str, Any]:
    """Secondary analyses on pre-registered subsets (no breakdowns, no walk-forward).

    There is no early-close subset: an early close of a "by <date>" market means YES, so
    selecting on it conditions on the outcome."""
    from pricedin.eval.evaluate import evaluate

    lo, hi = a.get("contested", [0.05, 0.95])
    masks = {"contested": (panel.market >= lo) & (panel.market <= hi)}
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        sub = panel.subset(mask)
        try:
            out[name] = evaluate(
                sub,
                [key],
                split=split,
                eps=eps,
                B=B,
                seed=int(a.get("seed", 0)),
                alpha=float(a.get("alpha", 0.05)),
                include_walk_forward=False,
                include_breakdowns=False,
                include_reliability=False,
                include_cluster_sensitivity=False,
            )
        except Exception as exc:  # a tiny subset can be unfittable; record why
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _pnl(panel, key, questions, snapshots, split, a, eps, B) -> dict[str, Any]:
    from pricedin.eval import pnl

    cfg = a.get("pnl", {})
    try:
        return pnl.backtest_split(
            panel,
            questions,
            snapshots,
            key,
            split=split,
            eps=eps,
            threshold=float(cfg.get("threshold", 0.02)),
            sizing=pnl.Sizing(kind="contracts", contracts=int(cfg.get("contracts", 100))),
            B=B,
            seed=int(a.get("seed", 0)),
            alpha=float(a.get("alpha", 0.05)),
        )
    except Exception as exc:  # recorded in the results rather than failing the evaluation
        return {"error": f"{type(exc).__name__}: {exc}"}


def study_split(study: Study, questions: Sequence[Question], snapshots: Sequence[Snapshot]) -> str:
    """The split time: a fixed ISO time, or the given quantile of close times over questions
    that have at least one snapshot. Shared by every forecaster so all use the same periods."""
    split = study.analysis.get("split", 0.5)
    if isinstance(split, str):
        return split
    have = {s.question_id for s in snapshots}
    from pricedin.eval.panel import resolved_at

    closes = sorted(parse_time(resolved_at(q)) for q in questions if q.id in have)
    if not closes:
        raise SystemExit("no questions with snapshots")
    return to_iso(closes[min(len(closes) - 1, int(float(split) * len(closes)))])


def cmd_report(args: argparse.Namespace) -> int:
    from pricedin import report

    study = load_study(args.study)
    out = report.write_report(study, Path(args.out) if args.out else study.results_dir)
    _log(f"wrote {out}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    from pricedin.eval.power import power_curve

    rows = power_curve(
        args.n,
        args.w,
        reps=args.reps,
        B=args.bootstrap,
        seed=args.seed,
    )
    print(f"{'n':>6} {'w':>5} {'encompassing':>13} {'oos gain':>9} {'mean dBrier':>12}")
    for r in rows:
        print(
            f"{r['n_questions']:>6} {r['w']:>5.2f} {r['reject_encompassing_full']:>13.2f} "
            f"{r['reject_gain']:>9.2f} {r['mean_delta_brier']:>12.4f}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from pricedin import live

    return live.main(args)


def cmd_status(args: argparse.Namespace) -> int:
    """Row counts for each stage of a study."""
    study = load_study(args.study)
    print(f"study {study.name}: {study.data_dir}")
    for label, path in [
        ("candidates", study.candidates_path),
        ("questions", study.questions_path),
        ("snapshots", study.snapshots_path),
        ("news", study.news_path),
    ]:
        n = sum(1 for _ in read_jsonl(path)) if path.exists() else 0
        print(f"  {label:<11} {n:>7}")
    for p in sorted(study.forecasts_dir.glob("**/*.jsonl*")):
        rows = list(read_jsonl(p))
        ok = Counter(r.get("prob") is not None for r in rows)
        print(f"  {p.relative_to(study.forecasts_dir)}: {ok[True]} ok, {ok[False]} failed")
    return 0


# --- parser -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pricedin",
        description="Does an LLM add information beyond the prediction-market price?",
    )
    ap.add_argument("--version", action="version", version=f"pricedin {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="list eligible questions and draw the sample")
    p.add_argument("study")
    p.add_argument("--venue", action="append", choices=["kalshi", "polymarket"])
    p.add_argument("--reuse", action="store_true", help="re-sample from saved candidates")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("snapshots", help="market state at each horizon before close")
    p.add_argument("study")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retry-missing", action="store_true")
    p.set_defaults(func=cmd_snapshots)

    p = sub.add_parser("news", help="dated news published before each snapshot")
    p.add_argument("study")
    p.add_argument("--source", action="append", choices=["wikipedia", "gdelt"])
    p.add_argument("-k", type=int, default=25, help="items kept per snapshot")
    p.add_argument("--window-days", type=int, default=14)
    p.add_argument("--retry-errors", action="store_true")
    p.set_defaults(func=cmd_news)

    p = sub.add_parser("forecast", help="LLM probabilities for every snapshot")
    p.add_argument("study")
    p.add_argument(
        "-f",
        "--forecaster",
        required=True,
        help="e.g. claude-cli:claude-opus-5-5@low, anthropic:claude-opus-5-5",
    )
    p.add_argument("-c", "--condition", default="base", choices=["base", "news"])
    p.add_argument("--horizon", action="append", help="restrict to these horizons")
    p.add_argument("--venue", action="append", choices=["kalshi", "polymarket"])
    p.add_argument("--shard", help="i/n: run one of n deterministic shards")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="run at most this many new forecasts")
    p.add_argument("--out", help="output JSONL (default: data/<study>/forecasts/...)")
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("merge", help="merge forecast shards into .jsonl.gz files")
    p.add_argument("study")
    p.add_argument("--delete", action="store_true", help="delete merged source files")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("evaluate", help="pool, score, test and backtest")
    p.add_argument("study")
    p.add_argument("--key", action="append", help="forecaster|condition keys to evaluate")
    p.add_argument("--bootstrap", type=int, help="override the study's bootstrap count")
    p.add_argument("--out", help="output folder (default: results/<study>)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="Markdown report and figures")
    p.add_argument("study")
    p.add_argument("--out", help="output folder (default: results/<study>)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("power", help="power analysis on synthetic data")
    p.add_argument("--n", type=int, nargs="+", default=[300, 600, 1200])
    p.add_argument("--w", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write the rows to this file")
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("live", help="prospective collector for open markets")
    p.add_argument("study")
    p.add_argument("--once", action="store_true", help="one pass, then exit (for cron)")
    p.add_argument("--dry-run", action="store_true", help="snapshot only, no LLM calls")
    p.add_argument("-f", "--forecaster", action="append", help="replace [live] forecasters")
    p.add_argument("--data-dir", help="state folder (default: the study's data_dir)")
    p.add_argument("--max-forecasts", type=int, help="override max_forecasts_per_pass")
    p.add_argument("--max-tracked", type=int, help="override max_tracked")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("status", help="row counts per stage")
    p.add_argument("study")
    p.set_defaults(func=cmd_status)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
