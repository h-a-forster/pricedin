"""A small synthetic study with known structure, written to a temp folder at test time.

Truth: theta = event effect + question effect, y ~ Bernoulli(sigmoid(theta)). At each horizon the
market quotes sigmoid(0.8 * (theta + noise)) with noise shrinking as close nears (7d > 1d > 1h),
compressed toward 0.5 (a longshot bias ``market_recal`` should absorb). Two scripted forecasters:

- ``scripted:informed``: 30% the market's view, 70% a private signal on theta (adds information);
- ``scripted:echo``: the market's own view plus noise (encompassed by the market).

Kalshi rows carry candle bid/ask and Kalshi fee terms; Polymarket rows are mid-only with
Polymarket fee terms. About one question in six closed early (``extra.early_close``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from pricedin.schema import HORIZONS, Forecast, Question, Snapshot, to_iso, write_jsonl

INFORMED = "scripted:informed"
ECHO = "scripted:echo"
CATEGORIES = ("politics", "sports", "crypto", "economics")
KALSHI_FEE = {"kind": "kalshi", "fee_type": "quadratic", "multiplier": 1.0}
PM_FEE = {"kind": "polymarket", "rate": 0.04, "exponent": 1, "taker_only": True}
NOISE = {"7d": 1.0, "1d": 0.6, "1h": 0.3}  # market noise on theta by horizon

TOML = """\
[study]
name = "synth"
created_after = "2026-06-01T00:00:00Z"
closed_before = "2026-10-09T00:00:00Z"
horizons = ["7d", "1d", "1h"]
data_dir = "data"
results_dir = "results"

[universe]
venues = ["kalshi", "polymarket"]
per_event = 2
per_cell = 100
seed = "synth"

[forecast]
conditions = ["base"]
forecasters = ["scripted:informed", "scripted:echo"]

[analysis]
eps = 0.01
split = 0.5
bootstrap = 100
seed = 0
alpha = 0.05
contested = [0.05, 0.95]

[analysis.pnl]
threshold = 0.02
contracts = 100
"""


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_study(root: Path, n_questions: int = 200, seed: int = 3) -> Path:
    """Write ``root/synth.toml`` and its data folder. Returns the TOML path."""
    rng = np.random.default_rng(seed)
    data = root / "data"
    n_events = int(n_questions * 0.8)  # some events hold two questions
    ev = np.sort(rng.integers(0, n_events, n_questions))
    theta = rng.normal(0, 1.2, n_events)[ev] + rng.normal(0, 0.8, n_questions)
    y = (rng.random(n_questions) < _sigmoid(theta)).astype(int)
    start = datetime(2026, 7, 20, tzinfo=UTC)
    close_s = rng.random(n_events)[ev] * 78 * 86400  # closes spread over ~11 weeks
    signal = theta + rng.normal(0, 0.5, n_questions)

    questions, snapshots, forecasts = [], [], {INFORMED: [], ECHO: []}
    missing = {}
    for i in range(n_questions):
        venue = "kalshi" if i % 2 == 0 else "polymarket"
        cat = CATEGORIES[int(ev[i]) % len(CATEGORIES)]
        close = start + timedelta(seconds=float(close_s[i]))
        early = i % 6 == 0
        qid = f"{venue}:SYN-{i:04d}"
        questions.append(
            Question(
                id=qid,
                venue=venue,
                event_id=f"{venue}:EV-{int(ev[i]):04d}",
                title=f"Synthetic question {i}",
                rules="Resolves YES if the synthetic event happens.",
                category=cat,
                created_at=to_iso(close - timedelta(days=30)),
                open_at=to_iso(close - timedelta(days=29)),
                close_at=to_iso(close),
                scheduled_close_at="" if early else to_iso(close),
                outcome=int(y[i]),
                volume=5000.0,
                venue_category=cat.title(),
                extra={
                    "fee": KALSHI_FEE if venue == "kalshi" else PM_FEE,
                    "early_close": early,
                    "tick_size": 0.01,
                }
                | ({"series": f"KXSYN{int(ev[i]) % 12}"} if venue == "kalshi" else {}),
            )
        )
        for h in HORIZONS:
            if i % 40 == 3 and h == "7d":  # a few missing snapshots, as in real data
                missing[f"{qid}|{h}"] = "no observation within max_stale"
                continue
            as_of = to_iso(close - HORIZONS[h])
            view = theta[i] + rng.normal(0, NOISE[h])
            mid = float(np.clip(_sigmoid(0.8 * view), 0.01, 0.99))
            bid = ask = None
            if venue == "kalshi":
                m = round(mid, 2)
                bid, ask = max(0.01, m - 0.01), min(0.99, m + 0.01)
                mid = (bid + ask) / 2
            snapshots.append(
                Snapshot(qid, h, as_of, mid=round(mid, 4), bid=bid, ask=ask, source="synthetic")
            )
            p_inf = _sigmoid(0.3 * view + 0.7 * signal[i] + rng.normal(0, 0.3))
            p_echo = _sigmoid(view + rng.normal(0, 0.6))
            for name, p in ((INFORMED, p_inf), (ECHO, p_echo)):
                failed = name == ECHO and i % 50 == 7  # a few unparsed replies
                forecasts[name].append(
                    Forecast(
                        question_id=qid,
                        horizon=h,
                        as_of=as_of,
                        forecaster=name,
                        condition="base",
                        prob=None if failed else round(float(p), 4),
                        prompt_version="v1",
                        error="unparsed" if failed else None,
                    )
                )

    write_jsonl(data / "questions.jsonl", questions)
    write_jsonl(data / "snapshots.jsonl", snapshots)
    for name, rows in forecasts.items():
        slug = name.replace(":", "_").replace("@", "_").replace("/", "_")
        write_jsonl(data / "forecasts" / f"{slug}__base.jsonl.gz", rows)
    (data / "snapshots_missing.json").write_text(json.dumps(missing, indent=1), "utf-8")
    cells = {}
    for q in questions:
        cells[(q.venue, q.category)] = cells.get((q.venue, q.category), 0) + 1
    sample = {
        "candidates": n_questions + 37,
        "events": len({q.event_id for q in questions}) + 12,
        "after_event_cap": n_questions + 5,
        "sampled": n_questions,
        "cells": [
            {"venue": v, "category": c, "available": n + 2, "sampled": n}
            for (v, c), n in sorted(cells.items())
        ],
    }
    (data / "sample.json").write_text(json.dumps(sample, indent=2), "utf-8")
    toml = root / "synth.toml"
    toml.write_text(TOML, "utf-8")
    return toml
