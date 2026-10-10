"""Study configuration and the pre-registered question sample.

A study is a TOML file (see ``studies/v1.toml``) that fixes, before any forecast is scored:
the eligibility window, venue filters, the sampling rule, horizons, forecasters and the
analysis plan. Sampling is deterministic: every candidate gets a stable hash of
``seed + question id``, and the sample is a pure function of the candidate list and the
config. Re-running the collector reproduces the same sample.

Sampling rule:

1. Keep at most ``per_event`` questions per event (lowest hash first). Markets in one event
   share news and are often mutually exclusive, so extra members add little information
   and make the effective sample smaller than it looks.
2. Within each (venue, category) cell keep the ``per_cell`` lowest-hash questions. Without
   a cap, sports and crypto ladders would make up most of the sample.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pricedin.schema import HORIZONS, Question, parse_time


@dataclass
class Study:
    name: str
    path: Path
    created_after: datetime
    closed_before: datetime
    min_open_days: float
    horizons: list[str]
    venues: list[str]
    min_volume: dict[str, float]
    per_event: int
    per_cell: int
    seed: str
    data_dir: Path
    results_dir: Path
    forecasters: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def file(self, name: str) -> Path:
        return self.data_dir / name

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "questions.jsonl"

    @property
    def candidates_path(self) -> Path:
        return self.data_dir / "candidates.jsonl.gz"

    @property
    def snapshots_path(self) -> Path:
        return self.data_dir / "snapshots.jsonl"

    @property
    def news_path(self) -> Path:
        return self.data_dir / "news.jsonl.gz"

    @property
    def forecasts_dir(self) -> Path:
        return self.data_dir / "forecasts"


def load_study(path: str | Path) -> Study:
    """Read a study TOML. Relative data/results paths resolve against the TOML's folder."""
    p = Path(path)
    with open(p, "rb") as fh:
        raw = tomllib.load(fh)
    s, u = raw["study"], raw["universe"]
    base = p.parent
    horizons = list(s.get("horizons", list(HORIZONS)))
    unknown = [h for h in horizons if h not in HORIZONS]
    if unknown:
        raise ValueError(f"unknown horizons {unknown}; known: {list(HORIZONS)}")
    f = raw.get("forecast", {})
    return Study(
        name=s["name"],
        path=p,
        created_after=parse_time(s["created_after"]),
        closed_before=parse_time(s["closed_before"]),
        min_open_days=float(s.get("min_open_days", 8)),
        horizons=horizons,
        venues=list(u.get("venues", ["kalshi", "polymarket"])),
        min_volume={k: float(v) for k, v in u.get("min_volume", {}).items()},
        per_event=int(u.get("per_event", 1)),
        per_cell=int(u.get("per_cell", 100)),
        seed=str(u.get("seed", s["name"])),
        data_dir=(base / s.get("data_dir", f"../data/{s['name']}")).resolve(),
        results_dir=(base / s.get("results_dir", f"../results/{s['name']}")).resolve(),
        forecasters=list(f.get("forecasters", [])),
        conditions=list(f.get("conditions", ["base"])),
        analysis=dict(raw.get("analysis", {})),
        raw=raw,
    )


def stable_hash(seed: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}|{question_id}".encode()).hexdigest()


def sample(
    candidates: Iterable[Question], *, per_event: int, per_cell: int, seed: str
) -> tuple[list[Question], dict[str, Any]]:
    """Apply the sampling rule. Returns the sample (sorted by id) and a summary table."""
    pool = sorted(candidates, key=lambda q: stable_hash(seed, q.id))
    seen: set[str] = set()
    unique: list[Question] = []
    for q in pool:  # a candidate list may repeat a question; keep the first copy
        if q.id not in seen:
            seen.add(q.id)
            unique.append(q)

    per_ev: dict[str, int] = defaultdict(int)
    stage1: list[Question] = []
    for q in unique:
        if per_ev[q.event_id] < per_event:
            per_ev[q.event_id] += 1
            stage1.append(q)

    cells: dict[tuple[str, str], int] = defaultdict(int)
    available: dict[tuple[str, str], int] = defaultdict(int)
    chosen: list[Question] = []
    for q in stage1:
        key = (q.venue, q.category)
        available[key] += 1
        if cells[key] < per_cell:
            cells[key] += 1
            chosen.append(q)

    summary = {
        "candidates": len(unique),
        "events": len(per_ev),
        "after_event_cap": len(stage1),
        "sampled": len(chosen),
        "cells": [
            {"venue": v, "category": c, "available": available[(v, c)], "sampled": cells[(v, c)]}
            for v, c in sorted(available)
        ],
    }
    return sorted(chosen, key=lambda q: q.id), summary
