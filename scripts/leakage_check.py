"""Leakage check: where the market was unsure (mid 20-80%), how often is a confident LLM
(>= 90% or <= 10%) right? A model that knew outcomes would be right almost always; chance is
about half. Run from the repo root: uv run python scripts/leakage_check.py
"""

import glob
from collections import defaultdict

from pricedin.schema import Forecast, Question, Snapshot, read_jsonl

qs = {q.id: q for q in read_jsonl("data/v1/questions.jsonl", Question)}
snaps = {(s.question_id, s.horizon): s for s in read_jsonl("data/v1/snapshots.jsonl", Snapshot)}
for path in sorted(glob.glob("data/v1/forecasts/*.jsonl.gz")):
    rows = [f for f in read_jsonl(path, Forecast) if f.prob is not None]
    stat = defaultdict(lambda: [0, 0])
    tools = sum(1 for f in rows if "search" in (f.rationale or "").lower())
    for f in rows:
        s, q = snaps.get((f.question_id, f.horizon)), qs.get(f.question_id)
        if not s or not q or s.mid is None or not 0.2 <= s.mid <= 0.8:
            continue
        if f.prob >= 0.9 or f.prob <= 0.1:
            right = (f.prob >= 0.9) == (q.outcome == 1)
            stat[f.horizon][0] += right
            stat[f.horizon][1] += 1
    print(
        path.split("/")[-1].split("\\")[-1],
        {h: f"{r}/{n}" for h, (r, n) in stat.items()},
        "rationales mentioning 'search':",
        tools,
    )
