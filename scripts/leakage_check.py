"""Leakage check: where the market was unsure (mid 20-80%), how often is a confident LLM
(>= 90% or <= 10%) right? A model that knew outcomes would be right almost always; chance is
about half. Writes results/v1/leakage_check.json, which the README's figure is checked against.
Run from the repo root: uv run python scripts/leakage_check.py
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

from pricedin.schema import Forecast, Question, Snapshot, read_jsonl

OUT = Path("results/v1/leakage_check.json")

qs = {q.id: q for q in read_jsonl("data/v1/questions.jsonl", Question)}
snaps = {(s.question_id, s.horizon): s for s in read_jsonl("data/v1/snapshots.jsonl", Snapshot)}
by_file: dict[str, dict[str, list[int]]] = {}
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
    name = Path(path).name
    by_file[name] = {h: stat[h] for h in sorted(stat)}
    print(
        name,
        {h: f"{r}/{n}" for h, (r, n) in stat.items()},
        "rationales mentioning 'search':",
        tools,
    )

base = [hs for name, hs in by_file.items() if name.endswith("__base.jsonl.gz")]
right = sum(r for hs in base for r, _ in hs.values())
calls = sum(n for hs in base for _, n in hs.values())
summary = {"base": {"right": right, "calls": calls}, "by_file": by_file}
OUT.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
print(f"base: {right} of {calls} right; wrote {OUT}")
