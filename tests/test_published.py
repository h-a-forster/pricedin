"""The published numbers: README, report and summary must be copies of the committed results.

CI also runs ``pricedin check studies/v1.toml``, which re-scores the study from its frozen data
and compares the evaluation itself; these tests only render, so they stay fast.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pricedin import readme, report
from pricedin.universe import load_study

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v1():
    study = load_study(ROOT / "studies" / "v1.toml")
    return study, report.load_evaluation(study.results_dir)


def test_report_and_summary_are_rendered_from_the_committed_evaluation(v1):
    study, ev = v1
    assert (study.results_dir / "report.md").read_text(encoding="utf-8") == report.render(study, ev)
    summary = json.dumps(report.summary(ev), indent=1) + "\n"
    assert (study.results_dir / "summary.json").read_text(encoding="utf-8") == summary


def test_readme_result_is_rendered_from_the_committed_results(v1):
    study, ev = v1
    block = readme.render(study, ev, report.featured_key(ev))
    assert block in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("suffix", "path", "value", "claim"),
    [
        ("|news", ("pool", "delta_vs_market_recal", "brier", "mean"), 0.01, "every horizon"),
        ("|base", ("llm", "brier"), 0.16, "barely move"),
    ],
)
def test_readme_prose_refuses_results_that_contradict_it(v1, suffix, path, value, claim):
    study, ev = v1
    ev = copy.deepcopy(ev)
    key = next(k for k in ev["keys"] if k.endswith(suffix))
    node = ev["keys"][key]["by_horizon"]["1h"]["forecasters"][key]["models"]
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    with pytest.raises(readme.ClaimError, match=claim):
        readme.render(study, ev, report.featured_key(ev))


def test_readme_sync_refuses_a_readme_without_its_markers(tmp_path):
    target = tmp_path / "README.md"
    target.write_text("# no markers\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        readme.sync(target, "block")
    assert target.read_text(encoding="utf-8") == "# no markers\n"


def test_readme_leakage_figure_is_the_scripts_output():
    base = json.loads((ROOT / "results/v1/leakage_check.json").read_text(encoding="utf-8"))["base"]
    share = round(100 * base["right"] / base["calls"])
    text = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert f"({base['right']} of {base['calls']} `base` calls, {share}%;" in text
