"""End to end: synthetic study -> `pricedin evaluate` -> `pricedin report`.

The informed forecaster carries a private signal, so the encompassing coefficient c must be
clearly positive; the echo forecaster only re-encodes the market, so c must not be.
"""

from __future__ import annotations

import json

import pytest
from fixtures.synthetic import ECHO, INFORMED, make_study

from pricedin.cli import main

INF, ECH = f"{INFORMED}|base", f"{ECHO}|base"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    toml = make_study(root)
    assert main(["evaluate", str(toml), "--bootstrap", "100"]) == 0
    assert main(["report", str(toml)]) == 0
    return root / "results"


@pytest.mark.slow
def test_outputs_exist(run):
    assert (run / "evaluation.json").exists()
    assert (run / "report.md").exists()
    assert (run / "summary.json").exists()
    pytest.importorskip("matplotlib")
    for name in ("forest", "reliability", "encompassing", "pnl", "brier_by_horizon"):
        for ext in ("png", "svg"):
            assert (run / "figures" / f"{name}.{ext}").stat().st_size > 1000


@pytest.mark.slow
def test_numbers_are_sane(run):
    ev = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    assert set(ev["keys"]) == {INF, ECH}
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    inf, ech = s["keys"][INF], s["keys"][ECH]
    # H1: the informed LLM adds information, the echo does not
    assert inf["all"]["c"] > 0 and inf["all"]["c_z"] > 2
    assert abs(ech["all"]["c_z"]) < 2
    assert inf["7d"]["reject_h1"]
    assert inf["all"]["delta"] > ech["all"]["delta"]
    # the market sharpens as close nears
    b = {h: inf[h]["brier"]["market"] for h in ("7d", "1d", "1h")}
    assert b["7d"] > b["1d"] > b["1h"]
    for h in ("7d", "1d", "1h", "all"):
        for v in inf[h]["brier"].values():
            assert 0.05 < v < 0.3
    # panel drops: 5 missing snapshots for both, 12 unparsed echo replies
    assert ev["keys"][ECH]["meta"]["panel_dropped"] == {
        "missing_snapshot": 5,
        f"unparsed_forecast:{ECH}": 12,
    }
    # P&L: computed, and only Polymarket rows are approximate
    pnl = ev["keys"][INF]["pnl"]
    assert "error" not in pnl
    groups = pnl["by_group"]
    assert pnl["meta"]["n_approximate_rows"] == groups["venue:polymarket"]["n_rows"]
    assert groups["venue:kalshi"]["models"]["pool"]["n_approximate"] == 0
    # secondary subsets ran
    for name in ("contested", "no_early_close"):
        assert "error" not in ev["keys"][INF]["subsets"][name]


@pytest.mark.slow
def test_report_text(run):
    text = (run / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## Sample",
        "## Primary hypotheses",
        "## Calibration",
        "## Secondary analyses",
        "### Contested markets",
        "### No early close",
        "### By category",
        "### By venue",
        "### Walk-forward",
        "## P&L after fees",
    ):
        assert heading in text
    assert "informed · base" in text and "echo · base" in text
    assert "polymarket (approx.)" in text
    assert "nan" not in text.lower()
