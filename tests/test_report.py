from __future__ import annotations

import json

import numpy as np
import pytest
from fixtures.synthetic import ECHO, INFORMED, make_study

from pricedin import plots, report
from pricedin.cli import _questions, _snapshots, load_all_forecasts, study_split
from pricedin.eval import pnl
from pricedin.eval.evaluate import evaluate
from pricedin.eval.inference import holm
from pricedin.eval.panel import build_panel
from pricedin.universe import load_study

KEYS = (f"{INFORMED}|base", f"{ECHO}|base")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """A small synthetic evaluation, built with the library (B small, no subsets)."""
    root = tmp_path_factory.mktemp("rep")
    s = load_study(make_study(root, n_questions=120))
    qs, snaps, fcs = _questions(s), _snapshots(s), load_all_forecasts(s)
    split = study_split(s, qs, snaps)
    res = {"study": s.name, "version": "test", "split_at": split, "keys": {}}
    for k in KEYS:
        panel = build_panel(qs, snaps, fcs, [k], horizons=s.horizons)
        r = evaluate(panel, [k], split=split, B=30, walk_forward_min_train=40)
        r["pnl"] = pnl.backtest_split(panel, qs, snaps, k, split=split, threshold=0.02, B=30)
        res["keys"][k] = r
    s.results_dir.mkdir(parents=True)
    (s.results_dir / "evaluation.json").write_text(json.dumps(res), encoding="utf-8")
    return s


def test_formatters():
    assert report.fb(0.123456) == "0.1235"
    assert report.fd(0.00123) == "+0.0012"
    assert report.fb(None) == "–" and report.fd(float("nan")) == "–"
    assert report.fci({"mean": 0.01, "lo": -0.002, "hi": 0.02}) == "+0.0100 [-0.0020, +0.0200]"
    assert report.fci({"mean": 0.01, "lo": None, "hi": None}) == "+0.0100"
    assert report.fp(0.0001) == "<0.001" and report.fp(0.0456) == "0.046"
    assert plots.short_key("claude-cli:claude-opus-5-5@low|news") == "claude-opus-5-5@low · news"


def test_primary_family_is_holm_over_all_keys(study):
    ev = report.load_evaluation(study.results_dir)
    fam = report.primary_family(ev)
    assert len(fam) == len(KEYS) * len(study.horizons)
    p = np.array([r["p_h1"] for r in fam])
    assert np.allclose([r["p_h1_holm"] for r in fam], holm(p))
    for r in fam:
        assert r["reject_h2"] == (r["p_h2_holm"] < 0.05 and r["delta"] > 0)


def test_write_report(study):
    path = report.write_report(study, study.results_dir)
    text = path.read_text(encoding="utf-8")
    for heading in (
        "## Sample",
        "## Primary hypotheses",
        "## Calibration",
        "## Secondary analyses",
        "### By category",
        "### Walk-forward",
        "## P&L after fees",
    ):
        assert heading in text
    assert "### Contested markets" not in text  # no subsets in this evaluation
    assert "| forecaster | h | n test | market |" in text
    s = json.loads((study.results_dir / "summary.json").read_text(encoding="utf-8"))
    assert s["featured"] in KEYS
    assert set(s["keys"][KEYS[0]]) >= {"7d", "1d", "1h", "all", "pnl"}


def test_report_without_matplotlib(study, monkeypatch, tmp_path):
    def missing():
        raise ImportError("no matplotlib")

    monkeypatch.setattr(plots, "_mpl", missing)
    out = tmp_path / "out"
    out.mkdir()
    (out / "evaluation.json").write_bytes((study.results_dir / "evaluation.json").read_bytes())
    text = report.write_report(study, out).read_text(encoding="utf-8")
    assert "matplotlib not installed" in text and "![" not in text
    assert not (out / "figures").exists()


def test_missing_evaluation(study, tmp_path):
    with pytest.raises(FileNotFoundError):
        report.write_report(study, tmp_path)


def test_figures_are_written_and_deterministic(study, tmp_path):
    pytest.importorskip("matplotlib")
    ev = report.load_evaluation(study.results_dir)
    key = report.featured_key(ev)
    a = plots.all_figures(ev, tmp_path / "a", key)
    b = plots.all_figures(ev, tmp_path / "b", key)
    assert set(a) == {"forest", "brier_by_horizon", "reliability", "encompassing", "pnl"}
    for name, paths in a.items():
        for pa, pb in zip(paths, b[name], strict=True):
            assert pa.stat().st_size > 1000
            assert pa.read_bytes() == pb.read_bytes(), pa.name
        svg = paths[1].read_text(encoding="utf-8")
        assert "<dc:date>" not in svg
