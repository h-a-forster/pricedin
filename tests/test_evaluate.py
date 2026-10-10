from __future__ import annotations

import json

import numpy as np
import pytest

from pricedin.eval.evaluate import ALL_MODELS, evaluate
from pricedin.eval.hypotheses import primary_family
from pricedin.eval.power import SIM_KEY, simulate_panel


def _panel(w=1.0, n=600, seed=0):
    p = simulate_panel(n, w, horizons=("7d", "1d"), sigma_s=0.3, seed=seed)
    # add a second forecaster (noisy copy) and two categories / venues
    rng = np.random.default_rng(seed)
    noisy = np.clip(p.forecasts[SIM_KEY] + rng.normal(0, 0.05, len(p)), 0.01, 0.99)
    p.forecasts["other|news"] = noisy
    p.category = np.where(np.arange(len(p)) % 10 == 0, "crypto", "politics").astype(object)
    p.venue = np.where(np.arange(len(p)) % 2 == 0, "kalshi", "polymarket").astype(object)
    return p


@pytest.fixture(scope="module")
def result():
    return evaluate(_panel(), B=300, seed=7, walk_forward_min_train=100)


def test_result_is_strict_json(result):
    s = json.dumps(result, allow_nan=False)
    assert json.loads(s) == result


def test_structure(result):
    assert result["meta"]["horizons"] == ["7d", "1d"]
    assert set(result["by_horizon"]) == {"7d", "1d", "all"}
    blk = result["by_horizon"]["1d"]
    assert blk["n_train"] + blk["n_test"] + blk["n_embargoed"] == blk["n"]
    f = blk["forecasters"][SIM_KEY]
    assert set(f["models"]) == set(ALL_MODELS) - {"pool_multi"}
    m = f["models"]["pool"]
    for key in ("n", "brier", "log_loss", "ece", "murphy", "reliability", "coef"):
        assert key in m
    for bench in ("market", "market_recal"):
        d = m[f"delta_vs_{bench}"]["brier"]
        assert d["lo"] <= d["mean"] <= d["hi"] and 0 < d["p"] <= 1
    assert f["models"]["market"]["delta_vs_market"]["brier"]["mean"] == 0.0
    assert set(f["encompassing"]) == {"train", "full"}
    assert set(f["breakdowns"]["category"]) == {"crypto", "politics"}
    assert f["breakdowns"]["category"]["crypto"]["small"] in (True, False)
    assert "pool_multi" in blk["pool_multi"]["models"]
    assert result["walk_forward"]["1d"][SIM_KEY]["n_folds"] > 0


def test_detects_strong_signal(result):
    f = result["by_horizon"]["1d"]["forecasters"][SIM_KEY]
    assert f["encompassing"]["full"]["llm_adds_info"]["p"] < 0.01
    assert f["models"]["pool"]["delta_vs_market_recal"]["brier"]["mean"] > 0
    assert f["models"]["pool"]["coef"][f"logit({SIM_KEY})"] > 0


def test_no_per_result_primary_family(result):
    # the Holm family is built once, over all keys, by hypotheses.primary_family
    assert "primary" not in result


def test_h2_p_is_one_sided(result):
    d = result["by_horizon"]["1d"]["forecasters"][SIM_KEY]["models"]["pool"]
    d = d["delta_vs_market_recal"]["brier"]
    assert d["mean"] > 0 and 0 < d["p_greater"] <= d["p"]


def test_cluster_sensitivity(result):
    for h in ("7d", "1d"):
        f = result["by_horizon"][h]["forecasters"][SIM_KEY]
        cs = f["cluster_sensitivity"]
        assert set(cs) == {"series", "resolution_date"}
        # simulated panels have no series: series clusters fall back to events
        ev = f["encompassing"]["full"]
        assert cs["series"]["h1"]["p"] == pytest.approx(ev["llm_adds_info"]["p"])
        assert cs["series"]["h1"]["c"] == pytest.approx(ev["coef"]["llm"])
        rd = cs["resolution_date"]
        assert rd["h1"]["c"] == pytest.approx(ev["coef"]["llm"])  # same fit, other errors
        assert 0 < rd["h2"]["p_greater"] <= 1 and rd["h2"]["n_clusters"] >= 2
        pool = f["models"]["pool"]["delta_vs_market_recal"]["brier"]
        assert rd["h2"]["mean"] == pytest.approx(pool["mean"])
    assert "cluster_sensitivity" not in result["by_horizon"]["all"]["forecasters"][SIM_KEY]


def test_cluster_sensitivity_can_be_disabled():
    r = evaluate(
        _panel(n=200, seed=8),
        [SIM_KEY],
        B=30,
        include_walk_forward=False,
        include_cluster_sensitivity=False,
    )
    assert "cluster_sensitivity" not in r["by_horizon"]["1d"]["forecasters"][SIM_KEY]


def test_deterministic_given_seed():
    p = _panel(n=300, seed=4)
    kw = dict(B=100, seed=3, include_walk_forward=False)
    assert evaluate(p, **kw) == evaluate(p, **kw)


def test_model_subset_and_split_string():
    p = _panel(n=300, seed=5)
    r = evaluate(
        p,
        [SIM_KEY],
        split="2025-07-01T00:00:00Z",
        models=["pool"],
        B=50,
        include_walk_forward=False,
        include_breakdowns=False,
    )
    models = r["by_horizon"]["1d"]["forecasters"][SIM_KEY]["models"]
    assert set(models) == {"market", "market_recal", "pool"}
    assert r["meta"]["split_at"] == "2025-07-01T00:00:00Z"
    with pytest.raises(ValueError):
        evaluate(p, models=["nope"])


def test_skips_when_training_period_empty():
    p = _panel(n=300, seed=6)
    r = evaluate(p, [SIM_KEY], split="2024-01-01T00:00:00Z", B=50, include_walk_forward=False)
    f = r["by_horizon"]["1d"]["forecasters"][SIM_KEY]
    assert "skipped" in f
    fam = primary_family({"keys": {SIM_KEY: r}})
    assert fam[0]["p_h2_holm"] is None and fam[0]["reject_h2"] is False
    assert f["cluster_sensitivity"]["series"]["h2"] == {"skipped": "pool not fitted"}
