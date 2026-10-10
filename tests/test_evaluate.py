from __future__ import annotations

import json

import numpy as np
import pytest

from pricedin.eval.evaluate import ALL_MODELS, evaluate
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


def test_primary_family_holm(result):
    fam = result["primary"]["family"]
    assert len(fam) == 2 * 2  # horizons (not "all") x forecasters
    for r in fam:
        assert r["p_holm"] >= r["p"]


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
    assert r["primary"]["family"][0]["p_holm"] is None
