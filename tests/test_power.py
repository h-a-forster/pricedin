from __future__ import annotations

import numpy as np
import pytest

from pricedin.eval.inference import encompassing_test
from pricedin.eval.panel import split_masks
from pricedin.eval.pool import fit_predict, standard_models
from pricedin.eval.power import SIM_KEY, power_curve, simulate_panel


def test_simulate_panel_shape_and_determinism():
    p = simulate_panel(90, 0.5, horizons=("7d", "1d"), seed=3)
    assert len(p) == 180
    assert set(p.horizon.tolist()) == {"7d", "1d"}
    assert np.unique(p.event_id).size == 30
    assert np.all(p.as_of < p.close_at)
    assert np.all((p.market >= 0.01) & (p.market <= 0.99))
    q = simulate_panel(90, 0.5, horizons=("7d", "1d"), seed=3)
    assert np.array_equal(p.forecasts[SIM_KEY], q.forecasts[SIM_KEY])
    tr, te = split_masks(p, "2025-07-01T00:00:00Z")
    assert tr.any() and te.any()


def test_pool_recovers_known_coefficients_on_large_synthetic_data():
    # market sees theta exactly but compressed by k_m; LLM = theta + noise adds nothing:
    # true pool is intercept 0, b = 1 / k_m, c = 0
    p = simulate_panel(
        40000,
        0.0,
        questions_per_event=1,
        s_event=0.6,
        s_question=0.6,
        sigma_m=0.0,
        k_m=0.7,
        eps=1e-6,
        seed=11,
    )
    out = fit_predict(standard_models(SIM_KEY), p.columns(), p.y, p.columns(), eps=1e-6)
    coef = out["pool"][1].coef
    assert coef["intercept"] == pytest.approx(0.0, abs=0.05)
    assert coef["logit(market)"] == pytest.approx(1 / 0.7, abs=0.08)
    assert coef[f"logit({SIM_KEY})"] == pytest.approx(0.0, abs=0.05)
    recal = out["market_recal"][1].coef
    assert recal["logit(market)"] == pytest.approx(1 / 0.7, abs=0.08)


def test_encompassing_size_and_power_light():
    rej0 = rej1 = 0
    rng = np.random.default_rng(0)
    for _ in range(40):
        p0 = simulate_panel(300, 0.0, seed=rng)
        r0 = encompassing_test(p0.y, p0.market, p0.forecasts[SIM_KEY], p0.event_id)
        rej0 += r0["llm_adds_info"]["p"] < 0.05
        p1 = simulate_panel(600, 1.0, sigma_s=0.3, seed=rng)
        r1 = encompassing_test(p1.y, p1.market, p1.forecasts[SIM_KEY], p1.event_id)
        rej1 += r1["llm_adds_info"]["p"] < 0.05
    assert rej0 / 40 <= 0.15
    assert rej1 / 40 >= 0.9


def test_power_curve_runs_pipeline_light():
    res = power_curve([300], [0.0, 1.0], reps=20, B=100, seed=1, sigma_s=0.3)
    assert [r["w"] for r in res] == [0.0, 1.0]
    null, alt = res
    assert null["reject_encompassing_full"] <= 0.2
    assert alt["reject_encompassing_full"] >= 0.8
    assert alt["mean_delta_brier"] > null["mean_delta_brier"]


@pytest.mark.slow
def test_encompassing_false_positive_rate_near_nominal():
    res = power_curve([300], [0.0], reps=200, B=100, seed=2)[0]
    assert 0.02 <= res["reject_encompassing_full"] <= 0.10
    assert res["reject_gain"] <= 0.10


@pytest.mark.slow
def test_power_high_with_strong_private_information():
    # the out-of-sample Brier test is much less powerful than the in-sample encompassing test
    res = power_curve([1500], [1.0], reps=60, B=200, seed=3, sigma_s=0.3)[0]
    assert res["reject_encompassing_full"] >= 0.95
    assert res["reject_gain"] >= 0.4
