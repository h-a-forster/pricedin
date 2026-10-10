from __future__ import annotations

import math

import numpy as np
import pytest

from pricedin.eval import scoring as S


def test_perfect_forecast_scores_zero():
    y = np.array([0, 1, 1, 0])
    assert S.brier(y, y.astype(float)) == 0.0
    assert S.log_loss(y, y.astype(float)) < 1e-12


def test_constant_half():
    y = np.array([0, 1, 1, 0, 1])
    p = np.full(5, 0.5)
    assert S.brier(y, p) == pytest.approx(0.25)
    assert S.log_loss(y, p) == pytest.approx(math.log(2))


def test_per_row_versions_match():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    p = rng.random(50)
    assert S.brier_rows(y, p).mean() == pytest.approx(S.brier(y, p))
    expected = -math.log(p[0]) if y[0] else -math.log(1 - p[0])
    assert S.log_loss_rows(y, p)[0] == pytest.approx(expected)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        S.brier([0, 1], [0.5])


def test_brier_skill():
    y = np.array([0, 1, 1, 0])
    assert S.brier_skill(y, y.astype(float), np.full(4, 0.5)) == pytest.approx(1.0)
    assert S.brier_skill(y, np.full(4, 0.5), np.full(4, 0.5)) == pytest.approx(0.0)


def test_bins_equal_mass_and_uniform():
    p = np.linspace(0.01, 0.99, 100)
    idx = S.bin_index(p, 10, "quantile")
    assert np.all(np.bincount(idx) == 10)
    idx_u = S.bin_index(np.array([0.0, 0.05, 0.15, 0.95, 1.0]), 10, "uniform")
    assert idx_u.tolist() == [0, 0, 1, 9, 9]


def test_ece_known_values():
    # two bins, each with forecast equal to its observed frequency
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.25] * 4 + [0.75] * 4)
    assert S.ece(y, p, 2, "quantile") == pytest.approx(0.0)
    assert S.ece(y, p, 2, "uniform") == pytest.approx(0.0)
    # all forecasts 0.9 but frequency 0.5: ECE = 0.4
    assert S.ece(np.array([0, 1] * 5), np.full(10, 0.9), 5) == pytest.approx(0.4)


def test_reliability_bins_counts_sum():
    rng = np.random.default_rng(1)
    p = rng.random(237)
    y = (rng.random(237) < p).astype(int)
    b = S.reliability_bins(y, p, 10, "uniform")
    assert sum(b["count"]) == 237
    assert all(lo <= m <= hi for lo, m, hi in zip(b["lo"], b["mean_p"], b["hi"], strict=True))


def test_murphy_exact_for_discrete_forecasts():
    rng = np.random.default_rng(2)
    vals = np.array([0.15, 0.35, 0.55, 0.85])
    p = vals[rng.integers(0, 4, 400)]
    y = (rng.random(400) < p).astype(int)
    m = S.murphy(y, p, 10, "uniform")
    assert m["residual"] == pytest.approx(0.0, abs=1e-12)
    assert m["brier"] == pytest.approx(m["reliability"] - m["resolution"] + m["uncertainty"])


def test_murphy_approximate_for_continuous_forecasts():
    rng = np.random.default_rng(3)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    m = S.murphy(y, p, 10)
    assert abs(m["residual"]) < 0.01
