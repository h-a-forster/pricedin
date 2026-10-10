from __future__ import annotations

import numpy as np
import pytest

from pricedin.eval import pool as P


def _logistic_data(n, coef, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(coef) - 1))
    y = (rng.random(n) < P.sigmoid(coef[0] + X @ np.asarray(coef[1:]))).astype(int)
    return X, y


def test_logit_sigmoid_roundtrip_and_clip():
    p = np.array([0.0, 0.2, 0.5, 1.0])
    assert P.sigmoid(P.logit(p, 0.01)) == pytest.approx([0.01, 0.2, 0.5, 0.99])
    assert np.all(np.isfinite(P.sigmoid(np.array([-1000.0, 1000.0]))))


def test_fit_recovers_coefficients():
    X, y = _logistic_data(20000, [-0.5, 1.2, -0.7])
    fit = P.fit_logistic(X, y)
    assert fit.converged
    assert fit.coef == pytest.approx([-0.5, 1.2, -0.7], abs=0.08)
    w = fit.prob * (1 - fit.prob)
    assert fit.hessian == pytest.approx((fit.design * w[:, None]).T @ fit.design)


def test_fit_solves_score_equations():
    X, y = _logistic_data(500, [0.3, 0.8])
    fit = P.fit_logistic(X, y)
    assert fit.design.T @ (y - fit.prob) == pytest.approx([0, 0], abs=1e-6)


def test_separation_is_finite_with_penalty():
    x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    fit = P.fit_logistic(x, y, l2=1e-3)
    assert fit.converged and np.all(np.isfinite(fit.coef))
    assert fit.coef[1] > 1  # steep but finite
    unpen = P.fit_logistic(x, y, l2=0.0, max_iter=30)
    assert np.all(np.isfinite(unpen.coef))


def test_penalty_skips_intercept():
    y = np.array([1] * 90 + [0] * 10)
    fit = P.fit_logistic(np.zeros(100), y, l2=10.0)
    assert P.sigmoid(fit.coef[0]) == pytest.approx(0.9, abs=1e-6)


def test_named_models_on_known_world():
    rng = np.random.default_rng(5)
    n = 20000
    lm, ll = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < P.sigmoid(0.2 + 0.6 * lm + 0.9 * ll)).astype(int)
    cols = {"market": P.sigmoid(lm), "x|base": P.sigmoid(ll)}
    out = P.fit_predict(P.standard_models("x|base"), cols, y, cols, eps=1e-6)
    pool = out["pool"][1]
    assert pool.coef["intercept"] == pytest.approx(0.2, abs=0.06)
    assert pool.coef["logit(market)"] == pytest.approx(0.6, abs=0.06)
    assert pool.coef["logit(x|base)"] == pytest.approx(0.9, abs=0.06)
    assert out["market"][0] == pytest.approx(cols["market"], abs=1e-9)
    assert out["average"][0] == pytest.approx(P.sigmoid((lm + ll) / 2), abs=1e-9)
    assert set(out["llm_extremized"][1].coef) == {"k"}
    assert out["market"][1].coef == {}


def test_extremize_recovers_k():
    rng = np.random.default_rng(6)
    lp = rng.normal(size=20000)
    y = (rng.random(20000) < P.sigmoid(1.8 * lp)).astype(int)
    fm = P.fit_model(P.ModelSpec("e", "extremize", ("a",)), {"a": P.sigmoid(lp)}, y, eps=1e-6)
    assert fm.coef["k"] == pytest.approx(1.8, abs=0.08)


def test_pool_multi_spec():
    s = P.pool_multi(["a|base", "b|news"])
    assert s.columns == ("market", "a|base", "b|news")
    assert P.coef_names(s) == ["intercept", "logit(market)", "logit(a|base)", "logit(b|news)"]
