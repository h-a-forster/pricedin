from __future__ import annotations

import math

import numpy as np
import pytest

from pricedin.eval import inference as I
from pricedin.eval.pool import fit_logistic, sigmoid


def test_normal_helpers():
    assert I.norm_cdf(0.0) == pytest.approx(0.5)
    assert I.norm_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)
    assert I.two_sided_p(1.959964) == pytest.approx(0.05, abs=1e-6)
    assert I.chi2_1_sf(3.841459) == pytest.approx(0.05, abs=1e-6)


def test_holm_textbook():
    assert I.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    adj = I.holm([0.5, np.nan, 0.01])
    assert adj[0] == pytest.approx(0.5) and np.isnan(adj[1]) and adj[2] == pytest.approx(0.02)


def test_bootstrap_weights_shape():
    W = I.bootstrap_weights(7, 50, np.random.default_rng(0))
    assert W.shape == (50, 7) and np.all(W.sum(axis=1) == 7)


def test_bootstrap_ci_covers_true_mean():
    rng = np.random.default_rng(1)
    hits, reps = 0, 200
    for _ in range(reps):
        cl = np.repeat(np.arange(60), 5)
        x = 1.0 + rng.normal(size=60)[cl] + rng.normal(size=cl.size)  # correlated within clusters
        r = I.cluster_bootstrap_mean(x, cl, B=400, seed=rng)
        hits += r["lo"] <= 1.0 <= r["hi"]
    assert 0.88 <= hits / reps <= 0.99


def test_paired_bootstrap_sign_p_and_determinism():
    rng = np.random.default_rng(2)
    cl = np.repeat(np.arange(100), 3)
    bench = rng.random(300) + 0.2
    model = bench - 0.2 + 0.01 * rng.normal(size=300)
    r = I.paired_bootstrap(bench, model, cl, B=500, seed=0)
    assert r["mean"] > 0 and r["lo"] > 0 and r["p"] < 0.01
    assert r["n"] == 300 and r["n_clusters"] == 100
    same = I.paired_bootstrap(bench, bench + 0.01 * rng.normal(size=300), cl, B=500, seed=0)
    assert same["p"] > 0.01
    assert I.paired_bootstrap(bench, model, cl, B=200, seed=7) == I.paired_bootstrap(
        bench, model, cl, B=200, seed=7
    )


def test_paired_bootstrap_one_sided_p():
    rng = np.random.default_rng(5)
    cl = np.repeat(np.arange(80), 2)
    bench = rng.random(160) + 0.2
    better = I.paired_bootstrap(bench, bench - 0.05 + 0.02 * rng.normal(size=160), cl, B=400)
    worse = I.paired_bootstrap(bench, bench + 0.05 + 0.02 * rng.normal(size=160), cl, B=400)
    # model clearly better: one-sided p at its floor; clearly worse: one-sided p near 1
    assert better["p_greater"] == pytest.approx(1 / 401)
    assert worse["p_greater"] == pytest.approx(1.0)
    assert worse["p"] < 0.01  # the two-sided p does not see direction
    d = rng.normal(0.01, 0.1, 160)
    r = I.paired_bootstrap(d, np.zeros(160), cl, B=400, seed=1)
    W = I.bootstrap_weights(80, 400, np.random.default_rng(1))
    sums = np.bincount(I.cluster_codes(cl), weights=d)
    draws = (W @ sums) / (W @ np.full(80, 2.0))
    assert r["p_greater"] == pytest.approx((1 + (draws <= 0).sum()) / 401)


def test_generic_bootstrap_matches_fast_mean():
    rng = np.random.default_rng(3)
    cl = np.repeat(np.arange(40), 4)
    x = rng.normal(size=160)
    slow = I.cluster_bootstrap(np.mean, [x], cl, B=300, seed=0)
    fast = I.cluster_bootstrap_mean(x, cl, B=300, seed=0)
    assert slow["estimate"] == pytest.approx(fast["mean"])
    assert slow["se"] == pytest.approx(fast["se"], rel=0.25)


def test_dm_test_agrees_with_bootstrap():
    rng = np.random.default_rng(4)
    cl = np.repeat(np.arange(80), 4)
    d = 0.05 + rng.normal(size=80)[cl] * 0.1 + rng.normal(size=320) * 0.1
    dm = I.dm_test(d, np.zeros_like(d), cl)
    bs = I.cluster_bootstrap_mean(d, cl, B=2000, seed=0)
    assert dm["mean"] == pytest.approx(bs["mean"])
    assert dm["se"] == pytest.approx(bs["se"], rel=0.15)


def _logit_data(n, seed, cluster_size=1, rho_x=0.0, re_sd=0.0):
    rng = np.random.default_rng(seed)
    G = n // cluster_size
    cl = np.repeat(np.arange(G), cluster_size)
    x = rho_x * rng.normal(size=G)[cl] + math.sqrt(1 - rho_x**2) * rng.normal(size=n)
    u = re_sd * rng.normal(size=G)[cl]
    y = (rng.random(n) < sigmoid(-0.2 + 0.8 * x + u)).astype(int)
    return x, y, cl


def test_sandwich_close_to_naive_with_singleton_clusters():
    x, y, cl = _logit_data(20000, 5)
    fit = fit_logistic(x, y)
    se_rob = np.sqrt(np.diag(I.cluster_robust_cov(fit, y, cl)))
    se_naive = np.sqrt(np.diag(I.naive_cov(fit)))
    assert se_rob == pytest.approx(se_naive, rel=0.05)


def test_cr1_widens_with_within_cluster_correlation():
    x, y, cl = _logit_data(4000, 6, cluster_size=20, rho_x=0.9, re_sd=1.5)
    fit = fit_logistic(x, y)
    se_rob = np.sqrt(np.diag(I.cluster_robust_cov(fit, y, cl)))
    se_naive = np.sqrt(np.diag(I.naive_cov(fit)))
    assert np.all(se_rob > 1.5 * se_naive)


def test_wald():
    rows = I.wald([1.0, 0.0], np.diag([0.25, 1.0]))
    assert rows[0]["z"] == pytest.approx(2.0)
    assert rows[1]["p"] == pytest.approx(1.0)


def test_encompassing_test_both_directions():
    rng = np.random.default_rng(8)
    n = 3000
    theta = rng.normal(size=n)
    m = theta + rng.normal(0, 0.8, n)
    cl = np.repeat(np.arange(n // 3), 3)
    y = (rng.random(n) < sigmoid(theta)).astype(int)
    private = 0.5 * m + 0.5 * (theta + rng.normal(0, 0.5, n))
    r = I.encompassing_test(y, sigmoid(m), sigmoid(private), cl, eps=1e-6)
    assert r["llm_adds_info"]["p"] < 1e-3 and r["lr"]["llm"]["p"] < 1e-3
    noisy = m + rng.normal(0, 0.5, n)  # market information plus noise only
    r0 = I.encompassing_test(y, sigmoid(m), sigmoid(noisy), cl, eps=1e-6)
    assert abs(r0["z"]["llm"]) < 3  # c = 0 is true here; fixed seed, so not flaky
    assert r0["coef"]["market"] > 0.3
    assert r0["market_adds_info"]["p"] < 1e-3
    assert r0["converged"] and r0["n_clusters"] == n // 3
