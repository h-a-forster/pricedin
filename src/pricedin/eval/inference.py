"""Uncertainty for score differences and pooled coefficients, clustered by event.

Questions in one event share news and often outcomes, so rows are not independent. Every interval
and test here resamples or aggregates by cluster (event): a cluster bootstrap for score
differences, a CR1 sandwich covariance for the logistic pool, and a Diebold-Mariano-style z test
on cluster sums as a cross-check on the bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .pool import DEFAULT_L2, LogitFit, fit_logistic, logit


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(z: float) -> float:
    """Two-sided normal p-value."""
    if not math.isfinite(z):
        return 0.0 if abs(z) == math.inf else float("nan")
    return math.erfc(abs(z) / math.sqrt(2.0))


def chi2_1_sf(x: float) -> float:
    """Survival function of chi-square with 1 degree of freedom."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def cluster_codes(clusters) -> np.ndarray:
    """Integer codes 0..G-1 for arbitrary cluster labels."""
    return np.unique(np.asarray(clusters), return_inverse=True)[1].ravel()


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def bootstrap_weights(n_clusters: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """(B, G) matrix: how many times each cluster is drawn in each bootstrap replicate."""
    draws = rng.integers(0, n_clusters, size=(B, n_clusters))
    flat = (draws + n_clusters * np.arange(B)[:, None]).ravel()
    return np.bincount(flat, minlength=B * n_clusters).reshape(B, n_clusters).astype(float)


def cluster_bootstrap_mean(
    x,
    clusters,
    *,
    B: int = 2000,
    seed: int | np.random.Generator | None = 0,
    alpha: float = 0.05,
    null: float = 0.0,
) -> dict[str, Any]:
    """Cluster bootstrap of the row mean of ``x``.

    Returns the mean, percentile CI, bootstrap SE and a two-sided p-value for ``mean == null``
    computed by centring the bootstrap distribution on the null.
    """
    x = np.asarray(x, dtype=float)
    codes = cluster_codes(clusters)
    G = int(codes.max()) + 1 if x.size else 0
    est = float(x.mean()) if x.size else float("nan")
    out: dict[str, Any] = {"mean": est, "n": int(x.size), "n_clusters": G}
    if G < 2:
        return out | {"lo": float("nan"), "hi": float("nan"), "se": float("nan"), "p": None}
    sums = np.bincount(codes, weights=x, minlength=G)
    counts = np.bincount(codes, minlength=G).astype(float)
    W = bootstrap_weights(G, B, _rng(seed))
    draws = (W @ sums) / (W @ counts)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    extreme = np.abs(draws - est) >= abs(est - null)
    p = (1 + int(extreme.sum())) / (B + 1)
    return out | {"lo": float(lo), "hi": float(hi), "se": float(draws.std(ddof=1)), "p": p}


def paired_bootstrap(
    loss_benchmark,
    loss_model,
    clusters,
    *,
    B: int = 2000,
    seed: int | np.random.Generator | None = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster bootstrap of Delta = loss(benchmark) - loss(model); positive means the model is
    better. Keys: mean, lo, hi, se, p (two-sided, H0: Delta = 0), n, n_clusters."""
    d = np.asarray(loss_benchmark, dtype=float) - np.asarray(loss_model, dtype=float)
    return cluster_bootstrap_mean(d, clusters, B=B, seed=seed, alpha=alpha)


def cluster_bootstrap(
    stat: Callable[..., float],
    arrays: Sequence[np.ndarray],
    clusters,
    *,
    B: int = 2000,
    seed: int | np.random.Generator | None = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster bootstrap of any statistic ``stat(*arrays)`` of per-row arrays.

    Each replicate draws G clusters with replacement and concatenates their rows. Slower than
    ``cluster_bootstrap_mean``; use it for non-linear statistics (ECE, skill scores, ...).
    """
    arrays = [np.asarray(a) for a in arrays]
    codes = cluster_codes(clusters)
    G = int(codes.max()) + 1
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(G + 1))
    members = [order[starts[g] : starts[g + 1]] for g in range(G)]
    rng = _rng(seed)
    est = float(stat(*arrays))
    draws = np.empty(B)
    for b in range(B):
        idx = np.concatenate([members[g] for g in rng.integers(0, G, size=G)])
        draws[b] = stat(*(a[idx] for a in arrays))
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": est,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(np.nanstd(draws, ddof=1)),
        "n": int(codes.size),
        "n_clusters": G,
    }


def dm_test(loss_benchmark, loss_model, clusters) -> dict[str, Any]:
    """Diebold-Mariano-style test of mean Delta = loss(benchmark) - loss(model) = 0 with a
    cluster-robust (CR1) standard error built from cluster sums of demeaned differences."""
    d = np.asarray(loss_benchmark, dtype=float) - np.asarray(loss_model, dtype=float)
    codes = cluster_codes(clusters)
    G, n = int(codes.max()) + 1, d.size
    mean = float(d.mean())
    if G < 2:
        return {"mean": mean, "se": float("nan"), "z": float("nan"), "p": None, "n": n}
    s = np.bincount(codes, weights=d - mean, minlength=G)
    se = math.sqrt(G / (G - 1) * float(np.sum(s**2))) / n
    z = mean / se if se > 0 else (0.0 if mean == 0 else math.copysign(math.inf, mean))
    return {"mean": mean, "se": se, "z": z, "p": two_sided_p(z), "n": n, "n_clusters": G}


def cluster_robust_cov(fit: LogitFit, y, clusters) -> np.ndarray:
    """CR1 sandwich covariance for a (possibly lightly penalized) logistic fit.

    bread = inverse penalized Hessian; meat = sum over clusters of outer products of summed
    per-row estimating-function contributions; small-sample factor G/(G-1) * (N-1)/(N-K).
    With singleton clusters this is the usual heteroskedasticity-robust (HC1) covariance.
    """
    X = fit.design
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    psi = X * (y - fit.prob)[:, None] - (fit.penalty * fit.coef)[None, :] / n
    codes = cluster_codes(clusters)
    G = int(codes.max()) + 1
    S = np.zeros((G, k))
    np.add.at(S, codes, psi)
    bread = np.linalg.pinv(fit.hessian)
    factor = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else float("nan")
    return factor * bread @ (S.T @ S) @ bread


def naive_cov(fit: LogitFit) -> np.ndarray:
    """Model-based covariance: inverse Hessian (assumes independent rows)."""
    return np.linalg.pinv(fit.hessian)


def wald(coef, cov) -> list[dict[str, float]]:
    """Per-coefficient Wald z and two-sided p (H0: coefficient = 0)."""
    coef = np.asarray(coef, dtype=float)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    out = []
    for b, s in zip(coef, se, strict=True):
        z = b / s if s > 0 else float("nan")
        out.append({"coef": float(b), "se": float(s), "z": float(z), "p": two_sided_p(z)})
    return out


def encompassing_test(
    y,
    p_market,
    p_llm,
    clusters,
    *,
    eps: float = 0.01,
    l2: float = DEFAULT_L2,
) -> dict[str, Any]:
    """Forecast-encompassing test on the given rows.

    Fits logit P(y=1) = a + b*logit(p_market) + c*logit(p_llm) with CR1 cluster-robust SEs.
    ``llm_adds_info`` tests H0: c = 0 (the market encompasses the LLM); ``market_adds_info``
    tests H0: b = 0 (the LLM encompasses the market). ``lr`` holds the matching likelihood-ratio
    statistics against the one-forecast recalibrations; they assume independent rows, so treat
    them as a cross-check on the cluster-robust Wald tests.
    """
    y = np.asarray(y, dtype=float)
    lm, ll = logit(p_market, eps), logit(p_llm, eps)
    full = fit_logistic(np.column_stack([lm, ll]), y, l2=l2)
    only_m = fit_logistic(lm, y, l2=l2)
    only_l = fit_logistic(ll, y, l2=l2)
    cov = cluster_robust_cov(full, y, clusters)
    names = ["intercept", "market", "llm"]
    rows = dict(zip(names, wald(full.coef, cov), strict=True))
    lr_c = max(0.0, 2 * (full.loglik - only_m.loglik))
    lr_b = max(0.0, 2 * (full.loglik - only_l.loglik))
    return {
        "n": int(y.size),
        "n_clusters": int(cluster_codes(clusters).max()) + 1 if y.size else 0,
        "converged": bool(full.converged),
        "coef": {k: v["coef"] for k, v in rows.items()},
        "se": {k: v["se"] for k, v in rows.items()},
        "z": {k: v["z"] for k, v in rows.items()},
        "p": {k: v["p"] for k, v in rows.items()},
        "llm_adds_info": {"z": rows["llm"]["z"], "p": rows["llm"]["p"]},
        "market_adds_info": {"z": rows["market"]["z"], "p": rows["market"]["p"]},
        "lr": {
            "llm": {"stat": lr_c, "p": chi2_1_sf(lr_c)},
            "market": {"stat": lr_b, "p": chi2_1_sf(lr_b)},
        },
        "loglik": {"pool": full.loglik, "market_recal": only_m.loglik, "llm_recal": only_l.loglik},
    }


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input; NaN entries are ignored
    and stay NaN)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out
