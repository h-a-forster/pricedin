"""Proper scoring rules and calibration diagnostics for binary probability forecasts.

Everything takes ``y`` (0/1 outcomes) and ``p`` (YES probabilities) as 1-D arrays. Per-row
versions exist so that score differences can be bootstrapped by cluster in ``inference``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Strategy = Literal["quantile", "uniform"]

# Floor for log loss so that a forecast of exactly 0 or 1 gives a large finite loss, not inf.
# Panels are already clipped to [eps, 1 - eps] (eps = 0.01 by default) before scoring.
_LOG_FLOOR = 1e-15


def _arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y and p must be 1-D and the same shape, got {y.shape} and {p.shape}")
    return y, p


def brier_rows(y, p) -> np.ndarray:
    """Squared error per row: (p - y)^2."""
    y, p = _arrays(y, p)
    return (p - y) ** 2


def brier(y, p) -> float:
    return float(np.mean(brier_rows(y, p)))


def log_loss_rows(y, p) -> np.ndarray:
    """Negative log likelihood per row (natural log)."""
    y, p = _arrays(y, p)
    p = np.clip(p, _LOG_FLOOR, 1 - _LOG_FLOOR)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def log_loss(y, p) -> float:
    return float(np.mean(log_loss_rows(y, p)))


def brier_skill(y, p, p_ref) -> float:
    """Brier skill score 1 - B(p) / B(p_ref); positive means ``p`` beats the reference."""
    b_ref = brier(y, p_ref)
    return float("nan") if b_ref == 0 else 1.0 - brier(y, p) / b_ref


def bin_index(p, n_bins: int = 10, strategy: Strategy = "quantile") -> np.ndarray:
    """Bin label per row.

    ``quantile`` gives equal-mass bins: rows sorted by ``p`` and cut into ``n_bins`` chunks of
    (nearly) equal size, so tied forecasts may straddle a boundary. ``uniform`` gives equal-width
    bins on [0, 1].
    """
    p = np.asarray(p, dtype=float)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    if strategy == "quantile":
        out = np.empty(p.size, dtype=int)
        for k, chunk in enumerate(np.array_split(np.argsort(p, kind="stable"), n_bins)):
            out[chunk] = k
        return out
    raise ValueError(f"unknown strategy {strategy!r}")


def reliability_bins(y, p, n_bins: int = 10, strategy: Strategy = "quantile") -> dict[str, list]:
    """Reliability-diagram data: per non-empty bin the count, mean forecast, observed frequency
    and the forecast range covered."""
    y, p = _arrays(y, p)
    idx = bin_index(p, n_bins, strategy)
    out: dict[str, list] = {"count": [], "mean_p": [], "freq": [], "lo": [], "hi": []}
    for k in range(n_bins):
        m = idx == k
        if not m.any():
            continue
        out["count"].append(int(m.sum()))
        out["mean_p"].append(float(p[m].mean()))
        out["freq"].append(float(y[m].mean()))
        out["lo"].append(float(p[m].min()))
        out["hi"].append(float(p[m].max()))
    return out


def ece(y, p, n_bins: int = 10, strategy: Strategy = "quantile") -> float:
    """Expected calibration error: count-weighted mean |observed frequency - mean forecast|."""
    b = reliability_bins(y, p, n_bins, strategy)
    n = np.asarray(b["count"], dtype=float)
    if n.sum() == 0:
        return float("nan")
    gap = np.abs(np.asarray(b["freq"]) - np.asarray(b["mean_p"]))
    return float(np.sum(n * gap) / n.sum())


def murphy(y, p, n_bins: int = 10, strategy: Strategy = "quantile") -> dict[str, float]:
    """Murphy decomposition of the Brier score: ``brier ~= reliability - resolution + uncertainty``.

    Computed on bins, so the identity is exact only when the forecast is constant within each bin
    (e.g. forecasts taking at most ``n_bins`` distinct values). Otherwise the within-bin variance
    of the forecast is left over; it is reported as ``residual = brier - (rel - res + unc)``.
    """
    y, p = _arrays(y, p)
    b = reliability_bins(y, p, n_bins, strategy)
    w = np.asarray(b["count"], dtype=float) / y.size
    mp, fr = np.asarray(b["mean_p"]), np.asarray(b["freq"])
    base = float(y.mean())
    rel = float(np.sum(w * (mp - fr) ** 2))
    res = float(np.sum(w * (fr - base) ** 2))
    unc = base * (1 - base)
    bs = brier(y, p)
    return {
        "brier": bs,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "residual": bs - (rel - res + unc),
    }
