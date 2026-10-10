"""Logistic recalibration and pooling of market and LLM probabilities.

Every fitted model is a logistic regression on log-odds, fit on a training period and applied to a
later test period. ``market_recal`` is the fair benchmark: it absorbs the market's systematic
miscalibration (e.g. favorite-longshot bias), so a pool that beats it is using the LLM's
information, not just fixing the market's calibration.

The fitter is plain Newton/IRLS with step halving and an optional small L2 penalty on the slopes
(never the intercept), which keeps coefficients finite under perfect separation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Default ridge on slopes for the named models: negligible at the sample sizes we use (hundreds of
# rows, objective summed over rows) but enough to stop divergence under separation.
DEFAULT_L2 = 1e-3

Kind = Literal["raw", "recal", "extremize", "average", "pool"]


def clip(p, eps: float) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p, eps: float = 0.01) -> np.ndarray:
    """Log-odds after clipping to [eps, 1 - eps]."""
    q = clip(p, eps)
    return np.log(q) - np.log1p(-q)


def sigmoid(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0.0, -x))


@dataclass
class LogitFit:
    """Result of ``fit_logistic``.

    ``hessian`` is the Hessian of the penalized negative log-likelihood at the solution
    (X'WX + l2 * diag(penalty mask)); its inverse is the naive covariance and the sandwich bread.
    ``loglik`` is the unpenalized log-likelihood.
    """

    coef: np.ndarray
    prob: np.ndarray
    loglik: float
    hessian: np.ndarray
    design: np.ndarray  # the X actually used, including the intercept column if any
    penalty: np.ndarray  # per-coefficient L2 weight (0 for the intercept)
    converged: bool
    n_iter: int


def _loglik(y: np.ndarray, eta: np.ndarray) -> float:
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _solve(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(h, g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(h, g, rcond=None)[0]


def fit_logistic(
    X,
    y,
    *,
    intercept: bool = True,
    l2: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> LogitFit:
    """Maximum-likelihood logistic regression by Newton's method with step halving.

    ``X`` is (n, k) without an intercept column; one is prepended when ``intercept``. ``l2``
    penalizes 0.5 * l2 * sum(slope^2). Returns ``converged=False`` if the iteration cap is hit
    (typical under separation with ``l2=0``); the coefficients are then the last iterate.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    k = X.shape[1]
    pen = np.full(k, float(l2))
    if intercept:
        pen[0] = 0.0
    beta = np.zeros(k)

    def objective(b: np.ndarray) -> float:
        return -_loglik(y, X @ b) + 0.5 * float(np.sum(pen * b * b))

    f = objective(beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 - n_iter is reported
        p = sigmoid(X @ beta)
        grad = -X.T @ (y - p) + pen * beta
        w = p * (1 - p)
        hess = (X * w[:, None]).T @ X + np.diag(pen)
        step = _solve(hess, grad)
        t = 1.0
        while True:
            cand = beta - t * step
            f_new = objective(cand)
            if f_new <= f + 1e-12 * (1 + abs(f)) or t < 1e-10:
                break
            t *= 0.5
        if f_new > f + 1e-12 * (1 + abs(f)):  # no descent possible: at the numerical optimum
            converged = bool(np.max(np.abs(grad)) < 1e-6 * (1 + abs(f)))
            break
        moved = float(np.max(np.abs(t * step)))
        decrease = f - f_new
        beta, f = cand, f_new
        if moved < 1e-8 or (0 <= decrease < tol * (1 + abs(f)) and moved < 1e-4):
            converged = True
            break

    eta = X @ beta
    p = sigmoid(eta)
    w = p * (1 - p)
    hess = (X * w[:, None]).T @ X + np.diag(pen)
    return LogitFit(beta, p, _loglik(y, eta), hess, X, pen, converged, it)


# --- named models -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A forecast model built from probability columns.

    kinds: ``raw`` (the column itself, no fit), ``recal`` sigma(a + b*logit(col)),
    ``extremize`` sigma(k*logit(col)), ``average`` sigma(mean of logits, no fit),
    ``pool`` sigma(a + sum_j b_j*logit(col_j)).
    """

    name: str
    kind: Kind
    columns: tuple[str, ...]


@dataclass
class FittedModel:
    spec: ModelSpec
    eps: float
    fit: LogitFit | None = None
    coef: dict[str, float] = field(default_factory=dict)

    def predict(self, cols: Mapping[str, np.ndarray]) -> np.ndarray:
        """Probabilities on new rows (clipped to [eps, 1 - eps])."""
        s = self.spec
        if s.kind == "raw":
            return clip(cols[s.columns[0]], self.eps)
        L = np.column_stack([logit(cols[c], self.eps) for c in s.columns])
        if s.kind == "average":
            return clip(sigmoid(L.mean(axis=1)), self.eps)
        assert self.fit is not None
        b = self.fit.coef
        eta = L @ b if s.kind == "extremize" else b[0] + L @ b[1:]
        return clip(sigmoid(eta), self.eps)


def coef_names(spec: ModelSpec) -> list[str]:
    if spec.kind == "extremize":
        return ["k"]
    return ["intercept"] + [f"logit({c})" for c in spec.columns]


def fit_model(
    spec: ModelSpec,
    cols: Mapping[str, np.ndarray],
    y,
    *,
    eps: float = 0.01,
    l2: float = DEFAULT_L2,
) -> FittedModel:
    """Fit ``spec`` on training columns (no-op for ``raw`` and ``average``)."""
    if spec.kind in ("raw", "average"):
        return FittedModel(spec, eps)
    L = np.column_stack([logit(cols[c], eps) for c in spec.columns])
    fit = fit_logistic(L, y, intercept=spec.kind != "extremize", l2=l2)
    coef = dict(zip(coef_names(spec), (float(v) for v in fit.coef), strict=True))
    return FittedModel(spec, eps, fit, coef)


def fit_predict(
    specs: Sequence[ModelSpec],
    train_cols: Mapping[str, np.ndarray],
    y_train,
    test_cols: Mapping[str, np.ndarray],
    *,
    eps: float = 0.01,
    l2: float = DEFAULT_L2,
) -> dict[str, tuple[np.ndarray, FittedModel]]:
    """Fit each spec on train and return ``{name: (test predictions, fitted model)}``."""
    out = {}
    for s in specs:
        fm = fit_model(s, train_cols, y_train, eps=eps, l2=l2)
        out[s.name] = (fm.predict(test_cols), fm)
    return out


def standard_models(llm: str, market: str = "market") -> list[ModelSpec]:
    """The named model set for one LLM column: market, market_recal, llm, llm_recal,
    llm_extremized, average, pool."""
    return [
        ModelSpec("market", "raw", (market,)),
        ModelSpec("market_recal", "recal", (market,)),
        ModelSpec("llm", "raw", (llm,)),
        ModelSpec("llm_recal", "recal", (llm,)),
        ModelSpec("llm_extremized", "extremize", (llm,)),
        ModelSpec("average", "average", (market, llm)),
        ModelSpec("pool", "pool", (market, llm)),
    ]


def pool_multi(llms: Sequence[str], market: str = "market") -> ModelSpec:
    """Pool of the market and several LLM columns."""
    return ModelSpec("pool_multi", "pool", (market, *llms))
