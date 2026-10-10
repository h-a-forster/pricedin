"""Out-of-sample evaluation: does pooling an LLM with the market beat the recalibrated market?

``evaluate`` fits every model on the training period, scores it on the embargoed test period,
and reports score differences with event-clustered bootstrap CIs, the forecast-encompassing test,
calibration, breakdowns by category and venue, a clustering sensitivity (series and resolution
date instead of event) and a walk-forward robustness check. The Holm-adjusted primary family is
built from these results by ``hypotheses.primary_family``. The result is a plain
JSON-serialisable dict and is deterministic given ``seed``.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .inference import dm_test, encompassing_test, paired_bootstrap
from .panel import Panel, split_at_quantile, split_masks, to_dt64, walk_forward
from .pool import DEFAULT_L2, ModelSpec, fit_predict, pool_multi, standard_models
from .scoring import brier_rows, ece, log_loss_rows, murphy, reliability_bins

ALL_MODELS = (
    "market",
    "market_recal",
    "llm",
    "llm_recal",
    "llm_extremized",
    "average",
    "pool",
    "pool_multi",
)
BENCHMARKS = ("market", "market_recal")
MIN_TRAIN = 20  # fewer training rows than this: the block is skipped, not fitted


@dataclass(frozen=True)
class _Cfg:
    eps: float
    B: int
    seed: int
    l2: float
    n_bins: int
    min_group: int
    alpha: float
    reliability: bool
    breakdowns: bool


def derive_seed(seed: int, label: str) -> int:
    """Stable per-block seed, so results don't depend on evaluation order. Every model in a
    block shares it, so their bootstrap replicates resample the same clusters."""
    return int(np.random.SeedSequence([seed, zlib.crc32(label.encode())]).generate_state(1)[0])


def evaluate(
    panel: Panel,
    forecaster_keys: Sequence[str] | None = None,
    split: float | str | datetime | np.datetime64 = 0.5,
    models: Sequence[str] | None = None,
    eps: float = 0.01,
    B: int = 2000,
    seed: int = 0,
    *,
    l2: float = DEFAULT_L2,
    n_bins: int = 10,
    min_group: int = 30,
    alpha: float = 0.05,
    include_walk_forward: bool = True,
    walk_forward_min_train: int = 100,
    include_breakdowns: bool = True,
    include_reliability: bool = True,
    include_cluster_sensitivity: bool = True,
) -> dict[str, Any]:
    """Evaluate the named models on ``panel``.

    ``split`` is a split time (ISO string / datetime) or a float in (0, 1) meaning that quantile
    of close times. ``models`` is a subset of ``ALL_MODELS`` (default: all; ``market`` and
    ``market_recal`` are always included as benchmarks; ``pool_multi`` only runs with 2+ keys).
    Results are per horizon plus ``"all"`` (horizons pooled) when there are several horizons.
    With ``include_cluster_sensitivity``, each forecaster block of an individual horizon also has
    ``cluster_sensitivity``: H1 (encompassing, full sample) and H2 (Delta Brier pool vs
    market_recal, test period) with clusters from ``Panel.sensitivity_clusters``.
    """
    keys = list(forecaster_keys) if forecaster_keys is not None else panel.keys
    wanted = list(dict.fromkeys([*BENCHMARKS, *(models or ALL_MODELS)]))
    unknown = set(wanted) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    cfg = _Cfg(eps, B, seed, l2, n_bins, min_group, alpha, include_reliability, include_breakdowns)
    split_at = (
        split_at_quantile(panel, float(split))
        if isinstance(split, float | int) and not isinstance(split, bool)
        else to_dt64(split)
    )
    horizons = sorted(set(panel.horizon.tolist()), key=_horizon_order)
    groups = horizons + (["all"] if len(horizons) > 1 else [])

    by_h: dict[str, Any] = {}
    wf: dict[str, Any] = {}
    for h in groups:
        hp = panel if h == "all" else panel.subset(panel.horizon == h)
        hp = _clip_panel(hp, eps)
        tr, te = split_masks(hp, split_at)
        train, test = hp.subset(tr), hp.subset(te)
        block: dict[str, Any] = {
            "n": len(hp),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "n_embargoed": int((~tr & ~te).sum()),
            "forecasters": {},
        }
        for k in keys:
            specs = [s for s in standard_models(k) if s.name in wanted]
            block["forecasters"][k] = _model_block(train, test, specs, cfg, f"{h}|{k}")
            fb = block["forecasters"][k]
            fb["encompassing"] = {
                "train": _encompassing(train, k, cfg),
                "full": _encompassing(hp, k, cfg),
            }
            if include_cluster_sensitivity and h != "all":
                fb["cluster_sensitivity"] = _cluster_sensitivity(hp, train, test, k, cfg, h)
        if "pool_multi" in wanted and len(keys) > 1:
            specs = [s for s in standard_models(keys[0]) if s.name in BENCHMARKS]
            specs.append(pool_multi(keys))
            block["pool_multi"] = _model_block(train, test, specs, cfg, f"{h}|pool_multi")
        by_h[h] = block
        if include_walk_forward:
            wf[h] = {k: _walk_forward_block(hp, k, cfg, walk_forward_min_train) for k in keys}

    result: dict[str, Any] = {
        "meta": {
            "n_rows": len(panel),
            "forecaster_keys": keys,
            "models": wanted,
            "split_at": str(split_at) + "Z",
            "eps": eps,
            "B": B,
            "seed": seed,
            "l2": l2,
            "n_bins": n_bins,
            "alpha": alpha,
            "horizons": horizons,
            "panel_dropped": dict(panel.dropped),
            "delta": "loss(benchmark) - loss(model); positive = model better",
        },
        "by_horizon": by_h,
    }
    if include_walk_forward:
        result["walk_forward"] = wf
    return _jsonable(result)


def _horizon_order(h: str) -> tuple[int, str]:
    order = ["7d", "1d", "1h"]
    return (order.index(h), h) if h in order else (len(order), h)


def _clip_panel(p: Panel, eps: float) -> Panel:
    p = p.subset(np.arange(len(p)))
    p.market = np.clip(p.market, eps, 1 - eps)
    p.forecasts = {k: np.clip(v, eps, 1 - eps) for k, v in p.forecasts.items()}
    return p


def _skip_reason(train: Panel, test: Panel) -> str | None:
    if len(train) < MIN_TRAIN:
        return f"fewer than {MIN_TRAIN} training rows"
    if np.unique(train.y).size < 2:
        return "training outcomes are all one class"
    if len(test) == 0:
        return "no test rows"
    return None


def _model_block(
    train: Panel, test: Panel, specs: list[ModelSpec], cfg: _Cfg, label: str
) -> dict[str, Any]:
    reason = _skip_reason(train, test)
    if reason:
        return {"skipped": reason, "n_train": len(train), "n_test": len(test)}
    preds = fit_predict(specs, train.columns(), train.y, test.columns(), eps=cfg.eps, l2=cfg.l2)
    probs = {name: p for name, (p, _) in preds.items()}
    s = derive_seed(cfg.seed, label)
    out: dict[str, Any] = {
        "n_train": len(train),
        "n_test": len(test),
        "n_test_clusters": int(np.unique(test.event_id).size),
        "models": {},
    }
    for name, (p, fm) in preds.items():
        out["models"][name] = _score_model(test.y, p, probs, test.event_id, cfg, s) | {
            "coef": fm.coef or None,
            "converged": None if fm.fit is None else fm.fit.converged,
        }
    if cfg.breakdowns:
        out["breakdowns"] = {
            "category": _breakdown(test, probs, test.category, cfg, s),
            "venue": _breakdown(test, probs, test.venue, cfg, s),
        }
    return out


def _score_model(y, p, probs, clusters, cfg: _Cfg, seed: int) -> dict[str, Any]:
    br, ll = brier_rows(y, p), log_loss_rows(y, p)
    out: dict[str, Any] = {
        "n": int(y.size),
        "brier": float(br.mean()),
        "log_loss": float(ll.mean()),
        "ece": ece(y, p, cfg.n_bins),
        "murphy": murphy(y, p, cfg.n_bins),
    }
    if cfg.reliability:
        out["reliability"] = reliability_bins(y, p, cfg.n_bins)
    for bench in BENCHMARKS:
        q = probs[bench]
        out[f"delta_vs_{bench}"] = {
            "brier": paired_bootstrap(
                brier_rows(y, q), br, clusters, B=cfg.B, seed=seed, alpha=cfg.alpha
            ),
            "log_loss": paired_bootstrap(
                log_loss_rows(y, q), ll, clusters, B=cfg.B, seed=seed, alpha=cfg.alpha
            ),
            "dm_brier": dm_test(brier_rows(y, q), br, clusters),
        }
    return out


def _breakdown(test: Panel, probs, labels, cfg: _Cfg, seed: int) -> dict[str, Any]:
    out = {}
    for g in sorted(set(labels.tolist())):
        m = labels == g
        y, cl = test.y[m], test.event_id[m]
        row: dict[str, Any] = {
            "n": int(m.sum()),
            "n_clusters": int(np.unique(cl).size),
            "small": bool(m.sum() < cfg.min_group),
            "brier": {name: float(brier_rows(y, p[m]).mean()) for name, p in probs.items()},
        }
        model = "pool" if "pool" in probs else ("pool_multi" if "pool_multi" in probs else None)
        if model:
            row[f"delta_{model}_vs_market_recal"] = paired_bootstrap(
                brier_rows(y, probs["market_recal"][m]),
                brier_rows(y, probs[model][m]),
                cl,
                B=cfg.B,
                seed=seed,
                alpha=cfg.alpha,
            )
        out[str(g)] = row
    return out


def _encompassing(p: Panel, key: str, cfg: _Cfg, clusters=None) -> dict[str, Any]:
    if len(p) < MIN_TRAIN or np.unique(p.y).size < 2:
        return {"skipped": "too few rows or one outcome class", "n": len(p)}
    cl = p.event_id if clusters is None else clusters
    return encompassing_test(p.y, p.market, p.forecasts[key], cl, eps=cfg.eps, l2=cfg.l2)


def _cluster_sensitivity(
    hp: Panel, train: Panel, test: Panel, key: str, cfg: _Cfg, h: str
) -> dict[str, Any]:
    """H1 and H2 with each alternative clustering. Same fits as the primary tests; only the
    clusters (and so the errors, CIs and p-values) change."""
    probs = None
    if _skip_reason(train, test) is None:
        specs = [s for s in standard_models(key) if s.name in ("market_recal", "pool")]
        fp = fit_predict(specs, train.columns(), train.y, test.columns(), eps=cfg.eps, l2=cfg.l2)
        probs = {name: p for name, (p, _) in fp.items()}
    seed = derive_seed(cfg.seed, f"{h}|{key}")
    full_cl, test_cl = hp.sensitivity_clusters(), test.sensitivity_clusters()
    out: dict[str, Any] = {}
    for name, cl in full_cl.items():
        e = _encompassing(hp, key, cfg, clusters=cl)
        h1 = (
            e
            if "skipped" in e
            else {
                "c": e["coef"]["llm"],
                "z": e["llm_adds_info"]["z"],
                "p": e["llm_adds_info"]["p"],
                "n_clusters": e["n_clusters"],
            }
        )
        h2: dict[str, Any] = {"skipped": "pool not fitted"}
        if probs is not None:
            h2 = paired_bootstrap(
                brier_rows(test.y, probs["market_recal"]),
                brier_rows(test.y, probs["pool"]),
                test_cl[name],
                B=cfg.B,
                seed=seed,
                alpha=cfg.alpha,
            )
        out[name] = {"h1": h1, "h2": h2}
    return out


def _walk_forward_block(hp: Panel, key: str, cfg: _Cfg, min_train: int) -> dict[str, Any]:
    """Out-of-fold predictions from monthly expanding-window refits, scored together."""
    specs = [s for s in standard_models(key) if s.name in ("market", "market_recal", "pool")]
    folds = walk_forward(hp, min_train=max(min_train, MIN_TRAIN))
    ys, cls, preds = [], [], {s.name: [] for s in specs}
    fold_info = []
    for f in folds:
        train, test = hp.subset(f.train), hp.subset(f.test)
        if _skip_reason(train, test):
            continue
        fp = fit_predict(specs, train.columns(), train.y, test.columns(), eps=cfg.eps, l2=cfg.l2)
        for name, (p, _) in fp.items():
            preds[name].append(p)
        ys.append(test.y)
        cls.append(test.event_id)
        fold_info.append({"start": str(f.start) + "Z", "n_train": len(train), "n_test": len(test)})
    if not ys:
        return {"skipped": "no usable folds", "n_folds": 0}
    y, cl = np.concatenate(ys), np.concatenate(cls)
    probs = {k: np.concatenate(v) for k, v in preds.items()}
    s = derive_seed(cfg.seed, f"wf|{key}|{len(y)}")
    models = {}
    for name, p in probs.items():
        br = brier_rows(y, p)
        models[name] = {
            "brier": float(br.mean()),
            "log_loss": float(log_loss_rows(y, p).mean()),
            "delta_vs_market_recal": {
                "brier": paired_bootstrap(
                    brier_rows(y, probs["market_recal"]), br, cl, B=cfg.B, seed=s, alpha=cfg.alpha
                )
            },
        }
    return {"n_folds": len(fold_info), "n_test": int(y.size), "folds": fold_info, "models": models}


def _jsonable(obj: Any) -> Any:
    """numpy scalars/arrays -> Python; NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj
