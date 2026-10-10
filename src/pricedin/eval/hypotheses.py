"""The pre-registered primary family and the clustering sensitivity.

Both read an evaluation dict shaped like ``evaluation.json``: ``{"keys": {key: result}}``, where
``key`` is ``forecaster|condition`` and ``result`` is what ``evaluate`` returns for that key.
This module is the single source of truth for the CLI (``evaluation.json``) and the report.

- **H1, encompassing.** c in logit(y) = a + b·logit(market) + c·logit(LLM) on the full sample,
  CR1 errors clustered by event. Wald test of c = 0, two-sided.
- **H2, out-of-sample gain.** Delta = Brier(market_recal) - Brier(pool) on the test period,
  event-clustered bootstrap. One-sided: H1 is Delta > 0.

The family is every (key x horizon x {H1, H2}) test, Holm-adjusted together. The pooled ``all``
horizon is excluded: it re-uses the same rows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .evaluate import _horizon_order
from .inference import holm

TESTS = ("h1", "h2")
CLUSTERINGS = ("event", "series", "resolution_date")  # event is the primary clustering


def _ok(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def _horizons(res: dict[str, Any]) -> list[str]:
    return sorted(res["meta"]["horizons"], key=_horizon_order)


def _fblock(res: dict[str, Any], key: str, h: str) -> dict[str, Any]:
    return res.get("by_horizon", {}).get(h, {}).get("forecasters", {}).get(key, {})


def _h2(fb: dict[str, Any]) -> dict[str, Any]:
    pool = fb.get("models", {}).get("pool", {})
    return pool.get("delta_vs_market_recal", {}).get("brier") or {}


def _h1(fb: dict[str, Any]) -> dict[str, Any]:
    return fb.get("encompassing", {}).get("full", {})


def _alpha(evaluation: dict[str, Any]) -> float:
    for res in evaluation["keys"].values():
        return float(res["meta"].get("alpha", 0.05))
    return 0.05


def primary_family(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per key x horizon with H1 and H2, Holm-adjusted as one family over all
    2 x keys x horizons tests. ``p_h1`` is two-sided (c != 0); ``p_h2`` is one-sided
    (Delta > 0). Missing p-values are left out of the adjustment."""
    alpha = _alpha(evaluation)
    rows = []
    for key, res in evaluation["keys"].items():
        for h in _horizons(res):
            fb = _fblock(res, key, h)
            d, e = _h2(fb), _h1(fb)
            rows.append(
                {
                    "key": key,
                    "horizon": h,
                    "n_test": fb.get("n_test"),
                    "delta": d.get("mean"),
                    "lo": d.get("lo"),
                    "hi": d.get("hi"),
                    "p_h2": d.get("p_greater"),
                    "c": e.get("coef", {}).get("llm"),
                    "c_z": e.get("llm_adds_info", {}).get("z"),
                    "p_h1": e.get("llm_adds_info", {}).get("p"),
                    "b": e.get("coef", {}).get("market"),
                }
            )
    flat = [(r, t) for r in rows for t in TESTS]
    p = np.array([r[f"p_{t}"] if _ok(r[f"p_{t}"]) else np.nan for r, t in flat], float)
    adj = holm(p) if len(p) else p
    for (r, t), a in zip(flat, adj, strict=True):
        r[f"p_{t}_holm"] = None if np.isnan(a) else float(a)
    for r in rows:
        r["reject_h1"] = bool(r["p_h1_holm"] is not None and r["p_h1_holm"] < alpha)
        r["reject_h2"] = bool(
            r["p_h2_holm"] is not None and r["p_h2_holm"] < alpha and (r["delta"] or 0) > 0
        )
    return rows


def cluster_sensitivity(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """H1 and H2 per key x horizon under each clustering in ``CLUSTERINGS``. Unadjusted
    p-values (two-sided for H1, one-sided for H2); not part of the Holm family. The ``event``
    rows repeat the primary tests for comparison."""
    rows = []
    for key, res in evaluation["keys"].items():
        for h in _horizons(res):
            fb = _fblock(res, key, h)
            alt = fb.get("cluster_sensitivity", {})
            for c in CLUSTERINGS:
                if c == "event":
                    e, d = _h1(fb), _h2(fb)
                    h1 = {
                        "c": e.get("coef", {}).get("llm"),
                        "z": e.get("llm_adds_info", {}).get("z"),
                        "p": e.get("llm_adds_info", {}).get("p"),
                        "n_clusters": e.get("n_clusters"),
                    }
                elif c in alt:
                    h1, d = alt[c].get("h1") or {}, alt[c].get("h2") or {}
                else:
                    continue
                rows.append(
                    {
                        "key": key,
                        "horizon": h,
                        "clustering": c,
                        "c": h1.get("c"),
                        "c_z": h1.get("z"),
                        "p_h1": h1.get("p"),
                        "n_clusters_h1": h1.get("n_clusters"),
                        "delta": d.get("mean"),
                        "lo": d.get("lo"),
                        "hi": d.get("hi"),
                        "p_h2": d.get("p_greater"),
                        "n_clusters_h2": d.get("n_clusters"),
                    }
                )
    return rows
