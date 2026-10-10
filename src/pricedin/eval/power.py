"""Synthetic world for power analysis and end-to-end validation of the evaluation pipeline.

Truth: theta = event effect + question effect, y ~ Bernoulli(sigmoid(theta)).
Market: sees m = theta + N(0, sigma_m) and quotes logit k_m * m (k_m < 1 compresses prices
toward 0.5, a favorite-longshot-style distortion that ``market_recal`` should absorb).
LLM: logit (1 - w) * m + w * s + N(0, sigma_l), with s = theta + N(0, sigma_s) a private signal.
With w = 0 the LLM only re-encodes the market's information plus noise, so the market
encompasses it (true pool coefficient c = 0); with w > 0 it carries information the market lacks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .evaluate import evaluate
from .panel import Panel
from .pool import sigmoid

SIM_KEY = "sim|base"
_HOURS = {"7d": 7 * 24, "1d": 24, "1h": 1}


def simulate_panel(
    n_questions: int,
    w: float,
    *,
    questions_per_event: int = 3,
    s_event: float = 1.0,
    s_question: float = 1.0,
    sigma_m: float = 0.7,
    k_m: float = 1.0,
    sigma_s: float = 0.7,
    sigma_l: float = 0.5,
    horizons: Sequence[str] = ("1d",),
    span_days: int = 365,
    eps: float = 0.01,
    seed: int | np.random.Generator | None = 0,
) -> Panel:
    """Simulate a panel with one forecaster column ``SIM_KEY``.

    Questions are grouped into events of ``questions_per_event`` that share an event effect and a
    close time (spread uniformly over ``span_days`` from 2025-01-01). Each horizon gets its own
    noise draws for market and LLM; the outcome is shared across horizons.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n_events = -(-n_questions // questions_per_event)
    ev = np.repeat(np.arange(n_events), questions_per_event)[:n_questions]
    theta = rng.normal(0, s_event, n_events)[ev] + rng.normal(0, s_question, n_questions)
    y = (rng.random(n_questions) < sigmoid(theta)).astype(int)
    t0 = np.datetime64("2025-01-01T00:00:00", "s")
    close_ev = t0 + (rng.random(n_events) * span_days * 86400).astype("int64").astype(
        "timedelta64[s]"
    )
    close = close_ev[ev]

    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("h", "m", "l", "as_of")}
    for h in horizons:
        m = theta + rng.normal(0, sigma_m, n_questions)
        s = theta + rng.normal(0, sigma_s, n_questions)
        llm = (1 - w) * m + w * s + rng.normal(0, sigma_l, n_questions)
        parts["h"].append(np.full(n_questions, h, dtype=object))
        parts["m"].append(sigmoid(k_m * m))
        parts["l"].append(sigmoid(llm))
        parts["as_of"].append(close - np.timedelta64(_HOURS.get(h, 24), "h"))

    reps = len(horizons)
    qid = np.array([f"sim:{i}" for i in range(n_questions)], dtype=object)
    return Panel(
        question_id=np.tile(qid, reps),
        horizon=np.concatenate(parts["h"]),
        y=np.tile(y, reps),
        market=np.clip(np.concatenate(parts["m"]), eps, 1 - eps),
        forecasts={SIM_KEY: np.clip(np.concatenate(parts["l"]), eps, 1 - eps)},
        event_id=np.tile(np.array([f"ev:{e}" for e in ev], dtype=object), reps),
        category=np.tile(np.array(["other"] * n_questions, dtype=object), reps),
        venue=np.tile(np.array(["kalshi"] * n_questions, dtype=object), reps),
        close_at=np.tile(close, reps),
        as_of=np.concatenate(parts["as_of"]),
        eps=eps,
    )


def run_once(panel: Panel, *, B: int = 500, seed: int = 0, split: float = 0.5) -> dict[str, Any]:
    """One pass of the real ``evaluate`` pipeline (pool vs market_recal only), reduced to the
    numbers a power analysis needs."""
    res = evaluate(
        panel,
        [SIM_KEY],
        split=split,
        models=["pool"],
        eps=panel.eps,
        B=B,
        seed=seed,
        include_walk_forward=False,
        include_breakdowns=False,
        include_reliability=False,
    )
    h = res["meta"]["horizons"][0] if len(res["meta"]["horizons"]) == 1 else "all"
    blk = res["by_horizon"][h]["forecasters"][SIM_KEY]
    pool = blk.get("models", {}).get("pool")
    d = pool["delta_vs_market_recal"]["brier"] if pool else {}
    enc = blk["encompassing"]
    return {
        "delta_brier": d.get("mean"),
        "delta_p": d.get("p"),
        "enc_full_p": enc["full"].get("llm_adds_info", {}).get("p"),
        "enc_train_p": enc["train"].get("llm_adds_info", {}).get("p"),
        "enc_full_c": enc["full"].get("coef", {}).get("llm"),
    }


def power_curve(
    n_questions_list: Sequence[int],
    w_list: Sequence[float],
    reps: int = 100,
    *,
    alpha: float = 0.05,
    B: int = 500,
    seed: int = 0,
    split: float = 0.5,
    **sim_kwargs: Any,
) -> list[dict[str, Any]]:
    """Rejection rates over ``reps`` simulated panels for each (n_questions, w).

    ``reject_encompassing_full`` / ``_train``: Wald test of c = 0 at ``alpha`` on all rows /
    training rows. ``reject_gain``: out-of-sample Delta Brier(pool vs market_recal) > 0 with
    bootstrap p < alpha. At w = 0 these are false-positive rates.
    """
    out = []
    root = np.random.SeedSequence(seed)
    cells = [(n, w) for n in n_questions_list for w in w_list]
    for (n, w), ss in zip(cells, root.spawn(len(cells)), strict=True):
        runs = []
        for child in ss.spawn(reps):
            rng = np.random.default_rng(child)
            panel = simulate_panel(n, w, seed=rng, **sim_kwargs)
            runs.append(run_once(panel, B=B, seed=int(rng.integers(2**31)), split=split))

        def rate(pred, runs=runs) -> float:
            return float(np.mean([pred(r) for r in runs]))

        out.append(
            {
                "n_questions": n,
                "w": w,
                "reps": reps,
                "alpha": alpha,
                "reject_encompassing_full": rate(
                    lambda r: r["enc_full_p"] is not None and r["enc_full_p"] < alpha
                ),
                "reject_encompassing_train": rate(
                    lambda r: r["enc_train_p"] is not None and r["enc_train_p"] < alpha
                ),
                "reject_gain": rate(
                    lambda r: (
                        r["delta_p"] is not None and r["delta_p"] < alpha and r["delta_brier"] > 0
                    )
                ),
                "mean_delta_brier": float(
                    np.mean([r["delta_brier"] for r in runs if r["delta_brier"] is not None])
                ),
            }
        )
    return out
