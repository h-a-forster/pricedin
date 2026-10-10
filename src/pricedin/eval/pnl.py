"""Trading backtest: if forecasts beat the market statistically, is there money in it after costs?

For each test row (question x horizon) a model probability ``q`` is turned into at most one taker
order at the snapshot quotes, held to resolution (payoff 1 per winning contract):

- buy YES at the ask if ``q - ask - fee/contract > threshold``;
- buy NO at ``1 - bid`` (i.e. sell YES at the bid) if ``(1 - q) - (1 - bid) - fee/contract >
  threshold``; otherwise no trade.

Fees are charged per order on the actual contract count (``fees.order_fee``), so Kalshi's
round-up-to-the-cent makes small orders expensive. Rows with a known order book (live snapshots)
walk the book for the fill price and are capped by its depth; historical rows trade at the top of
book with unknown capacity. Rows without bid/ask (Polymarket price history) get an assumed spread
around the mid and are flagged ``approximate``.

The same rules run for every model, so the LLM's incremental value is P&L(pool) - P&L(market_recal):
``market_recal`` already trades on the market's known miscalibration (favorite-longshot bias).
Thresholds and sizing are fixed inputs; nothing is tuned on the test rows.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np

from ..schema import Question, Snapshot
from .evaluate import _jsonable, derive_seed
from .fees import KALSHI_PRECISION, order_fee, raw_fee_per_contract, raw_order_fee
from .inference import cluster_bootstrap, cluster_bootstrap_mean
from .panel import Panel, split_at_quantile, split_masks, to_dt64
from .pool import DEFAULT_L2, fit_predict, standard_models

# Assumed half-spread for rows with only a mid: one full cent each side (a 2c spread), i.e. one
# tick on 0.01-tick markets. Liquid Polymarket markets often quote a 1c spread, so this is on the
# conservative side for them, but thin markets can be wider: treat such P&L as approximate.
DEFAULT_HALF_SPREAD = 0.01
DEFAULT_TICK = 0.01
DEFAULT_MODELS = ("market_recal", "pool", "llm", "llm_recal")
BASELINE = "market_recal"


@dataclass
class Quotes:
    """Per-row executable quotes aligned with a panel's rows.

    ``bid``/``ask`` are YES prices in dollars; ``approximate`` marks rows whose spread was
    assumed; ``fee`` holds ``Question.extra["fee"]`` terms; ``book`` the live order book
    (``{"bids": [[p, size], ...], "asks": [...]}``) or None.
    """

    bid: np.ndarray
    ask: np.ndarray
    approximate: np.ndarray
    fee: np.ndarray  # object: fee-term dicts
    book: np.ndarray  # object: dict | None

    def __len__(self) -> int:
        return int(self.bid.size)

    def subset(self, mask) -> Quotes:
        m = np.asarray(mask)
        return Quotes(self.bid[m], self.ask[m], self.approximate[m], self.fee[m], self.book[m])


def build_quotes(
    panel: Panel,
    questions: Iterable[Question],
    snapshots: Iterable[Snapshot],
    *,
    half_spread: float | None = None,
) -> Quotes:
    """Quotes for every panel row from (question_id, horizon) lookups.

    Rows with both bid and ask use them. Otherwise bid/ask = mid -/+ half-spread (default
    ``max(DEFAULT_HALF_SPREAD, tick)``, tick from ``extra["tick_size"]``), clipped to one tick
    inside [0, 1], and the row is ``approximate``. Missing fee terms raise.
    """
    qs = {q.id: q for q in questions}
    snaps = {(s.question_id, s.horizon): s for s in snapshots}
    n = len(panel)
    bid, ask = np.empty(n), np.empty(n)
    approx = np.zeros(n, dtype=bool)
    fee = np.empty(n, dtype=object)
    book = np.empty(n, dtype=object)
    for i, (qid, h) in enumerate(zip(panel.question_id, panel.horizon, strict=True)):
        q, s = qs.get(qid), snaps.get((qid, h))
        if q is None or s is None:
            raise KeyError(f"no question/snapshot for panel row {qid} {h}")
        terms = q.extra.get("fee")
        if not terms:
            raise ValueError(f"{qid}: Question.extra['fee'] missing; cannot price fees")
        fee[i] = dict(terms)
        book[i] = s.book if s.book and (s.book.get("bids") or s.book.get("asks")) else None
        if s.bid is not None and s.ask is not None:
            bid[i], ask[i] = float(s.bid), float(s.ask)
        else:
            tick = float(q.extra.get("tick_size") or DEFAULT_TICK)
            hs = half_spread if half_spread is not None else max(DEFAULT_HALF_SPREAD, tick)
            bid[i] = max(float(s.mid) - hs, tick)
            ask[i] = min(float(s.mid) + hs, 1.0 - tick)
            approx[i] = True
    return Quotes(bid, ask, approx, fee, book)


@dataclass(frozen=True)
class Sizing:
    """Order size per signal.

    ``contracts``: a fixed count (default 100, so Kalshi's cent rounding is realistic rather than
    dominant). ``notional``: floor(notional / all-in price per contract) contracts. ``kelly``:
    ``kelly_fraction`` x the Kelly stake (q - c) / (1 - c) of a fixed ``bankroll`` (no
    compounding, so results don't depend on row order), capped at ``max_fraction`` of it.
    """

    kind: Literal["contracts", "notional", "kelly"] = "contracts"
    contracts: int = 100
    notional: float = 100.0
    kelly_fraction: float = 0.25
    bankroll: float = 10_000.0
    max_fraction: float = 0.02

    def size(self, q_side: float, cost: float) -> int:
        """Contracts to buy given the side's probability and all-in cost per contract."""
        if self.kind == "contracts":
            return int(self.contracts)
        if self.kind == "notional":
            return int(math.floor(self.notional / cost + 1e-9))
        if self.kind == "kelly":
            f = self.kelly_fraction * (q_side - cost) / (1.0 - cost)
            stake = min(max(f, 0.0), self.max_fraction) * self.bankroll
            return int(math.floor(stake / cost + 1e-9))
        raise ValueError(f"unknown sizing kind: {self.kind!r}")


@dataclass
class Trade:
    side: Literal["yes", "no"]
    contracts: int
    cost: float  # dollars paid for the contracts, before fees
    fee: float
    raw_fee: float  # fee before rounding
    edge: float  # expected profit per contract after fees, under the model
    capacity_known: bool
    capped: bool  # book depth limited the size
    fills: list[tuple[float, int]] = field(default_factory=list)

    @property
    def price(self) -> float:
        return self.cost / self.contracts


def _levels(book: Mapping[str, Any] | None, side: str) -> list[tuple[float, float]] | None:
    """Price levels for buying ``side`` (YES walks asks up; NO walks bids down at 1 - bid)."""
    if not book:
        return None
    if side == "yes":
        lv = sorted((float(p), float(s)) for p, s in book.get("asks") or [])
    else:
        lv = sorted((1.0 - float(p), float(s)) for p, s in book.get("bids") or [])
    return [(p, s) for p, s in lv if s > 0]


def _side(
    side: Literal["yes", "no"],
    q_side: float,
    best: float,
    terms: Mapping[str, Any],
    book: Mapping[str, Any] | None,
    sizing: Sizing,
    threshold: float,
    kalshi_precision: float,
) -> Trade | None:
    if not 0.0 < best < 1.0 or not math.isfinite(best):
        return None
    unit = best + raw_fee_per_contract(terms, best)
    if q_side - unit <= threshold:  # rounding and deeper levels only make it worse
        return None
    n = sizing.size(q_side, unit)
    if n <= 0:
        return None
    levels = _levels(book, side)
    capped = False
    if levels is None:
        fills = [(best, n)]
    else:
        fills, left = [], n
        for p, s in levels:
            if left <= 0 or q_side - p - raw_fee_per_contract(terms, p) <= threshold:
                break
            take = int(min(left, math.floor(s)))
            if take > 0:
                fills.append((p, take))
                left -= take
        capped = left > 0
        n = n - left
        if n <= 0:
            return None
    cost = sum(p * c for p, c in fills)
    fee = order_fee(terms, fills, kalshi_precision=kalshi_precision)
    edge = q_side - (cost + fee) / n
    if edge <= threshold:
        return None
    raw = raw_order_fee(terms, fills)
    return Trade(side, n, cost, fee, raw, edge, levels is not None, capped, fills)


def decide(
    q: float,
    bid: float,
    ask: float,
    terms: Mapping[str, Any],
    *,
    book: Mapping[str, Any] | None = None,
    sizing: Sizing | None = None,
    threshold: float = 0.0,
    kalshi_precision: float = KALSHI_PRECISION,
) -> Trade | None:
    """The order (if any) a model with probability ``q`` places at these quotes.

    Edges are strict: a trade needs expected profit per contract after the order's actual fee
    above ``threshold``. If both sides qualify (crossed quotes) the larger edge wins.
    """
    sizing = sizing or Sizing()
    args = (terms, book, sizing, threshold, kalshi_precision)
    yes = _side("yes", q, ask, *args)
    no = _side("no", 1.0 - q, 1.0 - bid, *args)
    if yes and no:
        return yes if yes.edge >= no.edge else no
    return yes or no


# --- backtest -----------------------------------------------------------------------------------


def _time(t) -> str:
    return str(np.datetime64(t, "s")) + "Z"


def backtest(
    panel: Panel,
    probs: Mapping[str, np.ndarray],
    quotes: Quotes,
    *,
    threshold: float = 0.0,
    sizing: Sizing | None = None,
    kalshi_precision: float = KALSHI_PRECISION,
    baseline: str = BASELINE,
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    include_trades: bool = False,
) -> dict[str, Any]:
    """Trade every model's probabilities on the panel rows and report P&L.

    ``probs`` maps model names to arrays aligned with ``panel`` rows (e.g. test-period
    predictions from ``pool.fit_predict``); ``quotes`` comes from ``build_quotes`` on the same
    panel. Results are split by venue, horizon and venue x horizon (``"all"`` pools rows); CIs
    are event-clustered bootstraps. ``incremental_vs_<baseline>`` is the paired per-row P&L
    difference (model - baseline, zero for rows without a trade). Returns a JSON-safe dict.
    """
    sizing = sizing or Sizing()
    n = len(panel)
    if len(quotes) != n:
        raise ValueError(f"quotes have {len(quotes)} rows, panel has {n}")
    for name, p in probs.items():
        if np.asarray(p).shape != (n,):
            raise ValueError(f"probs[{name!r}] must have shape ({n},)")

    rows = {
        name: _trade_rows(panel, np.asarray(p, float), quotes, sizing, threshold, kalshi_precision)
        for name, p in probs.items()
    }
    groups = _groups(panel)
    by_group: dict[str, Any] = {}
    for label, mask in groups.items():
        s = derive_seed(seed, f"pnl|{label}")
        block: dict[str, Any] = {"n_rows": int(mask.sum()), "models": {}}
        for name, r in rows.items():
            block["models"][name] = _summary(r, mask, panel.event_id, B, s, alpha)
        if baseline in rows:
            block[f"incremental_vs_{baseline}"] = {
                name: cluster_bootstrap_mean(
                    (r["pnl"] - rows[baseline]["pnl"])[mask],
                    panel.event_id[mask],
                    B=B,
                    seed=s,
                    alpha=alpha,
                )
                for name, r in rows.items()
                if name != baseline and mask.any()
            }
        by_group[label] = block

    out: dict[str, Any] = {
        "meta": {
            "n_rows": n,
            "models": list(probs),
            "baseline": baseline,
            "threshold": threshold,
            "sizing": sizing.__dict__,
            "kalshi_precision": kalshi_precision,
            "role": "taker",
            "hold": "to resolution, payoff 1 per winning contract",
            "n_approximate_rows": int(quotes.approximate.sum()),
            "n_rows_with_book": int(sum(b is not None for b in quotes.book)),
            "B": B,
            "seed": seed,
            "alpha": alpha,
            "notes": [
                "approximate rows trade at an assumed spread around the mid",
                "capacity is unknown for rows without an order book (size uncapped)",
                "pnl_per_row and incremental use all rows (0 when no trade)",
            ],
        },
        "by_group": by_group,
        "curves": {name: _curve(panel, r) for name, r in rows.items()},
    }
    if include_trades:
        out["trades"] = {name: _trade_list(panel, r) for name, r in rows.items()}
    return _jsonable(out)


def _trade_rows(panel, p, quotes: Quotes, sizing, threshold, precision) -> dict[str, np.ndarray]:
    n = len(panel)
    cols = {
        k: np.zeros(n)
        for k in ("contracts", "cost", "fee", "raw_fee", "fee_if_1", "payoff", "pnl", "price")
    }
    side = np.full(n, "", dtype=object)
    flags = {k: np.zeros(n, dtype=bool) for k in ("traded", "win", "capacity_known", "capped")}
    for i in range(n):
        t = decide(
            float(p[i]),
            float(quotes.bid[i]),
            float(quotes.ask[i]),
            quotes.fee[i],
            book=quotes.book[i],
            sizing=sizing,
            threshold=threshold,
            kalshi_precision=precision,
        )
        if t is None:
            continue
        win = (t.side == "yes") == bool(panel.y[i])
        payoff = float(t.contracts) if win else 0.0
        side[i] = t.side
        flags["traded"][i], flags["win"][i] = True, win
        flags["capacity_known"][i], flags["capped"][i] = t.capacity_known, t.capped
        cols["contracts"][i], cols["cost"][i], cols["price"][i] = t.contracts, t.cost, t.price
        cols["fee"][i], cols["raw_fee"][i] = t.fee, t.raw_fee
        cols["fee_if_1"][i] = order_fee(
            quotes.fee[i], [(t.fills[0][0], 1)], kalshi_precision=precision
        )
        cols["payoff"][i], cols["pnl"][i] = payoff, payoff - t.cost - t.fee
    venue = np.asarray(panel.venue, dtype=object)
    return cols | flags | {"side": side, "approximate": quotes.approximate, "venue": venue}


def _groups(panel: Panel) -> dict[str, np.ndarray]:
    order = {"7d": 0, "1d": 1, "1h": 2}
    hs = sorted(set(panel.horizon.tolist()), key=lambda h: (order.get(h, 9), h))
    vs = sorted(set(panel.venue.tolist()))
    g = {"all": np.ones(len(panel), dtype=bool)}
    g |= {f"venue:{v}": panel.venue == v for v in vs}
    g |= {f"horizon:{h}": panel.horizon == h for h in hs}
    g |= {f"{v}|{h}": (panel.venue == v) & (panel.horizon == h) for v in vs for h in hs}
    return {k: np.asarray(m, dtype=bool) for k, m in g.items() if np.any(m)}


def _ratio_ci(num, den, clusters, B, seed, alpha) -> dict[str, Any]:
    if num.size == 0 or np.unique(clusters).size < 2:
        est = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
        return {"estimate": est, "lo": None, "hi": None, "se": None, "n": int(num.size)}
    return cluster_bootstrap(
        lambda a, b: a.sum() / b.sum() if b.sum() > 0 else np.nan,
        [num, den],
        clusters,
        B=B,
        seed=seed,
        alpha=alpha,
    )


def _summary(r, mask, events, B, seed, alpha) -> dict[str, Any]:
    t = mask & r["traded"]
    ev = events[t]
    pnl, cost, fee = r["pnl"][t], r["cost"][t], r["fee"][t]
    risked = cost + fee
    gross = r["payoff"][t] - cost
    k = t & (r["venue"] == "kalshi")
    out: dict[str, Any] = {
        "n_rows": int(mask.sum()),
        "n_trades": int(t.sum()),
        "n_yes": int((r["side"][t] == "yes").sum()),
        "n_no": int((r["side"][t] == "no").sum()),
        "n_trade_clusters": int(np.unique(ev).size),
        "n_approximate": int(r["approximate"][t].sum()),
        "n_capacity_known": int(r["capacity_known"][t].sum()),
        "n_capped": int(r["capped"][t].sum()),
        "contracts": float(r["contracts"][t].sum()),
        "risked": float(risked.sum()),
        "fees": float(fee.sum()),
        "gross_pnl": float(gross.sum()),
        "pnl": float(pnl.sum()),
        "hit_rate": float(r["win"][t].mean()) if t.any() else None,
        "mean_price": float(r["price"][t].mean()) if t.any() else None,
        "pnl_per_trade": cluster_bootstrap_mean(pnl, ev, B=B, seed=seed, alpha=alpha),
        "pnl_per_row": cluster_bootstrap_mean(
            r["pnl"][mask], events[mask], B=B, seed=seed, alpha=alpha
        ),
        "roi": _ratio_ci(pnl, risked, ev, B, seed, alpha),
        "roi_before_fees": _ratio_ci(gross, cost, ev, B, seed, alpha),
    }
    if k.any():
        c = r["contracts"][k]
        out["kalshi_fee_per_contract"] = {
            "charged": float(r["fee"][k].sum() / c.sum()),
            "unrounded": float(r["raw_fee"][k].sum() / c.sum()),
            "if_1_contract_orders": float(r["fee_if_1"][k].mean()),
        }
    return out


def _curve(panel: Panel, r) -> list[dict[str, Any]]:
    """Cumulative P&L over trades ordered by close time (when the P&L is realised)."""
    idx = np.flatnonzero(r["traded"])
    idx = idx[np.argsort(panel.close_at[idx], kind="stable")]
    cum = np.cumsum(r["pnl"][idx])
    return [
        {
            "close_at": _time(panel.close_at[i]),
            "venue": str(panel.venue[i]),
            "horizon": str(panel.horizon[i]),
            "pnl": float(r["pnl"][i]),
            "cum_pnl": float(c),
        }
        for i, c in zip(idx, cum, strict=True)
    ]


def _trade_list(panel: Panel, r) -> list[dict[str, Any]]:
    return [
        {
            "question_id": str(panel.question_id[i]),
            "horizon": str(panel.horizon[i]),
            "side": r["side"][i],
            "contracts": int(r["contracts"][i]),
            "price": float(r["price"][i]),
            "fee": float(r["fee"][i]),
            "win": bool(r["win"][i]),
            "pnl": float(r["pnl"][i]),
            "approximate": bool(r["approximate"][i]),
        }
        for i in np.flatnonzero(r["traded"])
    ]


def backtest_split(
    panel: Panel,
    questions: Iterable[Question],
    snapshots: Iterable[Snapshot],
    forecaster_key: str,
    *,
    split: float | str | datetime | np.datetime64 = 0.5,
    models: Sequence[str] = DEFAULT_MODELS,
    eps: float = 0.01,
    l2: float = DEFAULT_L2,
    half_spread: float | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Fit ``models`` (names from ``pool.standard_models``) on the training period, predict the
    embargoed test period (same split rule as ``evaluate``) and ``backtest`` them there."""
    split_at = (
        split_at_quantile(panel, float(split))
        if isinstance(split, float | int) and not isinstance(split, bool)
        else to_dt64(split)
    )
    tr, te = split_masks(panel, split_at)
    train, test = panel.subset(tr), panel.subset(te)
    specs = [s for s in standard_models(forecaster_key) if s.name in models]
    missing = set(models) - {s.name for s in specs}
    if missing:
        raise ValueError(f"unknown models: {sorted(missing)}")
    preds = fit_predict(specs, train.columns(), train.y, test.columns(), eps=eps, l2=l2)
    quotes = build_quotes(test, questions, snapshots, half_spread=half_spread)
    out = backtest(test, {k: p for k, (p, _) in preds.items()}, quotes, **kw)
    out["meta"] |= {
        "forecaster_key": forecaster_key,
        "split_at": _time(split_at),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
    }
    return out
