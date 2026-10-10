"""Trading fees for Kalshi and Polymarket binary contracts, per order.

Fee terms come from ``Question.extra["fee"]`` (see the venue contract in the build spec). Fees
are charged on the whole order, then rounded, so small orders pay proportionally more.

Kalshi (sources accessed 2026-10-10):
- Taker: ``M x 0.07 x C x P x (1 - P)`` and, for maker-fee series, maker
  ``M x 0.0175 x C x P x (1 - P)`` (0.0175 = 0.25 x the taker rate), each rounded up.
  https://docs.allium.so/historical-data/predictions/kalshi/series-fee-changes (restates
  Kalshi's published formula, M = series fee multiplier); the official PDF
  https://kalshi.com/docs/kalshi-fee-schedule.pdf blocks automated fetches.
- ``fee_type`` values: ``quadratic`` (General Trading Fees Table, takers only),
  ``quadratic_with_maker_fees`` (plus Maker Fees section), ``quadratic_with_combo_maker_fees``
  ("the same maker-fee structure with a 0.5 maker multiplier instead of 0.25") and ``flat``
  ("described by the Specific Trading Fees Table").
  https://docs.kalshi.com/api-reference/market/get-series
- Rounding: the schedule rounds the order's fee up to the next cent. Since the per-fill fee
  accumulator (https://docs.kalshi.com/getting_started/fee_rounding) direct members settle to
  $0.0001 and FCM-cleared members to $0.01, with over-rounding across fills of one order rebated;
  so the net order fee is the model fee rounded up to the member's precision. Default: $0.01.
- The series ``multiplier`` is applied to maker fees too (sources only name a separate maker
  multiplier, which the venue terms don't carry); backtests here charge taker fees only.
- ``flat``: the Specific Trading Fees Table is only in the PDF, which we could not read, so it
  raises ``NotImplementedError``.

Polymarket (https://docs.polymarket.com/trading/fees and
https://docs.polymarket.com/market-data/market-details#trading-fees, accessed 2026-10-10):
- Current docs: ``fee = C x feeRate x p x (1 - p)``, takers only ("Makers are never charged
  fees"), rounded to 5 decimals USDC (fees below 0.00001 round to zero). The market's
  ``feeSchedule`` carries ``rate``, ``exponent`` ("exponent applied to the price component"),
  ``takerOnly`` and ``rebateRate``; ``feesEnabled`` false means no fee (stored as ``enabled`` in
  ``extra["fee"]``).
- We read the exponent as the power of the price component p(1-p), so ``exponent = 1`` gives the
  documented formula. Exponents other than 1 are not in the current docs page; they appear in the
  rollout-era formula ``C x p x feeRate x (p(1-p))^exponent`` (Jan-Mar 2026, per
  https://pineanalytics.substack.com/p/polymarket-fee-rollout), available as
  ``formula="rollout_2026"``. Which formula applied to a historical trade date is not documented.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal

KALSHI_TAKER_RATE = 0.07
KALSHI_MAKER_RATE = 0.0175  # 0.25 x taker
KALSHI_COMBO_MAKER_RATE = 0.035  # 0.5 x taker
KALSHI_PRECISION = 0.01  # dollars; 0.0001 for direct members under the fee accumulator
POLYMARKET_DECIMALS = 5

_KALSHI_MAKER_RATES = {
    "quadratic": 0.0,
    "quadratic_with_maker_fees": KALSHI_MAKER_RATE,
    "quadratic_with_combo_maker_fees": KALSHI_COMBO_MAKER_RATE,
}

Role = Literal["taker", "maker"]
Fills = Iterable[tuple[float, float]]  # (price in dollars, contracts)


def ceil_to(x: float, step: float) -> float:
    """Round ``x`` up to a multiple of ``step``, ignoring float noise below 1e-9 steps."""
    k = x / step
    return round(math.ceil(round(k, 9)) * step, 12) if x > 0 else 0.0


def _check_price(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"price must be in [0, 1] dollars, got {p}")


def kalshi_raw_fee(
    contracts: float,
    price: float,
    *,
    fee_type: str = "quadratic",
    multiplier: float = 1.0,
    role: Role = "taker",
) -> float:
    """Unrounded Kalshi fee in dollars for one fill (``price`` in dollars)."""
    _check_price(price)
    if fee_type == "flat":
        raise NotImplementedError(
            "Kalshi fee_type 'flat' uses the Specific Trading Fees Table in "
            "https://kalshi.com/docs/kalshi-fee-schedule.pdf, which we could not verify; "
            "add it here once the table is confirmed"
        )
    if fee_type not in _KALSHI_MAKER_RATES:
        raise ValueError(f"unknown Kalshi fee_type: {fee_type!r}")
    rate = KALSHI_TAKER_RATE if role == "taker" else _KALSHI_MAKER_RATES[fee_type]
    return multiplier * rate * contracts * price * (1.0 - price)


def kalshi_fee(
    contracts: float,
    price: float,
    *,
    fee_type: str = "quadratic",
    multiplier: float = 1.0,
    role: Role = "taker",
    precision: float = KALSHI_PRECISION,
) -> float:
    """Kalshi fee in dollars for one order of ``contracts`` at ``price``, rounded up to
    ``precision``. 100 contracts at 0.50 -> 1.75; 1 contract at 0.50 -> 0.02."""
    raw = kalshi_raw_fee(contracts, price, fee_type=fee_type, multiplier=multiplier, role=role)
    return ceil_to(raw, precision)


def polymarket_raw_fee(
    contracts: float,
    price: float,
    *,
    rate: float,
    exponent: float = 1.0,
    role: Role = "taker",
    taker_only: bool = True,
    formula: str = "current",
) -> float:
    """Unrounded Polymarket fee in USDC for one fill of ``contracts`` shares at ``price``."""
    _check_price(price)
    if role == "maker" and taker_only:
        return 0.0
    pq = price * (1.0 - price)
    if formula == "current":
        return contracts * rate * pq**exponent
    if formula == "rollout_2026":
        return contracts * price * rate * pq**exponent
    raise ValueError(f"unknown Polymarket fee formula: {formula!r}")


def polymarket_fee(contracts: float, price: float, **kw: Any) -> float:
    """Polymarket fee in USDC rounded to 5 decimals (keywords as ``polymarket_raw_fee``)."""
    return round(polymarket_raw_fee(contracts, price, **kw), POLYMARKET_DECIMALS)


def raw_order_fee(terms: Mapping[str, Any] | None, fills: Fills, *, role: Role = "taker") -> float:
    """Unrounded fee for one order filled at one or more ``(price, contracts)`` levels, from
    ``Question.extra["fee"]``. Missing terms raise rather than silently assume zero."""
    if not terms:
        raise ValueError("no fee terms for this question (Question.extra['fee'] missing)")
    fills = [(p, c) for p, c in fills if c > 0]
    kind = terms.get("kind")
    if kind == "kalshi":
        ft = terms.get("fee_type") or "quadratic"
        m = float(terms.get("multiplier", 1.0))
        return sum(kalshi_raw_fee(c, p, fee_type=ft, multiplier=m, role=role) for p, c in fills)
    if kind == "polymarket":
        enabled = terms.get("enabled", terms.get("fees_enabled", terms.get("feesEnabled", True)))
        if enabled is False:
            return 0.0
        exponent = terms.get("exponent")
        kw = {
            "rate": float(terms.get("rate") or 0.0),
            "exponent": 1.0 if exponent is None else float(exponent),
            "taker_only": bool(terms.get("taker_only", True)),
            "formula": terms.get("formula", "current"),
            "role": role,
        }
        return sum(polymarket_raw_fee(c, p, **kw) for p, c in fills)
    raise ValueError(f"unknown fee kind: {kind!r}")


def order_fee(
    terms: Mapping[str, Any] | None,
    fills: Fills,
    *,
    role: Role = "taker",
    kalshi_precision: float = KALSHI_PRECISION,
) -> float:
    """Fee charged for one order: the unrounded fee summed over its fills, rounded once (up to
    ``kalshi_precision`` on Kalshi, to 5 decimals on Polymarket). Summing before rounding is
    what Kalshi's per-order fee accumulator achieves across fills."""
    raw = raw_order_fee(terms, fills, role=role)
    if terms and terms.get("kind") == "kalshi":
        return ceil_to(raw, kalshi_precision)
    return round(raw, POLYMARKET_DECIMALS)


def raw_fee_per_contract(terms: Mapping[str, Any], price: float, *, role: Role = "taker") -> float:
    """Unrounded fee per contract at ``price`` (the large-order limit; used for sizing)."""
    return raw_order_fee(terms, [(price, 1.0)], role=role)
