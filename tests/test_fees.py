from __future__ import annotations

import pytest

from pricedin.eval.fees import (
    ceil_to,
    kalshi_fee,
    order_fee,
    polymarket_fee,
    raw_fee_per_contract,
)

K = {"kind": "kalshi", "fee_type": "quadratic", "multiplier": 1.0}


@pytest.mark.parametrize(
    ("c", "p", "want"),
    [
        (100, 0.50, 1.75),  # peak, no rounding needed
        (1, 0.50, 0.02),  # 0.0175 rounds up to 2 cents
        (100, 0.56, 1.73),  # 1.7248 -> 1.73
        (1, 0.56, 0.02),
        (10, 0.10, 0.07),  # 0.063 -> 0.07
        (100, 0.0, 0.0),
        (100, 1.0, 0.0),
    ],
)
def test_kalshi_taker_known_values(c, p, want):
    assert kalshi_fee(c, p) == pytest.approx(want, abs=1e-12)


def test_kalshi_multiplier_and_precision():
    assert kalshi_fee(100, 0.5, multiplier=0.0) == 0.0
    assert kalshi_fee(100, 0.5, multiplier=0.5) == pytest.approx(0.88)  # 0.875 -> 0.88
    assert kalshi_fee(1, 0.5, precision=0.0001) == pytest.approx(0.0175)


def test_kalshi_maker_fees():
    assert kalshi_fee(100, 0.5, role="maker") == 0.0  # plain quadratic: takers only
    mk = kalshi_fee(100, 0.5, fee_type="quadratic_with_maker_fees", role="maker")
    assert mk == pytest.approx(0.44)  # 0.4375 -> 0.44
    combo = kalshi_fee(100, 0.5, fee_type="quadratic_with_combo_maker_fees", role="maker")
    assert combo == pytest.approx(0.88)  # 0.5 x taker rate: 0.875 -> 0.88
    assert kalshi_fee(100, 0.5, fee_type="quadratic_with_maker_fees") == pytest.approx(1.75)


def test_kalshi_flat_and_unknown():
    with pytest.raises(NotImplementedError, match="Specific Trading Fees"):
        kalshi_fee(10, 0.5, fee_type="flat")
    with pytest.raises(ValueError):
        kalshi_fee(10, 0.5, fee_type="cubic")
    with pytest.raises(ValueError):
        kalshi_fee(10, 1.5)


@pytest.mark.parametrize(
    ("c", "p", "rate", "want"),
    [  # docs.polymarket.com/trading/fees worked examples
        (100, 0.50, 0.07, 1.75),
        (100, 0.30, 0.07, 1.47),
        (100, 0.70, 0.07, 1.47),
        (100, 0.50, 0.05, 1.25),
        (50, 0.20, 0.04, 0.32),
        (100, 0.01, 0.07, 0.0693),  # fee table shows $0.07
        (100, 0.05, 0.07, 0.3325),  # fee table shows $0.33
    ],
)
def test_polymarket_doc_examples(c, p, rate, want):
    assert polymarket_fee(c, p, rate=rate) == pytest.approx(want, abs=1e-12)


def test_polymarket_rounding_maker_and_exponent():
    assert polymarket_fee(1, 0.001, rate=0.04) == pytest.approx(0.00004)  # 3.996e-5
    assert polymarket_fee(1, 0.0001, rate=0.04) == 0.0  # below 0.00001 -> 0
    assert polymarket_fee(100, 0.5, rate=0.07, role="maker") == 0.0
    # exponent applies to p(1-p); rollout-era formula adds a leading p
    assert polymarket_fee(100, 0.5, rate=0.2, exponent=2) == pytest.approx(1.25)
    assert polymarket_fee(100, 0.5, rate=0.072, formula="rollout_2026") == pytest.approx(0.9)


def test_order_fee_terms():
    pm = {"kind": "polymarket", "rate": 0.07, "exponent": 1, "taker_only": True}
    assert order_fee(pm, [(0.5, 100)]) == pytest.approx(1.75)
    assert order_fee(pm | {"feesEnabled": False}, [(0.5, 100)]) == 0.0
    assert order_fee(pm | {"enabled": False}, [(0.5, 100)]) == 0.0  # key written by fee_terms
    assert order_fee(pm | {"enabled": True}, [(0.5, 100)]) == pytest.approx(1.75)
    assert order_fee(pm | {"rate": 0}, [(0.5, 100)]) == 0.0
    # Kalshi: summed over fills, rounded once per order
    assert order_fee(K, [(0.5, 1), (0.5, 1)]) == pytest.approx(0.04)  # 0.035 -> 0.04
    assert order_fee(K, [(0.5, 50), (0.56, 50)]) == pytest.approx(1.74)  # 0.875+0.8624
    assert raw_fee_per_contract(K, 0.5) == pytest.approx(0.0175)
    with pytest.raises(ValueError, match="missing"):
        order_fee(None, [(0.5, 1)])
    with pytest.raises(ValueError, match="kind"):
        order_fee({"kind": "nyse"}, [(0.5, 1)])


def test_ceil_to_ignores_float_noise():
    assert ceil_to(0.07 * 100 * 0.5 * 0.5, 0.01) == pytest.approx(1.75)
    assert ceil_to(0.0, 0.01) == 0.0
