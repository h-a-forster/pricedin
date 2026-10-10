from __future__ import annotations

import json

import numpy as np
import pytest

from pricedin.eval.panel import Panel, to_dt64
from pricedin.eval.pnl import Quotes, Sizing, backtest, backtest_split, build_quotes, decide
from pricedin.schema import Question, Snapshot

KALSHI = {"kind": "kalshi", "fee_type": "quadratic", "multiplier": 1.0}
PM = {"kind": "polymarket", "rate": 0.04, "exponent": 1, "taker_only": True}
FREE = {"kind": "polymarket", "rate": 0.0, "exponent": 1, "taker_only": True}


# --- decision rule ------------------------------------------------------------------------------


def test_threshold_is_strict():
    one = Sizing(contracts=1)
    assert decide(0.75, 0.4, 0.5, FREE, sizing=one, threshold=0.25) is None  # edge == threshold
    t = decide(0.75, 0.4, 0.5, FREE, sizing=one, threshold=0.24)
    assert t is not None and t.side == "yes" and t.price == 0.5
    assert decide(0.5, 0.5, 0.5, FREE, sizing=one) is None  # zero edge, zero threshold


def test_buy_no_at_one_minus_bid():
    t = decide(0.25, 0.5, 0.6, FREE, sizing=Sizing(contracts=10))
    assert t is not None and t.side == "no"
    assert t.price == pytest.approx(0.5) and t.cost == pytest.approx(5.0)
    assert t.edge == pytest.approx(0.25)
    assert decide(0.45, 0.4, 0.5, FREE) is None  # inside the spread: no trade


def test_kalshi_cent_rounding_kills_single_contract_edge():
    # raw fee 0.0175/contract leaves 0.0025 edge; 1 contract pays 0.02 -> no edge left
    assert decide(0.52, 0.4, 0.5, KALSHI, sizing=Sizing(contracts=1)) is None
    t = decide(0.52, 0.4, 0.5, KALSHI, sizing=Sizing(contracts=100))
    assert t is not None and t.fee == pytest.approx(1.75)
    assert t.edge == pytest.approx(0.0025)


def test_book_walk_vwap_and_cap():
    book = {"asks": [[0.52, 50], [0.50, 30], [0.60, 100]], "bids": [[0.45, 10]]}
    t = decide(0.58, 0.45, 0.50, FREE, book=book, sizing=Sizing(contracts=100))
    assert t is not None and t.capacity_known and t.capped  # stops before the 0.60 level
    assert t.contracts == 80 and t.cost == pytest.approx(0.5 * 30 + 0.52 * 50)
    t = decide(0.58, 0.45, 0.50, FREE, book=book, sizing=Sizing(contracts=20))
    assert t.contracts == 20 and not t.capped and t.price == pytest.approx(0.5)


def test_sizing_modes():
    t = decide(0.7, 0.4, 0.5, FREE, sizing=Sizing("notional", notional=100.0))
    assert t.contracts == 200
    t = decide(
        0.7,
        0.4,
        0.5,
        FREE,
        sizing=Sizing("kelly", kelly_fraction=0.5, max_fraction=1.0, bankroll=1000.0),
    )
    assert t.contracts == 400  # 0.5 * (0.7-0.5)/(1-0.5) * 1000 / 0.5
    t = decide(0.7, 0.4, 0.5, FREE, sizing=Sizing("kelly", max_fraction=0.01, bankroll=1000.0))
    assert t.contracts == 20


# --- synthetic backtests ------------------------------------------------------------------------


def _panel(n, rng, venue="polymarket"):
    p = rng.uniform(0.1, 0.9, n)
    y = (rng.random(n) < p).astype(int)
    close = to_dt64("2026-07-01T00:00:00Z") + np.arange(n).astype("timedelta64[h]")
    panel = Panel(
        question_id=np.array([f"q{i}" for i in range(n)], dtype=object),
        horizon=np.where(np.arange(n) % 2 == 0, "1d", "7d").astype(object),
        y=y,
        market=p.copy(),
        forecasts={},
        event_id=np.array([f"e{i // 2}" for i in range(n)], dtype=object),
        category=np.full(n, "other", dtype=object),
        venue=np.full(n, venue, dtype=object),
        close_at=close,
        as_of=close - np.timedelta64(1, "D"),
    )
    return panel, p


def _quotes(mid, half, fee):
    n = mid.size
    terms = np.empty(n, dtype=object)
    terms[:] = [dict(fee) for _ in range(n)]
    return Quotes(
        np.clip(mid - half, 0.01, 0.99),
        np.clip(mid + half, 0.01, 0.99),
        np.zeros(n, dtype=bool),
        terms,
        np.full(n, None, dtype=object),
    )


def _expected_pnl(res_trades, p_true, qids):
    """Expected P&L under the true probabilities, from the trade list."""
    idx = {q: i for i, q in enumerate(qids)}
    gross = fees = 0.0
    for t in res_trades:
        pt = p_true[idx[t["question_id"]]]
        win_p = pt if t["side"] == "yes" else 1 - pt
        gross += t["contracts"] * (win_p - t["price"])
        fees += t["fee"]
    return gross, gross - fees


def test_true_probability_makes_money():
    rng = np.random.default_rng(1)
    panel, p = _panel(3000, rng)
    mid = np.clip(p + rng.normal(0, 0.1, p.size), 0.02, 0.98)  # noisy market, efficient mean
    res = backtest(panel, {"oracle": p}, _quotes(mid, 0.01, PM), B=200, include_trades=True)
    m = res["by_group"]["all"]["models"]["oracle"]
    assert m["n_trades"] > 1000
    exp_gross, exp_net = _expected_pnl(res["trades"]["oracle"], p, panel.question_id)
    assert exp_net > 0
    assert m["pnl"] > 0 and m["roi"]["estimate"] > 0
    assert m["pnl_per_trade"]["lo"] > 0
    assert m["fees"] > 0 and m["gross_pnl"] > m["pnl"]


def test_noise_on_efficient_market_loses_after_fees():
    rng = np.random.default_rng(2)
    panel, p = _panel(3000, rng)
    q = np.clip(p + rng.normal(0, 0.1, p.size), 0.02, 0.98)  # no information beyond the market
    res = backtest(
        panel, {"noise": q, "market": p}, _quotes(p, 0.0, PM), B=200, include_trades=True
    )
    m = res["by_group"]["all"]["models"]
    assert m["market"]["n_trades"] == 0  # trading on the market price itself never fires
    exp_gross, exp_net = _expected_pnl(res["trades"]["noise"], p, panel.question_id)
    assert exp_gross == pytest.approx(0.0, abs=1e-6)  # zero spread at fair prices: zero edge
    assert exp_net < 0  # fees make it a loser
    noise = m["noise"]
    assert noise["n_trades"] > 500
    assert abs(noise["roi_before_fees"]["estimate"]) < 0.05
    assert noise["pnl"] < noise["gross_pnl"]


def test_result_structure_and_json():
    rng = np.random.default_rng(3)
    panel, p = _panel(200, rng, venue="kalshi")
    panel.venue[::4] = "polymarket"
    mid = np.clip(p + rng.normal(0, 0.1, p.size), 0.02, 0.98)
    probs = {"market_recal": mid, "pool": (mid + p) / 2, "llm": p}
    quotes = _quotes(mid, 0.01, KALSHI)
    res = backtest(panel, probs, quotes, B=50, sizing=Sizing(contracts=1))
    assert json.loads(json.dumps(res, allow_nan=False)) == res
    assert {"all", "venue:kalshi", "venue:polymarket", "horizon:1d", "kalshi|7d"} <= set(
        res["by_group"]
    )
    inc = res["by_group"]["all"]["incremental_vs_market_recal"]
    assert set(inc) == {"pool", "llm"}
    k = res["by_group"]["all"]["models"]["llm"]["kalshi_fee_per_contract"]
    assert k["charged"] >= k["unrounded"]
    curve = res["curves"]["llm"]
    assert curve == sorted(curve, key=lambda r: r["close_at"])
    assert curve[-1]["cum_pnl"] == pytest.approx(res["by_group"]["all"]["models"]["llm"]["pnl"])
    with pytest.raises(ValueError):
        backtest(panel, {"x": p[:10]}, quotes)


# --- record plumbing ----------------------------------------------------------------------------


def _records(n, rng):
    qs, snaps = [], []
    for i in range(n):
        venue = "kalshi" if i % 2 else "polymarket"
        p = float(rng.uniform(0.1, 0.9))
        close = f"2026-{6 + i * 6 // n:02d}-{1 + i % 27:02d}T00:00:00Z"
        q = Question(
            id=f"{venue}:{i}",
            venue=venue,
            event_id=f"{venue}:e{i // 3}",
            title="t",
            rules="r",
            category="other",
            created_at="2026-01-01T00:00:00Z",
            open_at="2026-01-01T00:00:00Z",
            close_at=close,
            scheduled_close_at="",
            outcome=int(rng.random() < p),
            volume=1.0,
            extra={"fee": KALSHI if venue == "kalshi" else PM},
        )
        qs.append(q)
        for h, d in (("7d", 7), ("1d", 1)):
            as_of = str(to_dt64(close) - np.timedelta64(d, "D")) + "Z"
            mid = float(np.clip(p + rng.normal(0, 0.05), 0.02, 0.98))
            kw = (
                {"bid": round(mid - 0.01, 2), "ask": round(mid + 0.01, 2)}
                if venue == "kalshi"
                else {}
            )
            snaps.append(Snapshot(question_id=q.id, horizon=h, as_of=as_of, mid=mid, **kw))
    return qs, snaps


def test_build_quotes_and_split():
    from pricedin.eval.panel import build_panel
    from pricedin.schema import Forecast

    rng = np.random.default_rng(4)
    qs, snaps = _records(240, rng)
    fc = [
        Forecast(
            question_id=s.question_id,
            horizon=s.horizon,
            as_of=s.as_of,
            forecaster="m",
            condition="base",
            prob=float(np.clip(s.mid + rng.normal(0, 0.05), 0.01, 0.99)),
        )
        for s in snaps
    ]
    panel = build_panel(qs, snaps, fc)
    quotes = build_quotes(panel, qs, snaps)
    pm = panel.venue == "polymarket"
    assert quotes.approximate[pm].all() and not quotes.approximate[~pm].any()
    assert np.allclose(quotes.ask[pm] - quotes.bid[pm], 0.02)
    res = backtest_split(panel, qs, snaps, "m|base", B=50)
    assert set(res["meta"]["models"]) == {"market_recal", "pool", "llm", "llm_recal"}
    assert res["meta"]["n_test"] > 0
    json.dumps(res, allow_nan=False)
    bad = [Question(**{**qs[0].__dict__, "extra": {}})] + qs[1:]
    with pytest.raises(ValueError, match="fee"):
        build_quotes(panel, bad, snaps)
