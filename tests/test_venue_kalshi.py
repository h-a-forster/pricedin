from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from pricedin.http import HttpClient
from pricedin.schema import parse_time, to_iso
from pricedin.venues import kalshi as K

PREFIX = "/trade-api/v2/"

# Hand-trimmed from real responses (2026-10-10).
SERIES = {
    "ticker": "KXTRUMPMEETING",
    "category": "Politics",
    "tags": ["Trump"],
    "frequency": "annual",
    "fee_type": "quadratic",
    "fee_multiplier": 1,
    "title": "Who will Trump meet?",
}
EARLY = {  # resolved YES months before its deadline: close_time overwritten, expected kept
    "ticker": "KXTRUMPMEETING-27JAN01-RERD",
    "event_ticker": "KXTRUMPMEETING-27JAN01",
    "title": "Will Donald Trump meet in person Recep Tayyip Erdoğan before Jan 1, 2027?",
    "yes_sub_title": "Recep Tayyip Erdoğan",
    "rules_primary": "If Donald Trump and Recep Tayyip Erdoğan meet in person ..., Yes.",
    "rules_secondary": "A qualifying meeting must ...",
    "created_time": "2026-05-15T19:03:16.177012Z",
    "open_time": "2026-05-15T21:00:00Z",
    "close_time": "2026-07-07T16:49:08Z",
    "expected_expiration_time": "2027-01-01T15:00:00Z",
    "expiration_time": "2027-01-08T15:00:00Z",
    "latest_expiration_time": "2027-01-08T15:00:00Z",
    "settlement_ts": "2026-07-07T17:19:14.442601Z",
    "can_close_early": True,
    "result": "yes",
    "market_type": "binary",
    "volume_fp": "1971.20",
    "open_interest_fp": "1303.45",
    "status": "finalized",
}
FED = {
    "ticker": "KXFEDDECISION-26SEP-H0",
    "event_ticker": "KXFEDDECISION-26SEP",
    "title": "Will the Federal Reserve Hike rates by 0bps at their September 2026 meeting?",
    "yes_sub_title": "Hike 0bps",
    "rules_primary": "If the Federal Reserve does a Hike of 0bps on September 16, 2026, Yes.",
    "rules_secondary": "",
    "created_time": "2025-09-26T15:30:59.234134Z",
    "open_time": "2025-09-29T14:00:00Z",
    "close_time": "2026-09-16T17:59:00Z",
    "expected_expiration_time": "2026-09-16T18:05:00Z",
    "latest_expiration_time": "2026-12-16T18:01:00Z",
    "settlement_ts": "2026-09-16T18:30:00Z",
    "can_close_early": True,
    "result": "no",
    "market_type": "binary",
    "volume_fp": "48286115.02",
    "open_interest_fp": "29229841.29",
}
FED_SERIES = {
    "ticker": "KXFEDDECISION",
    "category": "Economics",
    "tags": [],
    "frequency": "custom",
    "fee_type": "quadratic_with_maker_fees",
    "fee_multiplier": 0.5,
}
LIVE_CANDLE = {
    "end_period_ts": 1789577940,
    "price": {"close_dollars": "0.1300", "previous_dollars": "0.1100"},
    "volume_fp": "50359.04",
    "open_interest_fp": "31416640.33",
    "yes_ask": {"close_dollars": "0.1300", "open_dollars": "0.1200"},
    "yes_bid": {"close_dollars": "0.1100", "open_dollars": "0.1100"},
}
HIST_CANDLE = {
    "end_period_ts": 1785261180,
    "price": {"close": None, "previous": "0.0100"},
    "volume": "0.00",
    "open_interest": "7191138.04",
    "yes_ask": {"close": "0.0200"},
    "yes_bid": {"close": "0.0000"},
}

T0 = datetime(2026, 7, 1, tzinfo=UTC)
T1 = datetime(2026, 10, 10, tzinfo=UTC)


def utc(s: str) -> datetime:
    return parse_time(s)


# -- parsing ---------------------------------------------------------------------------------


def test_parse_candle_live_and_historical_names():
    live = K.parse_candle(LIVE_CANDLE)
    assert (live.ts, live.bid, live.ask, live.last) == (1789577940, 0.11, 0.13, 0.13)
    hist = K.parse_candle(HIST_CANDLE)
    # no trade in the minute: last falls back to the previous trade
    assert (hist.ts, hist.bid, hist.ask, hist.last) == (1785261180, 0.0, 0.02, 0.01)
    empty = K.parse_candle({"end_period_ts": 5, "price": {}, "yes_bid": {}, "yes_ask": {}})
    assert (empty.bid, empty.ask, empty.last) == (None, None, None)


def _candle(ts: int, bid: str = "0.4", ask: str = "0.5") -> dict:
    return {
        "end_period_ts": ts,
        "price": {},
        "yes_bid": {"close_dollars": bid},
        "yes_ask": {"close_dollars": ask},
    }


def test_select_observation_boundaries():
    as_of = datetime.fromtimestamp(1_000_000, tz=UTC)
    stale = timedelta(hours=6)
    lo = 1_000_000 - 6 * 3600
    # candle ending exactly at as_of is the state at as_of: usable; one minute later is not
    obs = K.select_observation(
        [_candle(999_940), _candle(1_000_000), _candle(1_000_060)], as_of, stale
    )
    assert obs is not None and obs.ts == 1_000_000
    assert K.select_observation([_candle(1_000_060)], as_of, stale) is None
    assert K.select_observation([_candle(lo)], as_of, stale).ts == lo
    assert K.select_observation([_candle(lo - 60)], as_of, stale) is None
    # unaligned as_of (early-closed markets close at odd seconds)
    odd = datetime.fromtimestamp(1_000_030, tz=UTC)
    assert (
        K.select_observation([_candle(1_000_000), _candle(1_000_060)], odd, stale).ts == 1_000_000
    )


def test_quote_rules():
    ob = K.Observation
    assert K.quote(ob(1, 0.11, 0.13, 0.12)) == (0.12, 0.11, 0.13)
    # empty book (bid 0 / ask 1) must not become 0.5: last trade instead
    assert K.quote(ob(1, 0.0, 1.0, 0.07)) == (0.07, None, None)
    assert K.quote(ob(1, 0.0, 1.0, None)) is None
    # one-sided: a lone bid of 0.99 overrides a stale 0.87 trade
    assert K.quote(ob(1, 0.99, 1.0, 0.87)) == (0.99, 0.99, None)
    assert K.quote(ob(1, 0.0, 0.03, 0.10)) == (0.03, None, 0.03)
    assert K.quote(ob(1, 0.0, 0.03, 0.02)) == (0.02, None, 0.03)
    assert K.quote(ob(1, None, None, 1.0)) is None


def test_market_to_question_early_close_anchors_on_scheduled_end():
    q = K.market_to_question(EARLY, SERIES)
    assert q is not None
    assert q.id == "kalshi:KXTRUMPMEETING-27JAN01-RERD"
    assert q.event_id == "kalshi:KXTRUMPMEETING-27JAN01"
    assert (q.close_at, q.scheduled_close_at) == ("2027-01-01T15:00:00Z", "")
    assert q.extra["actual_close"] == "2026-07-07T16:49:08Z"
    assert q.extra["early_close"] is True
    assert q.extra["close_gap_h"] > 4000
    assert q.outcome == 1 and q.volume == 1971.2
    assert q.category == "politics" and q.venue_category == "Politics"
    assert q.extra["fee"] == {"kind": "kalshi", "fee_type": "quadratic", "multiplier": 1.0}
    assert q.rules.startswith("If Donald Trump") and q.rules.endswith("must ...")
    # title already names the outcome: nothing appended
    assert q.title == EARLY["title"]


def test_market_to_question_on_time_close_and_rejects():
    q = K.market_to_question(FED, FED_SERIES)
    assert q.close_at == "2026-09-16T18:05:00Z"
    assert q.extra["actual_close"] == "2026-09-16T17:59:00Z"
    assert q.extra["early_close"] is False
    assert q.extra["fee"]["multiplier"] == 0.5
    assert q.extra["fee"]["fee_type"] == "quadratic_with_maker_fees"
    for bad in (
        {**FED, "result": "scalar"},
        {**FED, "result": ""},
        {**FED, "market_type": "scalar"},
        {**FED, "expected_expiration_time": None},
    ):
        assert K.market_to_question(bad, FED_SERIES) is None


def test_question_title():
    m = {"title": "Who will win the Nobel  Peace Prize?", "yes_sub_title": "Mark Carney"}
    assert K.question_title(m) == "Who will win the Nobel Peace Prize? — Mark Carney"
    m = {"title": "Will the **high temp in NYC** be >97° on Aug 9?", "yes_sub_title": ""}
    assert K.question_title(m) == "Will the high temp in NYC be >97° on Aug 9?"
    m = {"title": "Will approval be above 40.9?", "yes_sub_title": "Above 40.9"}
    assert K.question_title(m) == "Will approval be above 40.9?"


@pytest.mark.parametrize(
    ("category", "tags", "expected"),
    [
        ("Politics", [], "politics"),
        ("Elections", [], "politics"),
        ("Economics", [], "economics"),
        ("Financials", [], "finance"),
        ("Companies", [], "finance"),
        ("Crypto", [], "crypto"),
        ("Science and Technology", ["AI"], "tech"),
        ("Science and Technology", ["Space"], "science"),
        ("Health", [], "science"),
        ("Climate and Weather", [], "weather"),
        ("Sports", [], "sports"),
        ("Entertainment", [], "culture"),
        ("Social", [], "culture"),
        ("World", [], "world"),
        ("Mentions", ["Earnings"], "finance"),
        ("Mentions", ["Politicians"], "politics"),
        ("Mentions", [], "other"),
        ("Transportation", [], "other"),
        ("", [], "other"),
    ],
)
def test_map_category(category, tags, expected):
    assert K.map_category(category, tags) == expected


def test_is_eligible_filters():
    kw = dict(created_after=T0, closed_before=T1, min_open_days=8, min_volume=100)
    base = {
        **FED,
        "created_time": "2026-07-02T00:00:00Z",
        "open_time": "2026-07-02T00:00:00Z",
    }
    assert K.is_eligible(K.market_to_question(base, FED_SERIES), **kw)

    def ok(**over):
        return K.is_eligible(K.market_to_question({**base, **over}, FED_SERIES), **kw)

    assert not ok(created_time="2026-06-30T23:59:59Z")
    assert not ok(open_time="2026-09-10T00:00:00Z")  # open < 8 days before scheduled close
    assert not ok(volume_fp="99")
    # scheduled end after closed_before: excluded even though it already closed (YES early)
    assert not ok(expected_expiration_time="2027-01-01T15:00:00Z")
    # actual close after closed_before (delayed past schedule): excluded
    assert not ok(close_time="2026-10-11T00:00:00Z")


def _prior(close: str, gap: timedelta, result: str = "no", settled: str | None = None) -> dict:
    m = {
        "close_time": close,
        "expected_expiration_time": to_iso(utc(close) + gap),
        "result": result,
    }
    if settled is not None:
        m["settlement_ts"] = settled
    return m


def test_buffer_uses_only_markets_settled_before_listing():
    listed = utc("2026-07-10T00:00:00Z")
    d14 = timedelta(days=14)
    prior = [_prior(f"2026-06-0{i}T00:00:00Z", d14) for i in (1, 2, 3)]  # known at close + 14 d
    after = [
        _prior("2026-07-10T00:00:00Z", timedelta(0)),  # closes at the listing instant: excluded
        _prior("2026-07-20T00:00:00Z", timedelta(0)),
        _prior("2026-07-21T00:00:00Z", timedelta(0)),
    ]
    assert K.settlement_buffer(prior + after, listed) == (14 * 24, 3)
    # closed before listing but not yet settled: its result was not public, so excluded
    late = [_prior(f"2026-07-0{i}T00:00:00Z", d14) for i in (1, 2, 3)]  # known Jul 15-17
    assert K.settlement_buffer(late, listed) == (0.0, 0)
    settled = [
        _prior(f"2026-07-0{i}T00:00:00Z", d14, settled=f"2026-07-0{i}T01:00:00Z") for i in (1, 2, 3)
    ]
    assert K.settlement_buffer(settled, listed) == (14 * 24, 3)


def test_buffer_takes_the_smaller_per_result_median():
    listed = utc("2026-08-01T00:00:00Z")
    days = lambda d: timedelta(days=d)  # noqa: E731
    # mention-style: every market closes at the event, both outcomes show the template gap
    mention = [
        _prior(f"2026-07-0{i}T00:00:00Z", days(14), r)
        for i, r in zip(range(1, 7), ["yes", "no", "yes", "no", "no", "yes"], strict=True)
    ]
    assert K.settlement_buffer(mention, listed)[0] == pytest.approx(14 * 24)
    # "by <date>" ladder: NO runs to the deadline (gap 0), YES closes early (large gaps)
    ladder = [_prior(f"2026-06-0{i}T00:00:00Z", days(0), "no") for i in (1, 2, 3)]
    ladder += [_prior(f"2026-06-1{i}T00:00:00Z", days(d), "yes") for i, d in ((1, 5), (2, 20))]
    assert K.settlement_buffer(ladder, listed) == (0.0, 5)
    # "not by <date>": the roles flip, still the deadline outcome's gap
    flipped = [{**m, "result": "yes" if m["result"] == "no" else "no"} for m in ladder]
    assert K.settlement_buffer(flipped, listed) == (0.0, 5)
    # one class only: its median
    assert K.settlement_buffer(ladder[3:] + [ladder[4]], listed)[0] == 20 * 24


def test_buffer_min_count_lookback_and_zero_fallback():
    listed = utc("2026-07-10T00:00:00Z")
    two = [_prior(f"2026-06-0{i}T00:00:00Z", timedelta(days=9)) for i in (1, 2)]
    assert K.settlement_buffer(two, listed) == (0.0, 2)
    assert K.settlement_buffer([], listed) == (0.0, 0)
    old = [_prior(f"2025-12-0{i}T00:00:00Z", timedelta(days=9)) for i in (1, 2, 3)]
    assert K.settlement_buffer(old, listed) == (0.0, 0)  # closed more than 180 days earlier
    assert K.settlement_buffer(old, listed, lookback=timedelta(days=365)) == (9 * 24, 3)
    # expected before close (delays): clipped to 0, not negative
    neg = [_prior(f"2026-07-0{i}T00:00:00Z", timedelta(hours=-5)) for i in (1, 2, 3)]
    assert K.settlement_buffer(neg, listed) == (0.0, 3)
    # rows missing a time or a clean result are ignored
    assert K.settlement_buffer([{"close_time": "2026-07-01T00:00:00Z"}] * 5, listed) == (0.0, 0)
    void = [{**m, "result": ""} for m in old]
    assert K.settlement_buffer(void, listed, lookback=timedelta(days=365)) == (0.0, 0)


def test_market_to_question_applies_buffer():
    m = {
        **FED,
        "close_time": "2026-09-16T17:59:00Z",
        "expected_expiration_time": "2026-09-30T18:00:00Z",
    }
    q = K.market_to_question(m, FED_SERIES, buffer_h=14 * 24, n_prior=12)
    assert (q.close_at, q.scheduled_close_at) == ("2026-09-16T18:00:00Z", "")
    assert q.extra["scheduled_end"] == "2026-09-30T18:00:00Z"
    assert (q.extra["buffer_h"], q.extra["n_prior"]) == (336, 12)
    assert q.extra["early_close"] is False
    assert q.extra["close_gap_h"] == pytest.approx(1 / 60, abs=1e-3)


def test_list_candidates_buffer_from_prior_markets_only():
    base = {**FED, "event_ticker": "KXMENTION-E", "result": "no", "volume_fp": "500"}

    def mk(ticker, created, close, buffer=timedelta(days=14)):
        return {
            **base,
            "ticker": ticker,
            "created_time": created,
            "open_time": created,
            "close_time": close,
            "expected_expiration_time": to_iso(utc(close) + buffer),
            "settlement_ts": to_iso(utc(close) + timedelta(hours=1)),
        }

    prior = [mk(f"P{i}", "2026-05-01T00:00:00Z", f"2026-06-0{i}T00:00:00Z") for i in (1, 2, 3)]
    q1 = mk("Q1", "2026-07-02T00:00:00Z", "2026-08-01T00:00:00Z")
    q2 = mk("Q2", "2026-07-05T00:00:00Z", "2026-07-20T00:00:00Z", buffer=timedelta(days=30))
    fake = FakeKalshi(
        {
            "series": {"series": [{**FED_SERIES, "ticker": "KXMENTION"}]},
            "markets": {"markets": [], "cursor": ""},
            "historical/markets": {"markets": [q1, q2, *prior], "cursor": ""},
            "events": {"events": [], "cursor": ""},
        }
    )
    qs = {
        q.id: q
        for q in K.list_candidates(
            make_client(fake),
            created_after=T0,
            closed_before=T1,
            min_volume=0,
            series=["KXMENTION"],
        )
    }
    assert sorted(qs) == ["kalshi:Q1", "kalshi:Q2"]  # priors are never questions
    # Q1 listed after the 3 priors: buffer 14 d, anchor = actual close
    assert qs["kalshi:Q1"].close_at == "2026-08-01T00:00:00Z"
    assert qs["kalshi:Q1"].extra["n_prior"] == 3
    # Q2's own 30 d buffer is not used; Q1 (closed after Q2 was listed) is not a prior either
    assert qs["kalshi:Q2"].extra["buffer_h"] == 14 * 24
    assert qs["kalshi:Q2"].close_at == to_iso(utc("2026-08-19T00:00:00Z") - timedelta(days=14))


def test_make_client_defaults():
    c = K.make_client(cache_dir=None)
    assert c.max_retries == K.MAX_RETRIES
    assert c.interval_for(K.HOST) == K.HOST_INTERVAL
    assert K.make_client(max_retries=2).max_retries == 2


def test_taker_fee():
    assert K.taker_fee(0.5) == 0.02  # 0.0175 rounds up to the cent
    assert K.taker_fee(0.5, contracts=100) == 1.75
    assert K.taker_fee(0.5, contracts=100, multiplier=0.5) == 0.88


def test_scan_series_drops_short_frequencies_and_combos():
    series = {
        "A": {"frequency": "weekly", "category": "Crypto"},
        "B": {"frequency": "daily", "category": "Crypto"},
        "C": {"frequency": "hourly", "category": "Crypto"},
        "D": {"frequency": "one_off", "category": "Exotics"},
        "KXMVEX": {"frequency": "custom", "category": "Sports"},
        "E": {"frequency": "custom", "category": "Sports"},
    }
    assert K.scan_series(sorted(series), series) == ["A", "E"]


def test_parse_orderbook_yes_terms():
    body = {
        "orderbook_fp": {
            "yes_dollars": [["0.0500", "1.00"], ["0.1200", "181.27"]],
            "no_dollars": [["0.5300", "37.00"], ["0.8600", "1.00"], ["0.8500", "0.00"]],
        }
    }
    book = K.parse_orderbook(body, depth=10)
    assert book["bids"] == [[0.12, 181.27], [0.05, 1.0]]
    assert book["asks"] == [[0.14, 1.0], [0.47, 37.0]]


# -- network functions against a mock API ------------------------------------------------------


class FakeKalshi:
    """Routes by path; records requests. Unknown paths 404."""

    def __init__(self, routes: dict, cutoff: str = "2026-08-11T00:00:00Z"):
        self.routes = {"historical/cutoff": {"market_settled_ts": cutoff}, **routes}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        parts = urlsplit(str(request.url))
        path = parts.path.removeprefix(PREFIX)
        params = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.calls.append((path, params))
        route = self.routes.get(path)
        if route is None:
            return httpx.Response(404, text="not found")
        body = route(params) if callable(route) else route
        return httpx.Response(200, content=json.dumps(body))


def make_client(fake: FakeKalshi) -> HttpClient:
    return HttpClient(
        transport=httpx.MockTransport(fake), sleep=lambda s: None, min_interval=0, cache_dir=None
    )


def fed_question(**extra) -> object:
    q = K.market_to_question(FED, FED_SERIES)
    q.extra.update(extra)
    return q


def test_snapshot_live_tier_and_window():
    q = fed_question()
    close = utc(q.close_at)
    as_of = close - timedelta(days=1)
    a = int(as_of.timestamp())
    candles = [_candle(a - 120, "0.10", "0.12"), _candle(a, "0.13", "0.14")]
    path = "series/KXFEDDECISION/markets/KXFEDDECISION-26SEP-H0/candlesticks"
    fake = FakeKalshi({path: {"candlesticks": candles}})
    snap = K.snapshot(make_client(fake), q, "1d")
    assert snap is not None
    assert (snap.bid, snap.ask, snap.mid) == (0.13, 0.14, 0.135)
    assert snap.as_of == to_iso(as_of) == snap.price_at
    assert snap.source == "kalshi-candles-1m"
    _, params = next(c for c in fake.calls if c[0] == path)
    assert params["end_ts"] == str(a) and params["start_ts"] == str(a - 6 * 3600)
    assert params["period_interval"] == "1"


def test_snapshot_historical_tier_by_cutoff_and_404_fallback():
    q = fed_question(settled_at="2026-07-01T00:00:00Z")  # before cutoff -> historical first
    a = int((utc(q.close_at) - timedelta(days=7)).timestamp())
    hist = {"candlesticks": [{**HIST_CANDLE, "end_period_ts": a - 60}]}
    fake = FakeKalshi({"historical/markets/KXFEDDECISION-26SEP-H0/candlesticks": hist})
    snap = K.snapshot(make_client(fake), q, "7d")
    assert snap.mid == 0.01 and snap.ask == 0.02 and snap.bid is None
    assert [c[0] for c in fake.calls][-1].startswith("historical/")

    # cutoff says live, but the market already moved to the historical tier: 404 -> retry there
    q2 = fed_question()
    a2 = int((utc(q2.close_at) - timedelta(days=7)).timestamp())
    hist2 = {"candlesticks": [{**HIST_CANDLE, "end_period_ts": a2}]}
    fake2 = FakeKalshi({"historical/markets/KXFEDDECISION-26SEP-H0/candlesticks": hist2})
    assert K.snapshot(make_client(fake2), q2, "7d") is not None
    assert [c[0].split("/")[0] for c in fake2.calls[-2:]] == ["series", "historical"]


def test_snapshot_none_after_actual_close_and_when_stale():
    q = K.market_to_question(EARLY, SERIES)
    fake = FakeKalshi({})  # must not even ask for candles
    client = make_client(fake)
    for h in ("7d", "1d", "1h"):
        assert K.snapshot(client, q, h) is None
    assert all("candlesticks" not in c[0] for c in fake.calls)

    # 1h horizon exactly 60 s before the actual close also counts as closed
    edge = fed_question(
        actual_close=to_iso(utc(FED["expected_expiration_time"]) - timedelta(minutes=59))
    )
    assert K.snapshot(make_client(FakeKalshi({})), edge, "1h") is None

    path = "series/KXFEDDECISION/markets/KXFEDDECISION-26SEP-H0/candlesticks"
    assert (
        K.snapshot(make_client(FakeKalshi({path: {"candlesticks": []}})), fed_question(), "7d")
        is None
    )


def test_list_candidates_merges_tiers_filters_and_attaches_events():
    early_no = {  # created before created_after: dropped
        **FED,
        "ticker": "KXFEDDECISION-26SEP-H25",
        "result": "no",
    }
    good = {
        **FED,
        "ticker": "KXFEDDECISION-26OCT-H0",
        "event_ticker": "KXFEDDECISION-26OCT",
        "created_time": "2026-07-10T00:00:00Z",
        "open_time": "2026-07-10T00:00:00Z",
        "close_time": "2026-09-30T17:59:00Z",
        "expected_expiration_time": "2026-09-30T18:05:00Z",
    }
    old_hist = {
        **good,
        "ticker": "KXFEDDECISION-26JUN-H0",
        "close_time": "2026-06-15T00:00:00Z",
        "expected_expiration_time": "2026-06-15T00:00:00Z",
    }
    hist_good = {
        **good,
        "ticker": "KXFEDDECISION-26JUL-H0",
        "event_ticker": "KXFEDDECISION-26JUL",
        "close_time": "2026-07-29T17:59:00Z",
        "expected_expiration_time": "2026-07-29T18:05:00Z",
        "created_time": "2026-07-01T00:00:00Z",
        "open_time": "2026-07-01T00:00:00Z",
    }
    hist_pages = {
        "": {"markets": [hist_good, good], "cursor": "p2"},  # good again: deduped
        "p2": {"markets": [old_hist], "cursor": "p3"},  # all older than created_after: stop
        "p3": {"markets": [hist_good], "cursor": ""},
    }
    fake = FakeKalshi(
        {
            "series": {"series": [FED_SERIES]},
            "markets": {"markets": [good, early_no], "cursor": ""},
            "historical/markets": lambda p: hist_pages[p.get("cursor", "")],
            "events": {
                "events": [
                    {
                        "event_ticker": "KXFEDDECISION-26OCT",
                        "mutually_exclusive": True,
                        "title": "Fed?",
                    }
                ],
                "cursor": "",
            },
        }
    )
    client = make_client(fake)
    seen = []
    qs = list(
        K.list_candidates(
            client,
            created_after=T0,
            closed_before=T1,
            min_volume=0,
            series=["KXFEDDECISION"],
            progress=lambda d, t, n: seen.append((d, t, n)),
        )
    )
    ids = sorted(q.id for q in qs)
    assert ids == ["kalshi:KXFEDDECISION-26JUL-H0", "kalshi:KXFEDDECISION-26OCT-H0"]
    by_id = {q.id: q for q in qs}
    assert by_id["kalshi:KXFEDDECISION-26OCT-H0"].extra["mutually_exclusive"] is True
    assert "mutually_exclusive" not in by_id["kalshi:KXFEDDECISION-26JUL-H0"].extra
    assert seen == [(1, 1, 2)]
    hist_calls = [p for path, p in fake.calls if path == "historical/markets"]
    # first pass stops at p2 (all closed before created_after); the buffer pass goes back further
    assert [p.get("cursor", "") for p in hist_calls] == ["", "p2", "", "p2", "p3"]
    live_params = next(p for path, p in fake.calls if path == "markets")
    assert live_params["min_settled_ts"] == str(int(T0.timestamp()))
    assert live_params["status"] == "settled" and live_params["series_ticker"] == "KXFEDDECISION"
    assert client.interval_for(K.HOST) == K.HOST_INTERVAL


def test_list_candidates_skips_historical_tier_after_cutoff():
    fake = FakeKalshi(
        {"series": {"series": [FED_SERIES]}, "markets": {"markets": [], "cursor": ""}},
        cutoff="2026-06-01T00:00:00Z",
    )
    qs = list(
        K.list_candidates(
            make_client(fake), created_after=T0, closed_before=T1, min_volume=0, series=["X"]
        )
    )
    assert qs == []
    assert not any(path == "historical/markets" for path, _ in fake.calls)


def test_live_snapshot_one_sided_falls_back_to_clamped_last():
    book = {"orderbook_fp": {"yes_dollars": [["0.9900", "10"]], "no_dollars": []}}
    fake = FakeKalshi(
        {
            "markets/T-1/orderbook": book,
            "markets/T-1": {"market": {"last_price_dollars": "0.8700"}},
        }
    )
    snap = K.live_snapshot(make_client(fake), "kalshi:T-1")
    assert (snap.bid, snap.ask, snap.mid, snap.last) == (0.99, None, 0.99, 0.87)
    assert snap.book == {"bids": [[0.99, 10.0]], "asks": []}
    two = {"orderbook_fp": {"yes_dollars": [["0.12", "5"]], "no_dollars": [["0.86", "1"]]}}
    fake2 = FakeKalshi({"markets/T-2/orderbook": two})
    snap2 = K.live_snapshot(make_client(fake2), "T-2")
    assert (snap2.bid, snap2.ask, snap2.mid) == (0.12, 0.14, 0.13)
    assert all(path != "markets/T-2" for path, _ in fake2.calls)
