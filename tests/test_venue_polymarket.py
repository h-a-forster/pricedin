from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from pricedin.http import HttpClient
from pricedin.venues import polymarket as pm

# Trimmed from the live Gamma record for the Fed "no change, Sept 2026" market (2026-10-10).
FED = {
    "id": "2252244",
    "conditionId": "0xfed0",
    "question": "Will there be no change in Fed interest rates after the September 2026 meeting?",
    "description": (
        "This market will resolve to the amount of basis points the upper bound of the target "
        "federal funds rate is changed by.\n\nThe resolution source for this market is the FOMC "
        "statement after its meeting scheduled for September 15-16, 2026."
    ),
    "slug": "will-there-be-no-change-in-fed-interest-rates-after-the-september-2026-meeting-615",
    "createdAt": "2026-05-13T17:23:05.187964Z",
    "startDate": "2026-05-13T21:23:13.517067Z",
    "endDate": "2026-09-16T00:00:00Z",
    "closedTime": "2026-09-16 20:02:43+00",
    "closed": True,
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0", "1"]',
    "clobTokenIds": '["111", "222"]',
    "umaResolutionStatus": "resolved",
    "volumeNum": 1234567.8,
    "negRisk": True,
    "negRiskMarketID": "0xnr",
    "groupItemTitle": "No change",
    "feeType": "economics_fees",
    "feesEnabled": True,
    "feeSchedule": {"exponent": 1, "rate": 0.05, "takerOnly": True, "rebateRate": 0.25},
    "orderPriceMinTickSize": 0.001,
    "events": [
        {
            "id": "481717",
            "slug": "fed-decision-in-september-762",
            "title": "Fed Decision in September?",
        }
    ],
    "tags": [{"slug": s} for s in ["fomc", "fed-rates", "politics", "economy", "fed"]],
}


def market(**over) -> dict:
    m = copy.deepcopy(FED)
    m.update(over)
    return m


def T(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# --- parsing ---------------------------------------------------------------------------------


def test_parse_pm_time_formats():
    assert pm.parse_pm_time("2026-09-16 20:02:43+00") == T("2026-09-16T20:02:43")
    assert pm.parse_pm_time("2026-05-13T17:23:05.187964Z") == T("2026-05-13T17:23:05.187964")
    assert pm.parse_pm_time(None) is None
    assert pm.parse_pm_time("garbage") is None


def test_fed_market_parses_with_reference_close_at_end_date():
    q, why = pm.parse_market(market())
    assert why == "" and q is not None
    assert q.id == "polymarket:0xfed0" and q.event_id == "polymarket:481717"
    assert q.outcome == 0 and q.category == "economics" and q.venue_category == "economy"
    assert q.close_at == "2026-09-16T00:00:00Z"  # endDate, not closedTime
    assert q.scheduled_close_at == "2026-09-16T00:00:00Z"
    assert q.extra["actual_close"] == "2026-09-16T20:02:43Z"
    assert q.extra["early_close"] is False
    assert q.extra["yes_token"] == "111" and q.extra["no_token"] == "222"
    assert q.extra["neg_risk"] is True and q.extra["neg_risk_id"] == "0xnr"
    assert q.extra["tick_size"] == 0.001
    assert q.extra["fee"]["rate"] == 0.05 and q.extra["fee"]["taker_only"] is True
    assert q.open_at == "2026-05-13T21:23:13Z" and q.created_at == "2026-05-13T17:23:05Z"
    assert "rules_flag" not in q.extra


def test_yes_token_matched_by_name_not_position():
    q, _ = pm.parse_market(
        market(outcomes='["No", "Yes"]', outcomePrices='["0", "1"]', clobTokenIds='["n", "y"]')
    )
    assert q.extra["yes_token"] == "y" and q.extra["no_token"] == "n"
    assert q.outcome == 1


@pytest.mark.parametrize(
    "over,reason",
    [
        ({"outcomePrices": '["0.5", "0.5"]'}, "unclean_resolution"),
        ({"outcomePrices": '["0.9995", "0.0005"]'}, "unclean_resolution"),
        ({"outcomes": '["Up", "Down"]'}, "not_yes_no"),
        ({"outcomes": '["Over", "Under"]'}, "not_yes_no"),
        ({"clobTokenIds": '["111"]'}, "not_yes_no"),
        ({"umaResolutionStatus": "disputed"}, "uma_disputed"),
        ({"closedTime": None}, "missing_times"),
        ({"closed": False}, "open"),
    ],
)
def test_ineligible_markets_dropped(over, reason):
    q, why = pm.parse_market(market(**over))
    assert q is None and why == reason


def test_early_resolution_keeps_scheduled_close():
    q, _ = pm.parse_market(market(closedTime="2026-08-20 12:00:00+00", outcomePrices='["1","0"]'))
    assert q.close_at == "2026-09-16T00:00:00Z"
    assert q.extra["actual_close"] == "2026-08-20T12:00:00Z"
    assert q.scheduled_close_at == "2026-09-16T00:00:00Z"
    assert q.extra["early_close"] is True and q.extra["close_gap_h"] < -24
    assert q.outcome == 1


@pytest.mark.parametrize(
    "end,reason",
    [
        (None, "no_end_date"),
        ("2026-05-01T00:00:00Z", "end_date_before_open"),  # before createdAt and startDate
        ("2026-05-13T20:00:00Z", "end_date_before_open"),  # after createdAt, before startDate
    ],
)
def test_bad_or_missing_end_date_dropped(end, reason):
    q, why = pm.parse_market(market(endDate=end))
    assert q is None and why == reason


@pytest.mark.parametrize(
    "tags,cat",
    [
        (["politics", "economy"], "economics"),
        (["politics", "geopolitics", "iran"], "world"),
        (["tech", "finance", "big-tech"], "finance"),
        (["sports", "politics"], "sports"),
        (["esports"], "sports"),
        (["crypto", "finance"], "crypto"),
        (["pop-culture", "tweets-markets"], "culture"),
        (["elections"], "politics"),
        (["weather"], "weather"),
        (["mentions"], "other"),
        ([], "other"),
    ],
)
def test_category_precedence(tags, cat):
    assert pm.map_category(tags)[0] == cat


def test_fee_formula_matches_docs_table():
    fee = pm.fee_terms(market())
    assert pm.taker_fee(fee, 100, 0.5) == 1.25  # docs: economics, 100 shares at 50c
    assert pm.taker_fee(fee, 100, 0.05) == pytest.approx(0.2375)
    assert pm.taker_fee(pm.fee_terms(market(feesEnabled=False)), 100, 0.5) == 0.0
    assert pm.taker_fee(pm.fee_terms(market(feeSchedule=None)), 100, 0.5) == 0.0


def test_self_contained_title():
    sport = pm.self_contained_title("Will Henan FC win on 2026-07-17?", "Henan FC vs. Qingdao FC")
    assert sport == "Will Henan FC win on 2026-07-17? (Henan FC vs. Qingdao FC)"
    btc = "Will Bitcoin dip to $52,500 in July?"
    assert pm.self_contained_title(btc, "What price will Bitcoin hit in July?") == btc
    afd = "Will AfD win between 42% and 45% of all valid second votes?"
    assert "Sachsen-Anhalt" in pm.self_contained_title(
        afd, "Sachsen-Anhalt Parliamentary Election: AfD Vote Share?"
    )
    assert pm.self_contained_title("Q?", "") == "Q?"


# --- rules leak detector ---------------------------------------------------------------------


def test_rules_clean_text_untouched():
    text = FED["description"] + "\n\nNote: the source reports to one decimal point."
    assert pm.check_rules(text) == (text, [], False)


def test_rules_appended_update_is_stripped():
    text = "Resolves Yes if X happens.\n\nUpdate (Sep 3, 2026): The deadline has been extended."
    cleaned, flags, drop = pm.check_rules(text)
    assert cleaned == "Resolves Yes if X happens." and flags == ["stripped_appendix"] and not drop
    cleaned, flags, _ = pm.check_rules("Rules.\n\nClarification: officials only count.")
    assert cleaned == "Rules." and flags == ["stripped_appendix"]


def test_rules_resolution_statement_drops():
    _, flags, drop = pm.check_rules("Rules.\n\nThis market has resolved to Yes per the FDIC.")
    assert drop and "resolution_statement" in flags
    q, why = pm.parse_market(market(description="Rules.\nUpdate: market was resolved to No."))
    assert q is None and why == "rules_leak"


def test_rules_post_listing_reference_flagged():
    text = 'Rules.\n\nNote: This market refers to the video titled "X", found here.'
    _, flags, drop = pm.check_rules(text)
    assert flags == ["post_listing_reference"] and not drop
    q, _ = pm.parse_market(market(description=text))
    assert q.extra["rules_flag"] == ["post_listing_reference"]


def test_rules_words_inside_sentences_not_flagged():
    text = "If the FDIC list has not been updated yet, this market may remain open until updated."
    assert pm.check_rules(text) == (text, [], False)


# --- eligibility -----------------------------------------------------------------------------


def test_eligible_windows():
    q, _ = pm.parse_market(market())
    kw = dict(closed_before=T("2026-10-10T00:00:00"), min_open_days=8, min_volume=10_000)
    assert pm.eligible(q, created_after=T("2026-05-01T00:00:00"), **kw) == ""
    assert pm.eligible(q, created_after=T("2026-06-01T00:00:00"), **kw) == "created_before"
    kw2 = dict(kw, closed_before=T("2026-09-15T00:00:00"))
    assert pm.eligible(q, created_after=T("2026-05-01T00:00:00"), **kw2) == "closed_after"
    # Scheduled end before the cutoff but resolved after it: not resolved yet at the cutoff.
    kw3 = dict(kw, closed_before=T("2026-09-16T10:00:00"))
    assert pm.eligible(q, created_after=T("2026-05-01T00:00:00"), **kw3) == "closed_after"
    short, _ = pm.parse_market(market(startDate="2026-09-10T00:00:00Z"))
    assert pm.eligible(short, created_after=T("2026-05-01T00:00:00"), **kw) == "too_short"
    assert (
        pm.eligible(q, created_after=T("2026-05-01T00:00:00"), **dict(kw, min_volume=2e6))
        == "low_volume"
    )


# --- title deadline --------------------------------------------------------------------------

JUL_CREATED = T("2026-07-01T12:00:00")
JUL_END = T("2026-07-31T23:59:00")


@pytest.mark.parametrize(
    "title, created, end, expected",
    [
        (
            "Israel military action against Yemen by July 21?",
            JUL_CREATED,
            JUL_END,
            "2026-07-22T04:59:00Z",
        ),
        (
            "Will Donald Trump publicly insult someone on July 2, 2026?",
            JUL_CREATED,
            JUL_END,
            "2026-07-03T04:59:00Z",
        ),
        (
            "Will US crude oil reserves fall to 280M by September 25, 2026?",
            JUL_CREATED,
            T("2026-10-08T00:00:00"),
            "2026-09-26T04:59:00Z",
        ),
        ("Deal signed before Friday, Sept. 4th?", JUL_CREATED, JUL_END, "2026-09-05T04:59:00Z"),
        ("Will it rain on July 2-3?", JUL_CREATED, JUL_END, "2026-07-04T04:59:00Z"),
        # Dec -> Jan wrap: a December listing asking "by January 5" means next year.
        (
            "Ceasefire by January 5?",
            T("2026-12-10T00:00:00"),
            T("2027-01-06T05:00:00"),
            "2027-01-06T04:59:00Z",
        ),
        (
            "Ceasefire by December 31?",
            T("2026-12-10T00:00:00"),
            T("2027-01-01T05:00:00"),
            "2027-01-01T04:59:00Z",
        ),
        # Period questions and dates without a deadline word are not deadlines.
        ("Will US CPI rise in July 2026?", JUL_CREATED, JUL_END, None),
        ("Will Bitcoin dip to $52,500 in July?", JUL_CREATED, JUL_END, None),
        ("Fed decision on July 2026 meeting?", JUL_CREATED, JUL_END, None),
        ("Will Henan FC win on 2026-07-17?", JUL_CREATED, JUL_END, None),
        ("Who wins the July 21 primary?", JUL_CREATED, JUL_END, None),
        ("Nothing by Feb 30?", JUL_CREATED, JUL_END, None),
    ],
)
def test_title_deadline(title, created, end, expected):
    d = pm.title_deadline(title, end, created)
    assert (pm.to_iso(d) if d else None) == expected


def test_title_deadline_recorded_and_enforced():
    kw = dict(
        created_after=T("2026-05-01T00:00:00"),
        closed_before=T("2026-10-10T00:00:00"),
        min_open_days=8,
        min_volume=10_000,
    )
    # endDate the day after the named day (00:00 UTC): within the 1-day slack, still eligible.
    ok, _ = pm.parse_market(market(question="Fed cut by September 15?"))
    assert ok.extra["title_deadline"] == "2026-09-16T04:59:00Z"
    assert pm.eligible(ok, **kw) == ""
    late, _ = pm.parse_market(market(question="Fed cut by September 10?"))
    assert late.extra["title_deadline"] == "2026-09-11T04:59:00Z"
    assert pm.eligible(late, **kw) == "title_deadline_before_end"
    # The rule works from a saved Question (title + close_at + created_at), not from extra.
    late.extra.pop("title_deadline")
    assert pm.eligible(late, **kw) == "title_deadline_before_end"
    none, _ = pm.parse_market(market())
    assert "title_deadline" not in none.extra


# --- prices ----------------------------------------------------------------------------------


def test_pick_point_boundaries():
    as_of = T("2026-09-15T00:00:00")
    ts = as_of.timestamp()
    hist = [
        {"t": ts - 7 * 3600, "p": 0.1},
        {"t": ts - 61, "p": 0.2},
        {"t": ts - 60, "p": 0.3},  # last usable: t <= as_of - 60 s
        {"t": ts - 59, "p": 0.4},
        {"t": ts, "p": 0.5},
        {"t": ts + 30, "p": 0.6},
    ]
    when, p = pm.pick_point(hist, as_of)
    assert p == 0.3 and when == as_of - timedelta(seconds=60)
    assert pm.pick_point([{"t": ts - 6 * 3600, "p": 0.7}], as_of) == (
        as_of - timedelta(hours=6),
        0.7,
    )
    assert pm.pick_point([{"t": ts - 6 * 3600 - 1, "p": 0.7}], as_of) is None
    assert pm.pick_point([{"t": ts + 1, "p": 0.7}, {"t": "x", "p": 1}], as_of) is None
    assert pm.pick_point([], as_of) is None
    # bad prices are skipped, not used: fall back to the last valid point
    bad = [{"t": ts - 120, "p": 0.25}, {"t": ts - 90, "p": "nan"}, {"t": ts - 61, "p": 1.5}]
    assert pm.pick_point(bad, as_of) == (as_of - timedelta(seconds=120), 0.25)


def mock_client(handler, **kw) -> HttpClient:
    return HttpClient(
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
        min_interval=0.0,
        host_intervals={h: 0.0 for h in pm.HOST_INTERVALS},
        seed=0,
        **kw,
    )


def test_snapshot_requests_window_and_picks_point():
    q, _ = pm.parse_market(market())
    as_of = T("2026-09-15T00:00:00")  # close_at 09-16 00:00 minus 1d
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(dict(req.url.params))
        ts = as_of.timestamp()
        return httpx.Response(
            200, json={"history": [{"t": ts - 130, "p": 0.125}, {"t": ts - 10, "p": 0.9}]}
        )

    snap = pm.snapshot(mock_client(handler), q, "1d")
    assert snap.mid == 0.125 and snap.bid is None and snap.ask is None
    assert snap.as_of == "2026-09-15T00:00:00Z" and snap.price_at == "2026-09-14T23:57:50Z"
    assert snap.source == "polymarket-prices-history"
    p = seen[0]
    assert p["market"] == "111" and p["fidelity"] == "1"
    assert int(p["endTs"]) == int(as_of.timestamp())
    assert int(p["startTs"]) == int((as_of - timedelta(hours=6)).timestamp())


def test_snapshot_none_when_empty_or_before_open():
    q, _ = pm.parse_market(market())
    client = mock_client(lambda req: httpx.Response(200, json={"history": []}))
    assert pm.snapshot(client, q, "7d") is None
    late, _ = pm.parse_market(market(startDate="2026-09-15T12:00:00Z"))
    assert pm.snapshot(client, late, "1d") is None  # as_of before trading opened


def test_snapshot_none_once_market_already_closed():
    # Resolved YES early on 09-15 12:00; close_at stays at the scheduled 09-16 00:00.
    q, _ = pm.parse_market(market(closedTime="2026-09-15 12:00:00+00", outcomePrices='["1","0"]'))
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        end = int(req.url.params["endTs"])
        return httpx.Response(200, json={"history": [{"t": end - 120, "p": 0.3}]})

    client = mock_client(handler)
    assert pm.snapshot(client, q, "1h") is None  # as_of 09-15 23:00 > actual close
    assert pm.snapshot(client, q, "1d") is not None  # as_of 09-15 00:00 < actual close
    assert len(calls) == 1  # no request for an unobservable row
    # Boundary: as_of must be earlier than actual_close - 60 s.
    edge = copy.deepcopy(q)
    edge.extra["actual_close"] = "2026-09-15T23:01:00Z"  # as_of(1h) = actual - 60 s
    assert pm.snapshot(client, edge, "1h") is None
    edge.extra["actual_close"] = "2026-09-15T23:01:01Z"
    assert pm.snapshot(client, edge, "1h") is not None


# --- listing ---------------------------------------------------------------------------------


def test_list_candidates_paginates_dedupes_and_filters():
    up = market(id="9", conditionId="0xup", outcomes='["Up", "Down"]')
    fifty = market(id="8", conditionId="0x50", outcomePrices='["0.5", "0.5"]')
    early = market(id="7", conditionId="0xear", closedTime="2026-08-01 00:00:00+00")
    pages = {
        None: {"markets": [market(), up], "next_cursor": "c1"},
        "c1": {"markets": [fifty, early, market()], "next_cursor": None},
    }
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        params = dict(req.url.params)
        seen.append(params)
        return httpx.Response(200, json=pages[params.get("after_cursor")])

    reasons: dict[str, int] = {}
    qs = list(
        pm.list_candidates(
            mock_client(handler),
            created_after=T("2026-05-01T00:00:00"),
            closed_before=T("2026-10-10T00:00:00"),
            min_volume=10_000,
            reasons=reasons,
        )
    )
    assert seen[0]["closed"] == "true" and seen[0]["include_tag"] == "true"
    assert seen[0]["start_date_min"] == "2026-05-01T00:00:00Z"
    assert seen[0]["start_date_max"] == "2026-05-08T00:00:00Z"  # 7-day slices
    assert seen[-1]["start_date_max"] == "2026-10-02T00:00:00Z"  # closed_before - 8 days
    assert seen[0]["volume_num_min"] == "10000"
    # The fake server returns the same pages for every slice: each market is still seen once.
    assert reasons == {"ok": 2, "not_yes_no": 1, "unclean_resolution": 1}
    assert [q.id for q in qs] == ["polymarket:0xfed0", "polymarket:0xear"]
    assert qs[1].close_at == "2026-09-16T00:00:00Z" and qs[1].extra["early_close"]


def test_iter_markets_bisects_failing_window():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        p = dict(req.url.params)
        a, b = pm.parse_pm_time(p["start_date_min"]), pm.parse_pm_time(p["start_date_max"])
        calls.append((a, b))
        if b - a > timedelta(days=2):
            return httpx.Response(500)
        return httpx.Response(200, json={"markets": [{"id": str(a)}], "next_cursor": None})

    client = mock_client(handler, max_retries=0)
    start = T("2026-07-01T00:00:00")
    out = list(pm.iter_markets(client, start, start + timedelta(days=7)))
    assert len(out) == 4  # 7d -> 3.5d halves -> 1.75d quarters
    assert sorted(x["id"] for x in out) == sorted({x["id"] for x in out})


# --- live book -------------------------------------------------------------------------------


def test_book_snapshot_orders_levels_best_first():
    body = {
        "bids": [{"price": "0.01", "size": "100"}, {"price": "0.10", "size": "5"}],
        "asks": [{"price": "0.99", "size": "7"}, {"price": "0.12", "size": "3"}],
        "timestamp": "1791627362592",
        "tick_size": "0.01",
        "last_trade_price": "0.110",
    }
    s = pm.book_snapshot(body, "polymarket:0xabc")
    assert s.bid == 0.10 and s.ask == 0.12 and s.mid == pytest.approx(0.11)
    assert s.book["bids"][0] == [0.10, 5.0] and s.book["asks"][0] == [0.12, 3.0]
    assert s.source == "live-book" and s.horizon == "live" and s.last == 0.11
    empty = pm.book_snapshot({"bids": [], "asks": [], "last_trade_price": "0.4"}, "x")
    assert empty.bid is None and empty.mid == 0.4


def test_book_snapshot_drops_bad_levels_and_timestamp():
    now = T("2026-09-15T00:00:00")
    body = {
        "bids": [
            {"price": "nan", "size": "1"},
            {"price": "1.2", "size": "1"},
            {"price": "0.30", "size": "0"},
            {"price": "0.20", "size": "4"},
            {"size": "9"},
        ],
        "asks": [{"price": "0.25", "size": "inf"}, {"price": "0.26", "size": "2"}],
        "timestamp": "not-a-number",
        "last_trade_price": "7",
    }
    s = pm.book_snapshot(body, "x", now=now)
    assert s.book["bids"] == [[0.20, 4.0]] and s.book["asks"] == [[0.26, 2.0]]
    assert s.mid == pytest.approx(0.23) and s.as_of == pm.to_iso(now)
    one_sided = pm.book_snapshot(
        {"bids": [{"price": "0.2", "size": "1"}], "asks": [], "last_trade_price": "7"}, "x", now=now
    )
    assert one_sided.mid == 0.2  # out-of-range last trade ignored


def test_live_snapshot_resolves_condition_id():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/markets":
            assert dict(req.url.params) == {"condition_ids": "0xfed0"}
            return httpx.Response(200, json=[market()])
        assert dict(req.url.params) == {"token_id": "111"}
        return httpx.Response(
            200,
            json={
                "bids": [{"price": "0.4", "size": "1"}],
                "asks": [{"price": "0.42", "size": "2"}],
            },
        )

    s = pm.live_snapshot(mock_client(handler), "polymarket:0xfed0")
    assert s.question_id == "polymarket:0xfed0" and s.mid == pytest.approx(0.41)


def test_tune_keeps_explicit_overrides():
    c = HttpClient(host_intervals={"clob.polymarket.com": 0.5})
    pm._tune(c)
    assert c.interval_for("clob.polymarket.com") == 0.5
    assert (
        c.interval_for("gamma-api.polymarket.com") == pm.HOST_INTERVALS["gamma-api.polymarket.com"]
    )
    assert json.dumps(pm.fee_terms(market()))  # serialisable for JSONL


def test_collect_reuse_reapplies_polymarket_eligibility(tmp_path):
    from pricedin import cli
    from pricedin.schema import Question, read_jsonl, write_jsonl

    toml = tmp_path / "studies" / "t.toml"
    toml.parent.mkdir()
    toml.write_text(
        '[study]\nname = "t"\ncreated_after = "2026-05-01T00:00:00Z"\n'
        'closed_before = "2026-10-10T00:00:00Z"\ndata_dir = "../data"\nresults_dir = "../res"\n'
        '[universe]\nvenues = ["polymarket"]\nper_cell = 10\n',
        encoding="utf-8",
    )
    ok, _ = pm.parse_market(market(question="Fed cut by September 15?"))
    late, _ = pm.parse_market(market(question="Fed cut by September 10?", conditionId="0xl"))
    late.extra.pop("title_deadline")  # saved before the rule existed
    (tmp_path / "data").mkdir()
    cand = tmp_path / "data" / "candidates.jsonl.gz"
    write_jsonl(cand, [ok, late])
    assert cli.main(["collect", str(toml), "--reuse"]) == 0
    chosen = list(read_jsonl(tmp_path / "data" / "questions.jsonl", Question))
    assert [q.id for q in chosen] == [ok.id]
    assert len(list(read_jsonl(cand, Question))) == 2  # saved candidates are left untouched
