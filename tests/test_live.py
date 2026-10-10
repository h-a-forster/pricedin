from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from pricedin import live
from pricedin.eval.panel import build_panel
from pricedin.http import HttpClient
from pricedin.live import (
    LiveConfig,
    LockBusy,
    MarketStatus,
    PassLock,
    run_pass,
)
from pricedin.schema import (
    Forecast,
    NewsContext,
    Question,
    Snapshot,
    read_jsonl,
    to_iso,
    write_jsonl,
)

T0 = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
ANCHOR = T0 + timedelta(days=7, hours=12)  # 7d target = T0 + 12 h


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def make_q(i: int = 0, anchor: datetime = ANCHOR, **kw) -> Question:
    base = dict(
        id=f"kalshi:M{i}",
        venue="kalshi",
        event_id=f"kalshi:E{i}",
        title=f"Will thing {i} happen?",
        rules="Resolves YES if it happens.",
        category="politics",
        created_at="2026-10-01T00:00:00Z",
        open_at="2026-10-01T00:00:00Z",
        close_at=to_iso(anchor),
        scheduled_close_at=to_iso(anchor),
        outcome=-1,
        volume=5000.0,
        extra={"first_seen": to_iso(T0)},
    )
    base.update(kw)
    return Question(**base)


class FakeVenue:
    """In-memory venue: listed markets, a mid per market, a status per market."""

    def __init__(self, clock: Clock, markets=(), name: str = "kalshi"):
        self.name = name
        self.clock = clock
        self.markets: dict[str, Question] = {q.id: q for q in markets}
        self.mid: dict[str, float] = {}
        self.statuses: dict[str, MarketStatus] = {}
        self.snapshot_calls = 0
        self.status_calls = 0

    def list_open(self, client, cfg, now):
        seen = {"first_seen": to_iso(now)}
        return [replace(q, extra={**q.extra, **seen}) for q in self.markets.values()]

    def snapshot(self, client, q):
        self.snapshot_calls += 1
        now = to_iso(self.clock())
        return Snapshot(
            q.id, "live", now, self.mid.get(q.id, 0.6), 0.59, 0.61, price_at=now, source="fake"
        )

    def status(self, client, q, cfg):
        self.status_calls += 1
        if q.id in self.statuses:
            return self.statuses[q.id]
        return MarketStatus("open", anchor=self.markets[q.id].close_at)


def fake_news(questions, snaps, out: Path):
    rows = [NewsContext(s.question_id, s.horizon, s.as_of, "q", [], "fake") for s in snaps]
    write_jsonl(out, rows, append=True)
    return {"rows_written": len(rows)}


def make_cfg(**kw) -> LiveConfig:
    base = dict(
        venues=["kalshi"],
        horizons=["7d", "1d", "1h"],
        created_after=datetime(2026, 7, 1, tzinfo=UTC),
        closed_before=datetime(2027, 7, 1, tzinfo=UTC),
        max_days=8,
        min_open_days=0,
        min_volume={"kalshi": 1000},
        forecasters=["constant:0.7"],
        conditions=["base"],
        workers=1,
    )
    base.update(kw)
    return LiveConfig(**base)


def go(cfg, data: Path, clock: Clock, venue: FakeVenue, *, dry_run=False):
    return run_pass(
        cfg,
        data,
        dry_run=dry_run,
        clock=clock,
        client=object(),  # never used by the fakes
        adapters={"kalshi": venue},
        news_fn=fake_news,
        cache_dir=data / "cache",
    )


def horizon_rows(data: Path) -> list[dict]:
    return list(read_jsonl(data / "horizons.jsonl"))


def snaps(data: Path) -> list[Snapshot]:
    p = data / "snapshots.jsonl"
    return list(read_jsonl(p, Snapshot)) if p.exists() else []


# --- window logic -----------------------------------------------------------------------


def test_snapshot_taken_inside_tolerance(tmp_path):
    clock = Clock(ANCHOR - timedelta(days=7) + timedelta(minutes=20))
    venue = FakeVenue(clock, [make_q()])
    s = go(make_cfg(), tmp_path, clock, venue, dry_run=True)
    assert s["admitted"] == 1 and s["taken"] == 1
    (snap,) = snaps(tmp_path)
    assert snap.horizon == "7d" and snap.as_of == to_iso(clock.t) and snap.source == "fake"
    (row,) = horizon_rows(tmp_path)
    assert row["status"] == "taken" and row["lag_min"] == 20.0
    assert row["target"] == to_iso(ANCHOR - timedelta(days=7))


def test_nothing_due_outside_tolerance(tmp_path):
    clock = Clock(ANCHOR - timedelta(days=7) - timedelta(minutes=40))
    venue = FakeVenue(clock, [make_q()])
    s = go(make_cfg(), tmp_path, clock, venue, dry_run=True)
    assert s["taken"] == 0 and s["missed"] == 0 and venue.snapshot_calls == 0
    assert snaps(tmp_path) == []


def test_late_pass_misses_and_never_backfills(tmp_path):
    cfg = make_cfg()
    clock = Clock(ANCHOR - timedelta(days=7) - timedelta(hours=1))
    venue = FakeVenue(clock, [make_q()])
    go(cfg, tmp_path, clock, venue, dry_run=True)  # tracked, nothing due yet
    clock.t = ANCHOR - timedelta(days=7) + timedelta(minutes=36)  # window closed
    s = go(cfg, tmp_path, clock, venue, dry_run=True)
    assert s["missed"] == 1 and s["taken"] == 0 and venue.snapshot_calls == 0
    (row,) = horizon_rows(tmp_path)
    assert row["status"] == "missed" and row["reason"] == "no_pass_in_window"
    go(cfg, tmp_path, clock, venue, dry_run=True)
    assert len(horizon_rows(tmp_path)) == 1 and snaps(tmp_path) == []


def test_market_first_seen_after_a_window_logs_that_horizon_missed(tmp_path):
    clock = Clock(ANCHOR - timedelta(days=1) + timedelta(minutes=5))
    venue = FakeVenue(clock, [make_q()])
    s = go(make_cfg(), tmp_path, clock, venue, dry_run=True)
    rows = {r["horizon"]: r for r in horizon_rows(tmp_path)}
    assert rows["7d"]["status"] == "missed" and rows["7d"]["reason"] == "tracked_after_window"
    assert rows["1d"]["status"] == "taken" and s["taken"] == 1
    assert "1h" not in rows


def test_snapshot_error_is_retried_within_window(tmp_path):
    cfg = make_cfg()
    clock = Clock(ANCHOR - timedelta(days=7) - timedelta(minutes=30))
    venue = FakeVenue(clock, [make_q()])
    venue.mid["kalshi:M0"] = float("nan")
    s = go(cfg, tmp_path, clock, venue, dry_run=True)
    assert s["snapshot_errors"] == 1 and s["taken"] == 0
    venue.mid["kalshi:M0"] = 0.4
    clock.t += timedelta(minutes=30)
    s = go(cfg, tmp_path, clock, venue, dry_run=True)
    assert s["taken"] == 1 and snaps(tmp_path)[0].mid == 0.4


# --- admission and anchors --------------------------------------------------------------


def test_admission_filters_and_caps(tmp_path):
    clock = Clock(T0)
    qs = [
        make_q(1),
        make_q(2, volume=10.0),  # low volume
        make_q(3, created_at="2026-06-01T00:00:00Z"),  # before the cutoff
        make_q(4, anchor=T0 + timedelta(days=20)),  # beyond max_days
        make_q(5, event_id="kalshi:E1"),  # same event as M1
        make_q(6),
        make_q(7),
    ]
    venue = FakeVenue(clock, qs)
    s = go(make_cfg(per_category=2), tmp_path, clock, venue, dry_run=True)
    tracked = {q.id for q in read_jsonl(tmp_path / "tracked.jsonl", Question)}
    assert len(tracked) == 2
    assert not tracked & {"kalshi:M2", "kalshi:M3", "kalshi:M4"}
    a = s["admission"]
    assert a["low_volume"] == 1 and a["created_before"] == 1 and a["outside_max_days"] == 1
    assert a["admitted"] == 2 and a.get("category_cap", 0) + a.get("event_cap", 0) == 2


def test_require_all_horizons(tmp_path):
    clock = Clock(ANCHOR - timedelta(days=1))
    venue = FakeVenue(clock, [make_q()])
    s = go(make_cfg(require_all_horizons=True), tmp_path, clock, venue, dry_run=True)
    assert s["admitted"] == 0 and s["admission"] == {"first_horizon_passed": 1}


def test_max_tracked(tmp_path):
    clock = Clock(T0)
    venue = FakeVenue(clock, [make_q(i) for i in range(5)])
    s = go(make_cfg(max_tracked=3), tmp_path, clock, venue, dry_run=True)
    assert s["admitted"] == 3 and s["admission"]["max_tracked"] == 2


def test_anchor_is_frozen_and_changes_are_recorded(tmp_path):
    cfg = make_cfg()
    clock = Clock(T0)
    venue = FakeVenue(clock, [make_q()])
    go(cfg, tmp_path, clock, venue, dry_run=True)
    moved = ANCHOR + timedelta(days=2)
    venue.markets["kalshi:M0"] = make_q(anchor=moved, title="edited title")
    s = go(cfg, tmp_path, clock, venue, dry_run=True)
    assert s["anchor_changes"] == 1
    (q,) = read_jsonl(tmp_path / "tracked.jsonl", Question)
    assert q.close_at == to_iso(ANCHOR) and q.title == "Will thing 0 happen?"
    (u,) = read_jsonl(tmp_path / "updates.jsonl")
    assert u["kind"] == "anchor_change" and u["new"] == to_iso(moved)
    go(cfg, tmp_path, clock, venue, dry_run=True)  # same new anchor: not logged again
    assert len(list(read_jsonl(tmp_path / "updates.jsonl"))) == 1
    # horizons stay on the frozen anchor
    clock.t = ANCHOR - timedelta(days=7)
    assert go(cfg, tmp_path, clock, venue, dry_run=True)["taken"] == 1


# --- idempotence, crash safety, lock ----------------------------------------------------


def test_rerunning_a_pass_is_harmless(tmp_path):
    cfg = make_cfg()
    clock = Clock(ANCHOR - timedelta(days=7))
    venue = FakeVenue(clock, [make_q(0), make_q(1)])
    go(cfg, tmp_path, clock, venue)
    names = ["tracked.jsonl", "snapshots.jsonl", "horizons.jsonl", "news.jsonl.gz"]
    before = {n: (tmp_path / n).read_bytes() for n in names}
    fc = list(read_jsonl(live.forecast_file(live.State(tmp_path), "constant:0.7", "base")))
    s = go(cfg, tmp_path, clock, venue)
    assert s["taken"] == 0 and s["admitted"] == 0 and s["forecast_tasks"] == 0
    assert {n: (tmp_path / n).read_bytes() for n in names} == before
    fc2 = list(read_jsonl(live.forecast_file(live.State(tmp_path), "constant:0.7", "base")))
    assert fc2 == fc and len(fc) == 2


def test_partial_last_line_is_repaired(tmp_path):
    cfg = make_cfg()
    clock = Clock(ANCHOR - timedelta(days=7))
    venue = FakeVenue(clock, [make_q()])
    go(cfg, tmp_path, clock, venue, dry_run=True)
    with open(tmp_path / "horizons.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"question_id": "kalshi:M0", "hor')  # a crash mid-append
    go(cfg, tmp_path, clock, venue, dry_run=True)
    assert len(horizon_rows(tmp_path)) == 1


def test_lock_blocks_a_concurrent_pass(tmp_path):
    clock = Clock(T0)
    venue = FakeVenue(clock, [make_q()])
    with PassLock(tmp_path / live.LOCK_NAME, timedelta(hours=3)), pytest.raises(LockBusy):
        go(make_cfg(), tmp_path, clock, venue, dry_run=True)
    assert not (tmp_path / live.LOCK_NAME).exists()  # released
    go(make_cfg(), tmp_path, clock, venue, dry_run=True)  # runs once the lock is gone


def test_stale_lock_is_broken(tmp_path):
    lock = tmp_path / live.LOCK_NAME
    lock.write_text("{}", encoding="utf-8")
    old = time.time() - 4 * 3600
    os.utime(lock, (old, old))
    with PassLock(lock, timedelta(hours=3)):
        assert lock.exists()
    assert not lock.exists()


# --- forecasts, budget, dry run ---------------------------------------------------------


def test_dry_run_makes_no_forecasts(tmp_path):
    clock = Clock(ANCHOR - timedelta(days=7))
    venue = FakeVenue(clock, [make_q()])
    s = go(make_cfg(), tmp_path, clock, venue, dry_run=True)
    assert s["taken"] == 1 and s["forecast_tasks"] == 0 and s["forecasts"] == []
    assert not (tmp_path / "forecasts_raw").exists()
    assert len(list(read_jsonl(tmp_path / "news.jsonl.gz"))) == 1


def test_forecast_uses_snapshot_time_and_logs_cost(tmp_path):
    clock = Clock(ANCHOR - timedelta(days=7) + timedelta(minutes=12))
    venue = FakeVenue(clock, [make_q()])
    s = go(make_cfg(conditions=["base", "news"]), tmp_path, clock, venue)
    assert s["forecast_tasks"] == 2
    st = live.State(tmp_path)
    for cond in ("base", "news"):
        (f,) = read_jsonl(live.forecast_file(st, "constant:0.7", cond), Forecast)
        assert f.prob == 0.7 and f.as_of == to_iso(clock.t) and f.horizon == "7d"
    ledger = list(read_jsonl(tmp_path / "ledger.jsonl"))
    assert [r["condition"] for r in ledger] == ["base", "news"] and ledger[0]["done"] == 1


def test_budget_caps_forecasts_per_pass(tmp_path):
    cfg = make_cfg(max_forecasts_per_pass=1)
    clock = Clock(ANCHOR - timedelta(days=7))
    venue = FakeVenue(clock, [make_q(0), make_q(1)])
    s = go(cfg, tmp_path, clock, venue)
    assert s["taken"] == 2 and s["forecast_tasks"] == 1
    clock.t += timedelta(minutes=20)  # next pass, still within the forecast grace
    s = go(cfg, tmp_path, clock, venue)
    assert s["forecast_tasks"] == 1
    path = live.forecast_file(live.State(tmp_path), "constant:0.7", "base")
    assert {f.question_id for f in read_jsonl(path, Forecast)} == {"kalshi:M0", "kalshi:M1"}


def test_forecast_not_made_in_time_is_logged_missed(tmp_path):
    cfg = make_cfg()
    clock = Clock(ANCHOR - timedelta(days=7))
    venue = FakeVenue(clock, [make_q()])
    go(cfg, tmp_path, clock, venue, dry_run=True)
    clock.t += timedelta(hours=1)  # past the grace: no late forecast
    s = go(cfg, tmp_path, clock, venue)
    assert s["forecast_tasks"] == 0 and s["forecast_missed"] == 1
    assert not live.forecast_file(live.State(tmp_path), "constant:0.7", "base").exists()
    assert go(cfg, tmp_path, clock, venue)["forecast_missed"] == 0  # logged once


# --- resolution and the evaluate view ---------------------------------------------------


def test_resolution_writes_an_evaluable_view(tmp_path):
    cfg = make_cfg()
    clock = Clock(ANCHOR - timedelta(days=7))
    venue = FakeVenue(clock, [make_q(0), make_q(1)])
    go(cfg, tmp_path, clock, venue)
    clock.t = ANCHOR + timedelta(hours=2)
    venue.statuses["kalshi:M0"] = MarketStatus(
        "resolved", 1, to_iso(ANCHOR - timedelta(days=3)), settled_at=to_iso(ANCHOR)
    )
    venue.statuses["kalshi:M1"] = MarketStatus("void", detail="finalized/''")
    s = go(cfg, tmp_path, clock, venue)
    assert s["resolved"] == 1 and s["void"] == 1
    (q,) = read_jsonl(tmp_path / "questions.jsonl", Question)
    assert q.id == "kalshi:M0" and q.outcome == 1 and q.close_at == to_iso(ANCHOR)
    assert q.extra["early_close"] is True and q.extra["live"] is True
    # the 1d and 1h windows came after the early close
    rows = {
        r["horizon"]: r["reason"]
        for r in horizon_rows(tmp_path)
        if r["status"] == "missed" and r["question_id"] == "kalshi:M0"
    }
    assert rows == {"1d": "closed_before_target", "1h": "closed_before_target"}
    views = sorted((tmp_path / "forecasts").glob("*.jsonl.gz"))
    assert [p.name for p in views] == ["constant_0.7__base.jsonl.gz"]
    panel = build_panel(
        list(read_jsonl(tmp_path / "questions.jsonl", Question)),
        snaps(tmp_path),
        list(read_jsonl(views[0], Forecast)),
        horizons=["7d", "1d", "1h"],
    )
    assert len(panel) == 1 and panel.y[0] == 1
    # resolved markets are no longer checked; the view is rewritten only on change
    calls, mtime = venue.status_calls, (tmp_path / "questions.jsonl").stat().st_mtime_ns
    go(cfg, tmp_path, clock, venue)
    assert venue.status_calls == calls
    assert (tmp_path / "questions.jsonl").stat().st_mtime_ns == mtime


def test_market_gone_from_listing_is_checked_early(tmp_path):
    cfg = make_cfg()
    clock = Clock(T0)
    venue = FakeVenue(clock, [make_q()])
    go(cfg, tmp_path, clock, venue, dry_run=True)
    assert venue.status_calls == 0  # listed and before its anchor: not checked
    venue.statuses["kalshi:M0"] = MarketStatus("closed", actual_close=to_iso(T0))
    venue.markets.clear()
    go(cfg, tmp_path, clock, venue, dry_run=True)
    (u,) = read_jsonl(tmp_path / "updates.jsonl")
    assert u["kind"] == "closed" and venue.status_calls == 1


def test_views_are_deterministic_gzip(tmp_path):
    st = live.State(tmp_path)
    st.raw_dir.mkdir(parents=True)
    f = Forecast("kalshi:M0", "7d", to_iso(T0), "constant:0.7", "base", 0.7)
    write_jsonl(st.raw_dir / "x__base.jsonl", [f])
    st.load()
    assert live.write_views(st)["x__base.jsonl.gz"] is True
    data = (st.view_dir / "x__base.jsonl.gz").read_bytes()
    assert live.write_views(st)["x__base.jsonl.gz"] is False
    assert json.loads(gzip.decompress(data))["prob"] == 0.7


# --- venue parsing (no network) ---------------------------------------------------------

KALSHI_MARKET = {
    "ticker": "KXUCLGAME-26NOV04ATMMUN-ATM",
    "event_ticker": "KXUCLGAME-26NOV04ATMMUN",
    "market_type": "binary",
    "status": "active",
    "created_time": "2026-10-20T00:00:00Z",
    "open_time": "2026-10-20T00:05:00Z",
    "close_time": "2026-11-06T19:00:00Z",
    "expected_expiration_time": "2026-11-04T22:00:00Z",
    "latest_expiration_time": "2026-11-06T19:00:00Z",
    "volume_fp": "2500.00",
    "title": "Atletico Madrid vs Manchester United",
    "yes_sub_title": "Atletico Madrid",
    "rules_primary": "If Atletico Madrid wins, the market resolves to Yes.",
    "result": "",
}
SERIES = {"ticker": "KXUCLGAME", "category": "Sports", "tags": ["Soccer"], "frequency": "custom"}


def test_kalshi_open_question_anchors_on_the_earlier_close():
    q = live.kalshi_open_question(KALSHI_MARKET, SERIES)
    assert q is not None and q.outcome == -1 and q.category == "sports"
    assert q.close_at == "2026-11-04T22:00:00Z" == q.scheduled_close_at
    assert q.extra["anchor_source"] == "expected_expiration_time"
    assert q.extra["listed_close_time"] == "2026-11-06T19:00:00Z" and q.volume == 2500.0
    q2 = live.kalshi_open_question(KALSHI_MARKET, SERIES, anchor_mode="close_time")
    assert q2 is not None and q2.close_at == "2026-11-06T19:00:00Z"
    assert live.kalshi_open_question({**KALSHI_MARKET, "ticker": "KXMVEX-1"}, SERIES) is None
    assert live.kalshi_open_question({**KALSHI_MARKET, "market_type": "scalar"}, SERIES) is None


def test_kalshi_status_mapping():
    close = "2026-11-04T21:50:00Z"
    m = {**KALSHI_MARKET, "status": "finalized", "result": "no", "close_time": close}
    st = live.kalshi_status(m)
    assert st.state == "resolved" and st.outcome == 0 and st.actual_close == "2026-11-04T21:50:00Z"
    assert live.kalshi_status({**m, "status": "determined", "result": "yes"}).state == "closed"
    assert live.kalshi_status({**m, "result": ""}).state == "void"
    st = live.kalshi_status(KALSHI_MARKET)
    assert st.state == "open" and st.anchor == "2026-11-04T22:00:00Z"


PM_MARKET = {
    "id": "123",
    "question": "Will X win the election?",
    "conditionId": "0xabc",
    "slug": "will-x-win",
    "createdAt": "2026-10-01T00:00:00Z",
    "startDate": "2026-10-01T01:00:00Z",
    "endDate": "2026-11-05T00:00:00Z",
    "closed": False,
    "active": True,
    "volumeNum": 25000.0,
    "outcomes": '["Yes", "No"]',
    "clobTokenIds": '["111", "222"]',
    "description": "Resolves Yes if X wins.",
    "tags": [{"slug": "politics"}, {"slug": "elections"}],
    "events": [{"id": "9", "slug": "x-election", "title": "X election"}],
}


def test_polymarket_open_question_and_status():
    q = live.polymarket_open_question(PM_MARKET)
    assert q is not None and q.id == "polymarket:0xabc" and q.close_at == "2026-11-05T00:00:00Z"
    assert q.extra["yes_token"] == "111" and q.category == "politics" and q.outcome == -1
    assert live.polymarket_open_question({**PM_MARKET, "closed": True}) is None
    assert live.polymarket_open_question({**PM_MARKET, "outcomes": '["A", "B"]'}) is None
    closed = {**PM_MARKET, "closed": True, "closedTime": "2026-11-03 10:00:00+00"}
    st = live.polymarket_status({**closed, "outcomePrices": '["1", "0"]'})
    assert st.state == "resolved" and st.outcome == 1 and st.actual_close == "2026-11-03T10:00:00Z"
    half = {**closed, "outcomePrices": '["0.5", "0.5"]', "umaResolutionStatus": "resolved"}
    st = live.polymarket_status(half)
    assert st.state == "void"
    pending = {**closed, "outcomePrices": '["0.6", "0.4"]', "umaResolutionStatus": "proposed"}
    st = live.polymarket_status(pending)
    assert st.state == "closed"


def _mock_client(handler) -> HttpClient:
    return HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_polymarket_list_open_paginates_with_mock_transport():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(dict(req.url.params))
        if "after_cursor" not in req.url.params:
            return httpx.Response(200, json={"markets": [PM_MARKET], "next_cursor": "c1"})
        other = {**PM_MARKET, "conditionId": "0xdef", "outcomes": '["A", "B"]'}
        return httpx.Response(200, json={"markets": [other], "next_cursor": None})

    with _mock_client(handler) as client:
        qs = live.PolymarketLive().list_open(client, make_cfg(min_volume={"polymarket": 1e4}), T0)
    assert [q.id for q in qs] == ["polymarket:0xabc"]
    assert seen[0]["closed"] == "false" and seen[0]["volume_num_min"] == "10000.0"
    assert seen[0]["end_date_min"] == to_iso(T0)


def test_kalshi_list_open_with_mock_transport():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/series"):
            return httpx.Response(200, json={"series": [SERIES]})
        assert req.url.params["status"] == "open"
        late = "2026-12-30T00:00:00Z"
        far = {
            **KALSHI_MARKET,
            "ticker": "KXUCLGAME-FAR-A",
            "expected_expiration_time": late,
            "close_time": late,
        }
        return httpx.Response(200, json={"markets": [KALSHI_MARKET, far], "cursor": ""})

    now = datetime(2026, 10, 30, tzinfo=UTC)
    with _mock_client(handler) as client:
        qs = live.KalshiLive().list_open(client, make_cfg(), now)
    assert [q.id for q in qs] == [f"kalshi:{KALSHI_MARKET['ticker']}"]


def test_config_from_live_toml():
    from pricedin.universe import load_study

    study = load_study(Path(__file__).parents[1] / "studies" / "live.toml")
    cfg = LiveConfig.from_study(study, max_tracked=5)
    assert cfg.horizons == ["7d", "1d", "1h"] and cfg.tolerance == timedelta(minutes=35)
    assert cfg.max_tracked == 5 and cfg.cap("sports") == 20 and cfg.cap("politics") == 30
    assert cfg.min_volume == {"kalshi": 1000.0, "polymarket": 10000.0}
    assert study.data_dir.name == "live"
