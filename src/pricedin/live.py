"""Prospective collector: track open markets and snapshot, forecast and resolve them in real time.

The retrospective study (v1) has three caveats a prospective run removes: Claude Code injects the
real current date into a prompt that is "as of" a past date, Polymarket has no historical bid/ask,
and the volume filter uses lifetime volume (information from after the snapshot). Here every
snapshot is a live order book taken at the moment it is recorded, the forecast is made at that
moment (so the prompt's current date is the real date), and eligibility uses only what is visible
when a market is first seen.

``pricedin live STUDY --once`` runs one pass (meant for an hourly scheduler):

1. Discover open binary markets on each venue that pass the ``[live]`` filters. The anchor
   (``close_at``) is the scheduled close as listed at that moment: Polymarket ``endDate``; Kalshi
   ``min(close_time, expected_expiration_time)`` by default (``kalshi_anchor``), because an open
   Kalshi market's ``close_time`` is usually a fallback days after the event. A market's
   listing-time fields are frozen the first time it is tracked; later venue changes to the
   close are logged in ``updates.jsonl`` and surface as ``extra["anchor_changes"]``.
2. For each tracked market and horizon h, when ``now`` is within ``tolerance`` of
   ``anchor - h`` and no snapshot exists, take the live book, collect news as of now and (unless
   ``dry_run``) forecast that row. A window that passes without a snapshot is logged as missed
   and never backfilled.
3. Markets past their anchor (or gone from the open listing) are checked for settlement. Clean
   YES/NO results set the outcome; voids are logged and excluded.

State lives in the study's data dir, all append-only except the derived views:

- ``tracked.jsonl``     one frozen ``Question`` per tracked market (``outcome`` -1 while open)
- ``updates.jsonl``     anchor changes, closes, resolutions, voids
- ``horizons.jsonl``    per (question, horizon): taken / missed / error, with target and lag
- ``snapshots.jsonl``   live books (``Snapshot``), ``news.jsonl.gz``, ``forecasts_raw/*.jsonl``
- ``ledger.jsonl``      forecast cost per pass and forecaster; ``passes.jsonl`` pass summaries
- views rewritten each pass (only when changed): ``questions.jsonl`` (resolved questions) and
  ``forecasts/*.jsonl.gz``, which ``pricedin evaluate`` reads as for any study.

A lock file (``.live.lock``) stops two passes from running at once; re-running a pass is
harmless because every step first checks what is already recorded.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pricedin.http import HttpClient, HttpError
from pricedin.schema import (
    HORIZONS,
    Forecast,
    NewsContext,
    Question,
    Snapshot,
    from_dict,
    parse_time,
    to_dict,
    to_iso,
    write_jsonl,
)
from pricedin.universe import Study, stable_hash

OPEN, CLOSED, RESOLVED, VOID, UNRESOLVED = "open", "closed", "resolved", "void", "unresolved"
ACTIVE = (OPEN, CLOSED)
LOCK_NAME = ".live.lock"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --- config -----------------------------------------------------------------------------


@dataclass
class LiveConfig:
    """The ``[live]`` section of a study TOML (falling back to ``[study]``/``[universe]``/
    ``[forecast]`` where a key is shared)."""

    venues: list[str]
    horizons: list[str]
    created_after: datetime
    closed_before: datetime
    tolerance: timedelta = timedelta(minutes=35)
    max_days: float = 8.0
    min_open_days: float = 0.0
    min_volume: dict[str, float] = field(default_factory=dict)
    per_event: int = 1
    per_category: int = 30
    category_caps: dict[str, int] = field(default_factory=dict)
    max_tracked: int = 200
    max_forecasts_per_pass: int = 40
    forecasters: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=lambda: ["base"])
    news_sources: list[str] = field(default_factory=lambda: ["wikipedia"])
    news_k: int = 25
    news_window_days: int = 14
    forecast_grace: timedelta | None = None  # default: tolerance
    kalshi_anchor: str = "min"  # "min" | "expected" | "close_time"
    kalshi_close_slack_days: float = 7.0
    give_up_days: float = 30.0
    workers: int = 4
    seed: str = "pricedin-live"
    interval_minutes: int = 60
    lock_stale_minutes: int = 180

    @property
    def grace(self) -> timedelta:
        return self.forecast_grace if self.forecast_grace is not None else self.tolerance

    def cap(self, category: str) -> int:
        return int(self.category_caps.get(category, self.per_category))

    @classmethod
    def from_study(cls, study: Study, **overrides: Any) -> LiveConfig:
        raw = study.raw.get("live", {})
        f = study.raw.get("forecast", {})
        horizons = list(raw.get("horizons", study.horizons))
        unknown = [h for h in horizons if h not in HORIZONS]
        if unknown:
            raise ValueError(f"unknown horizons {unknown}; known: {list(HORIZONS)}")
        grace = raw.get("forecast_grace_minutes")
        cfg = cls(
            venues=list(raw.get("venues", study.venues)),
            horizons=horizons,
            created_after=parse_time(raw.get("created_after", study.created_after)),
            closed_before=parse_time(raw.get("closed_before", study.closed_before)),
            tolerance=timedelta(minutes=float(raw.get("tolerance_minutes", 35))),
            max_days=float(raw.get("max_days", 8)),
            min_open_days=float(raw.get("min_open_days", study.min_open_days)),
            min_volume={k: float(v) for k, v in raw.get("min_volume", study.min_volume).items()},
            per_event=int(raw.get("per_event", study.per_event)),
            per_category=int(raw.get("per_category", study.per_cell)),
            category_caps={k: int(v) for k, v in raw.get("category_caps", {}).items()},
            max_tracked=int(raw.get("max_tracked", 200)),
            max_forecasts_per_pass=int(raw.get("max_forecasts_per_pass", 40)),
            forecasters=list(raw.get("forecasters", f.get("forecasters", []))),
            conditions=list(raw.get("conditions", f.get("conditions", ["base"]))),
            news_sources=list(raw.get("news_sources", ["wikipedia"])),
            news_k=int(raw.get("news_k", 25)),
            news_window_days=int(raw.get("news_window_days", 14)),
            forecast_grace=timedelta(minutes=float(grace)) if grace is not None else None,
            kalshi_anchor=str(raw.get("kalshi_anchor", "min")),
            kalshi_close_slack_days=float(raw.get("kalshi_close_slack_days", 7)),
            give_up_days=float(raw.get("give_up_days", 30)),
            workers=int(raw.get("workers", 4)),
            seed=str(raw.get("seed", study.seed)),
            interval_minutes=int(raw.get("interval_minutes", 60)),
            lock_stale_minutes=int(raw.get("lock_stale_minutes", 180)),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        if cfg.kalshi_anchor not in ("min", "expected", "close_time"):
            raise ValueError(f"kalshi_anchor must be min, expected or close_time: {cfg!r}")
        return cfg


# --- venue adapters ---------------------------------------------------------------------


@dataclass
class MarketStatus:
    state: str  # open | closed | resolved | void | unknown
    outcome: int | None = None
    actual_close: str = ""
    anchor: str = ""  # the venue's current scheduled close, if still open
    settled_at: str = ""
    detail: str = ""
    volume: float | None = None


class Adapter(Protocol):
    """What a pass needs from a venue (real ones below; tests use fakes)."""

    name: str

    def list_open(self, client: HttpClient, cfg: LiveConfig, now: datetime) -> list[Question]: ...

    def snapshot(self, client: HttpClient, q: Question) -> Snapshot | None: ...

    def status(self, client: HttpClient, q: Question, cfg: LiveConfig) -> MarketStatus: ...


def _t(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        t = parse_time(value)
    except (TypeError, ValueError):
        return None
    return None if t.year < 2000 else t


def _num(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def kalshi_anchor(market: Mapping[str, Any], mode: str = "min") -> tuple[datetime | None, str]:
    """(anchor, source) for an open Kalshi market: its scheduled close as listed now."""
    close, expected = _t(market.get("close_time")), _t(market.get("expected_expiration_time"))
    if mode == "close_time" or expected is None:
        return close, "close_time"
    if mode == "expected" or close is None:
        return expected, "expected_expiration_time"
    return (expected, "expected_expiration_time") if expected <= close else (close, "close_time")


def kalshi_open_question(
    market: Mapping[str, Any],
    series: Mapping[str, Any] | None = None,
    *,
    anchor_mode: str = "min",
    seen_at: datetime | None = None,
) -> Question | None:
    """An open Kalshi binary market -> Question (outcome -1) anchored on its listed close."""
    from pricedin.venues import kalshi

    s = series or {}
    ticker = str(market.get("ticker") or "")
    if (
        market.get("market_type") != "binary"
        or market.get("status") not in ("active", "open")
        or not ticker
        or ticker.startswith("KXMVE")
        or market.get("mve_collection_ticker")
        or s.get("category") in kalshi.SKIP_CATEGORIES
    ):
        return None
    anchor, source = kalshi_anchor(market, anchor_mode)
    created = _t(market.get("created_time"))
    opened = _t(market.get("open_time")) or created
    if anchor is None or created is None or opened is None or anchor <= opened:
        return None
    series_ticker = s.get("ticker") or ticker.split("-")[0]
    venue_category = s.get("category") or ""
    volume = _num(market.get("volume_fp")) or 0.0
    extra: dict[str, Any] = {
        "series": series_ticker,
        "anchor_source": source,
        "listed_close_time": to_iso(t) if (t := _t(market.get("close_time"))) else "",
        "scheduled_end": to_iso(t) if (t := _t(market.get("expected_expiration_time"))) else "",
        "latest_expiration": to_iso(t) if (t := _t(market.get("latest_expiration_time"))) else "",
        "can_close_early": bool(market.get("can_close_early")),
        "fee": kalshi.fee_terms(series),
        "volume_at_listing": volume,
        "open_interest": _num(market.get("open_interest_fp")),
        "frequency": s.get("frequency", ""),
        "tags": list(s.get("tags") or []),
    }
    if seen_at is not None:
        extra["first_seen"] = to_iso(seen_at)
    return Question(
        id=f"kalshi:{ticker}",
        venue="kalshi",
        event_id=f"kalshi:{market.get('event_ticker') or ticker}",
        title=kalshi.question_title(market),
        rules=kalshi.question_rules(market),
        category=kalshi.map_category(venue_category, s.get("tags") or ()),
        created_at=to_iso(created),
        open_at=to_iso(opened),
        close_at=to_iso(anchor),
        scheduled_close_at=to_iso(anchor),
        outcome=-1,
        volume=volume,
        url=f"https://kalshi.com/markets/{series_ticker.lower()}",
        venue_category=venue_category,
        extra=extra,
    )


class KalshiLive:
    name = "kalshi"

    def list_open(self, client: HttpClient, cfg: LiveConfig, now: datetime) -> list[Question]:
        from pricedin.venues import kalshi

        kalshi.configure(client)
        series = kalshi.fetch_series(client)
        until = now + timedelta(days=cfg.max_days)
        params: dict[str, Any] = {
            "status": "open",
            "min_close_ts": int(now.timestamp()),
            # close_time often trails the event by days; anchors come from expected expiration
            "max_close_ts": int((until + timedelta(days=cfg.kalshi_close_slack_days)).timestamp()),
            "limit": 1000,
            "mve_filter": "exclude",
        }
        out: list[Question] = []
        cursor = ""
        while True:
            body = client.get_json(
                f"{kalshi.BASE}/markets", {**params, "cursor": cursor} if cursor else params,
                use_cache=False,
            )  # fmt: skip
            for m in body.get("markets") or []:
                s = series.get(str(m.get("event_ticker") or m.get("ticker", "")).split("-")[0])
                q = kalshi_open_question(m, s, anchor_mode=cfg.kalshi_anchor, seen_at=now)
                if q is not None and q.close <= until:
                    out.append(q)
            cursor = body.get("cursor") or ""
            if not cursor or not body.get("markets"):
                return out

    def snapshot(self, client: HttpClient, q: Question) -> Snapshot | None:
        from pricedin.venues import kalshi

        return kalshi.live_snapshot(client, q)

    def status(self, client: HttpClient, q: Question, cfg: LiveConfig) -> MarketStatus:
        from pricedin.venues import kalshi

        ticker = q.id.removeprefix("kalshi:")
        try:
            m = client.get_json(f"{kalshi.BASE}/markets/{ticker}", use_cache=False)["market"]
        except HttpError as exc:
            if exc.status != 404:
                raise
            m = client.get_json(f"{kalshi.BASE}/historical/markets/{ticker}", use_cache=False)[
                "market"
            ]
        return kalshi_status(m, cfg.kalshi_anchor)


def kalshi_status(m: Mapping[str, Any], anchor_mode: str = "min") -> MarketStatus:
    """Kalshi market object -> MarketStatus. Only ``finalized``/``settled`` results count."""
    status, result = m.get("status"), m.get("result")
    close = _t(m.get("close_time"))
    actual = to_iso(close) if close else ""
    settled = to_iso(t) if (t := _t(m.get("settlement_ts"))) else ""
    volume = _num(m.get("volume_fp", m.get("volume")))
    if status in ("finalized", "settled"):
        if result in ("yes", "no"):
            return MarketStatus(
                RESOLVED, 1 if result == "yes" else 0, actual, "", settled, status, volume
            )
        return MarketStatus(VOID, None, actual, "", settled, f"{status}/{result!r}", volume)
    if status in ("active", "open", "inactive", "initialized", "unopened"):
        anchor, _ = kalshi_anchor(m, anchor_mode)
        return MarketStatus(OPEN, anchor=to_iso(anchor) if anchor else "", detail=str(status))
    return MarketStatus(CLOSED, actual_close=actual, detail=f"{status}/{result!r}", volume=volume)


def polymarket_open_question(
    market: Mapping[str, Any], *, seen_at: datetime | None = None
) -> Question | None:
    """An open Polymarket Yes/No market (Gamma object) -> Question (outcome -1) anchored on
    ``endDate``. Reuses the venue module's token, tag, fee and rules-leak helpers."""
    from pricedin.venues import polymarket as pm

    m = dict(market)
    if m.get("closed") or not m.get("active", True):
        return None
    tokens = pm.yes_no_tokens(m)
    if tokens is None:
        return None
    slug = str(m.get("slug") or "")
    if "parlay" in slug or "up-or-down" in slug:
        return None
    created = pm.parse_pm_time(m.get("createdAt"))
    opened = pm.parse_pm_time(m.get("startDate")) or created
    end = pm.parse_pm_time(m.get("endDate"))
    cid = str(m.get("conditionId") or "")
    if created is None or opened is None or end is None or not cid or end <= max(opened, created):
        return None
    rules, flags, drop = pm.check_rules(str(m.get("description") or ""))
    if drop:
        return None
    ev = (m.get("events") or [{}])[0] or {}
    tags = pm.tag_slugs(m)
    category, cat_tag = pm.map_category(tags)
    volume = float(m.get("volumeNum") or 0.0)
    extra: dict[str, Any] = {
        "market_id": str(m.get("id") or ""),
        "condition_id": cid,
        "slug": slug,
        "event_slug": ev.get("slug"),
        "event_title": ev.get("title"),
        "series": ev.get("seriesSlug"),
        "group_item": m.get("groupItemTitle"),
        "anchor_source": "endDate",
        "scheduled_end": to_iso(end),
        "yes_token": tokens[0],
        "no_token": tokens[1],
        "neg_risk": bool(m.get("negRisk")),
        "neg_risk_id": m.get("negRiskMarketID"),
        "tick_size": _num(m.get("orderPriceMinTickSize")),
        "fee": pm.fee_terms(m),
        "tags": tags,
        "volume_at_listing": volume,
    }
    if flags:
        extra["rules_flag"] = flags
    if seen_at is not None:
        extra["first_seen"] = to_iso(seen_at)
    return Question(
        id=f"polymarket:{cid}",
        venue="polymarket",
        event_id=f"polymarket:{ev.get('id') or cid}",
        title=pm.self_contained_title(str(m.get("question") or ""), str(ev.get("title") or "")),
        rules=rules,
        category=category,
        created_at=to_iso(created),
        open_at=to_iso(opened),
        close_at=to_iso(end),
        scheduled_close_at=to_iso(end),
        outcome=-1,
        volume=volume,
        url=f"https://polymarket.com/event/{ev.get('slug')}" if ev.get("slug") else "",
        venue_category=cat_tag,
        extra=extra,
    )


def polymarket_status(m: Mapping[str, Any]) -> MarketStatus:
    """Gamma market object -> MarketStatus. 50/50 or otherwise unclean resolutions are void."""
    from pricedin.venues import polymarket as pm

    if not m.get("closed"):
        end = pm.parse_pm_time(m.get("endDate"))
        return MarketStatus(OPEN, anchor=to_iso(end) if end else "", detail="open")
    closed = pm.parse_pm_time(m.get("closedTime"))
    actual = to_iso(closed) if closed else ""
    volume = _num(m.get("volumeNum"))
    uma = m.get("umaResolutionStatus")
    outcome = pm.resolved_outcome(dict(m))
    if outcome is not None and uma in (None, "resolved"):
        return MarketStatus(RESOLVED, outcome, actual, "", actual, f"uma={uma}", volume)
    if uma == "resolved":  # settled, but not cleanly YES or NO (e.g. 50/50)
        return MarketStatus(VOID, None, actual, "", actual, f"prices={m.get('outcomePrices')}")
    return MarketStatus(CLOSED, actual_close=actual, detail=f"uma={uma}", volume=volume)


class PolymarketLive:
    name = "polymarket"

    def list_open(self, client: HttpClient, cfg: LiveConfig, now: datetime) -> list[Question]:
        from pricedin.venues import polymarket as pm

        _tune_polymarket(client)
        params: dict[str, Any] = {
            "closed": "false",
            "limit": pm.PAGE_LIMIT,
            "end_date_min": to_iso(now),
            "end_date_max": to_iso(now + timedelta(days=cfg.max_days)),
            "start_date_min": to_iso(cfg.created_after),  # createdAt <= startDate
            "include_tag": "true",
        }
        if cfg.min_volume.get("polymarket", 0) > 0:
            params["volume_num_min"] = cfg.min_volume["polymarket"]
        out: list[Question] = []
        seen: set[str] = set()
        cursor = None
        while True:
            p = dict(params, after_cursor=cursor) if cursor else params
            page = client.get_json(f"{pm.GAMMA}/markets/keyset", p, use_cache=False)
            markets = page.get("markets") or []
            for m in markets:
                q = polymarket_open_question(m, seen_at=now)
                if q is not None and q.id not in seen:
                    seen.add(q.id)
                    out.append(q)
            cursor = page.get("next_cursor")
            if not cursor or not markets:
                return out

    def snapshot(self, client: HttpClient, q: Question) -> Snapshot | None:
        from pricedin.venues import polymarket as pm

        snap = pm.live_snapshot(client, q)
        return snap if _finite_prob(snap.mid) else None

    def status(self, client: HttpClient, q: Question, cfg: LiveConfig) -> MarketStatus:
        from pricedin.venues import polymarket as pm

        _tune_polymarket(client)
        cid = q.extra.get("condition_id") or q.id.removeprefix("polymarket:")
        # Gamma hides closed markets unless asked for them explicitly.
        for closed in ("false", "true"):
            rows = client.get_json(
                f"{pm.GAMMA}/markets", {"condition_ids": cid, "closed": closed}, use_cache=False
            )
            if rows:
                return polymarket_status(rows[0])
        return MarketStatus("unknown", detail="not found on gamma")


def _tune_polymarket(client: HttpClient) -> None:
    from pricedin.venues import polymarket as pm

    for host, interval in pm.HOST_INTERVALS.items():
        client.host_intervals.setdefault(host, interval)


def default_adapters() -> dict[str, Adapter]:
    return {"kalshi": KalshiLive(), "polymarket": PolymarketLive()}


def _finite_prob(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= x <= 1.0


# --- state ------------------------------------------------------------------------------


class LockBusy(RuntimeError):
    """Another pass holds the lock."""


class PassLock:
    """Exclusive lock file; one older than ``stale`` is assumed to be left by a dead pass."""

    def __init__(self, path: Path, stale: timedelta):
        self.path = path
        self.stale = stale

    def __enter__(self) -> PassLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if attempt == 0 and age > self.stale.total_seconds():
                    _log(f"removing stale lock {self.path} ({age / 60:.0f} min old)")
                    self.path.unlink(missing_ok=True)
                    continue
                raise LockBusy(f"{self.path} exists: another pass is running") from None
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"pid": os.getpid(), "started": to_iso(datetime.now(UTC))}))
            return self
        raise LockBusy(f"could not take {self.path}")

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


def _repair_tail(path: Path) -> None:
    """Drop a partial last line left by a crash mid-append, so readers never choke on it."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    _log(f"repairing {path.name}: dropping a partial last line ({len(data) - cut} bytes)")
    with open(path, "r+b") as fh:
        fh.truncate(cut)


def _rows(path: Path) -> list[dict[str, Any]]:
    """JSONL rows, skipping lines that do not parse (logged)."""
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    out = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                _log(f"{path.name}:{i}: skipping a line that is not JSON")
    return out


def _write_if_changed(path: Path, text: str, *, gz: bool = False) -> bool:
    """Atomically write ``text`` (deterministic gzip if ``gz``) unless the file already holds it."""
    data = text.encode("utf-8")
    if gz:
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as g:
            g.write(data)
        data = buf.getvalue()
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


@dataclass
class Tracked:
    q: Question  # frozen at first sight
    state: str = OPEN
    outcome: int | None = None
    actual_close: str = ""
    settled_at: str = ""
    final_volume: float | None = None
    anchor_now: str = ""  # the venue's latest scheduled close
    anchor_changes: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""

    @property
    def anchor(self) -> datetime:
        return self.q.close


class State:
    """Files under one data dir, loaded once per pass and appended to as the pass goes."""

    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.tracked_path = data_dir / "tracked.jsonl"
        self.updates_path = data_dir / "updates.jsonl"
        self.horizons_path = data_dir / "horizons.jsonl"
        self.snapshots_path = data_dir / "snapshots.jsonl"
        self.news_path = data_dir / "news.jsonl.gz"
        self.raw_dir = data_dir / "forecasts_raw"
        self.view_dir = data_dir / "forecasts"
        self.questions_path = data_dir / "questions.jsonl"
        self.ledger_path = data_dir / "ledger.jsonl"
        self.passes_path = data_dir / "passes.jsonl"

    def load(self) -> None:
        for p in (
            self.tracked_path,
            self.updates_path,
            self.horizons_path,
            self.snapshots_path,
            self.ledger_path,
            self.passes_path,
            *self.raw_dir.glob("*.jsonl"),
        ):
            _repair_tail(p)
        self.tracked: dict[str, Tracked] = {}
        for r in _rows(self.tracked_path):
            q = from_dict(Question, r)
            self.tracked.setdefault(q.id, Tracked(q, anchor_now=q.close_at))
        for u in _rows(self.updates_path):
            t = self.tracked.get(u.get("question_id", ""))
            if t is not None:
                _apply_update(t, u)
        self.horizon_log: dict[tuple[str, str], dict[str, Any]] = {}
        self.forecast_missed: set[tuple[str, str, str, str]] = set()
        self.last_error: dict[tuple[str, str], str] = {}
        for r in _rows(self.horizons_path):
            key = (r["question_id"], r["horizon"])
            if r["status"] in ("taken", "missed"):
                self.horizon_log.setdefault(key, r)
            elif r["status"] == "error":
                self.last_error[key] = r.get("reason", "")
            elif r["status"] == "forecast_missed":
                self.forecast_missed.add((*key, r["forecaster"], r["condition"]))
        self.snapshots: dict[tuple[str, str], Snapshot] = {}
        for r in _rows(self.snapshots_path):
            s = from_dict(Snapshot, r)
            self.snapshots.setdefault((s.question_id, s.horizon), s)
        self.news_keys = {(r["question_id"], r["horizon"]) for r in _rows(self.news_path)}

    def append(self, path: Path, rows: Iterable[Any]) -> None:
        write_jsonl(path, rows, append=True)

    def update(self, t: Tracked, kind: str, now: datetime, **fields: Any) -> None:
        u = {"time": to_iso(now), "question_id": t.q.id, "kind": kind, **fields}
        self.append(self.updates_path, [u])
        _apply_update(t, u)

    def log_horizon(self, row: dict[str, Any]) -> None:
        self.append(self.horizons_path, [row])
        key = (row["question_id"], row["horizon"])
        if row["status"] in ("taken", "missed"):
            self.horizon_log.setdefault(key, row)
        elif row["status"] == "error":
            self.last_error[key] = row.get("reason", "")
        elif row["status"] == "forecast_missed":
            self.forecast_missed.add((*key, row["forecaster"], row["condition"]))


def _apply_update(t: Tracked, u: Mapping[str, Any]) -> None:
    kind = u.get("kind")
    if kind == "anchor_change":
        t.anchor_now = u["new"]
        t.anchor_changes.append({"time": u["time"], "old": u["old"], "new": u["new"]})
    elif kind == CLOSED and t.state == OPEN:
        t.state = CLOSED
        t.actual_close = u.get("actual_close") or t.actual_close
        t.detail = u.get("detail", "")
    elif kind == RESOLVED:
        t.state, t.outcome = RESOLVED, int(u["outcome"])
        t.actual_close = u.get("actual_close") or t.actual_close
        t.settled_at = u.get("settled_at", "")
        t.final_volume = u.get("volume")
    elif kind in (VOID, UNRESOLVED):
        t.state = kind
        t.actual_close = u.get("actual_close") or t.actual_close
        t.detail = u.get("detail", "")


# --- the pass ---------------------------------------------------------------------------


NewsFn = Callable[[list[Question], list[Snapshot], Path], dict[str, Any]]
ProviderFactory = Callable[[str], Any]


def eligible_open(q: Question, cfg: LiveConfig, now: datetime) -> str:
    """ "" if an open market may be tracked now, else the reason it may not."""
    if parse_time(q.created_at) < cfg.created_after:
        return "created_before"
    if q.close > cfg.closed_before:
        return "closes_after_study"
    if q.close <= now or q.close > now + timedelta(days=cfg.max_days):
        return "outside_max_days"
    if parse_time(q.open_at) > q.close - timedelta(days=cfg.min_open_days):
        return "too_short"
    if q.volume < cfg.min_volume.get(q.venue, 0.0):
        return "low_volume"
    # at least one horizon window must still be ahead
    if all(q.close - HORIZONS[h] + cfg.tolerance < now for h in cfg.horizons):
        return "no_horizon_left"
    return ""


def _admit(
    state: State, candidates: list[Question], cfg: LiveConfig, now: datetime
) -> tuple[list[Question], Counter[str]]:
    reasons: Counter[str] = Counter()
    active = [t for t in state.tracked.values() if t.state in ACTIVE]
    n_active = len(active)
    per_event = Counter(t.q.event_id for t in state.tracked.values())  # lifetime
    per_cell = Counter((t.q.venue, t.q.category) for t in active)
    admitted: list[Question] = []
    for q in sorted(candidates, key=lambda q: stable_hash(cfg.seed, q.id)):
        if q.id in state.tracked:
            continue
        why = eligible_open(q, cfg, now)
        if not why and per_event[q.event_id] >= cfg.per_event:
            why = "event_cap"
        if not why and per_cell[(q.venue, q.category)] >= cfg.cap(q.category):
            why = "category_cap"
        if not why and n_active >= cfg.max_tracked:
            why = "max_tracked"
        reasons[why or "admitted"] += 1
        if why:
            continue
        per_event[q.event_id] += 1
        per_cell[(q.venue, q.category)] += 1
        n_active += 1
        admitted.append(q)
    return admitted, reasons


def _missed_reason(t: Tracked, h: str, state: State, cfg: LiveConfig) -> str:
    target = t.anchor - HORIZONS[h]
    if t.actual_close and parse_time(t.actual_close) <= target:
        return "closed_before_target"
    first = t.q.extra.get("first_seen")
    if first and parse_time(first) > target + cfg.tolerance:
        return "tracked_after_window"
    err = state.last_error.get((t.q.id, h))
    return f"error: {err}" if err else "no_pass_in_window"


def run_pass(
    cfg: LiveConfig,
    data_dir: str | Path,
    *,
    dry_run: bool = False,
    clock: Callable[[], datetime] | None = None,
    client: HttpClient | None = None,
    adapters: Mapping[str, Adapter] | None = None,
    news_fn: NewsFn | None = None,
    provider_factory: ProviderFactory | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """One collector pass over ``data_dir``. Returns a summary (also appended to passes.jsonl).

    ``clock`` (default: real UTC time), ``client``, ``adapters``, ``news_fn`` and
    ``provider_factory`` exist so tests can run without network. ``cache_dir`` holds the news
    and LLM response caches (default ``.pricedin-cache``). Raises ``LockBusy`` if another pass is
    running.
    """
    from pricedin.cli import CACHE_DIR

    clock = clock or (lambda: datetime.now(UTC))
    data = Path(data_dir)
    cache = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    adapters = adapters if adapters is not None else default_adapters()
    with PassLock(data / LOCK_NAME, timedelta(minutes=cfg.lock_stale_minutes)):
        own = client is None
        http = client or HttpClient(cache_dir=None)
        try:
            return _run_pass(
                cfg, State(data), dry_run, clock, http, adapters, news_fn, provider_factory, cache
            )
        finally:
            if own:
                http.close()


def _run_pass(
    cfg: LiveConfig,
    state: State,
    dry_run: bool,
    clock: Callable[[], datetime],
    http: HttpClient,
    adapters: Mapping[str, Adapter],
    news_fn: NewsFn | None,
    provider_factory: ProviderFactory | None,
    cache: Path,
) -> dict[str, Any]:
    started = clock()
    state.load()
    summary: dict[str, Any] = {
        "started": to_iso(started),
        "dry_run": dry_run,
        "listed": {},
        "admitted": 0,
        "admission": {},
        "anchor_changes": 0,
        "due": [],
        "taken": 0,
        "missed": 0,
        "snapshot_errors": 0,
        "news": None,
        "forecasts": [],
        "forecast_tasks": 0,
        "forecast_missed": 0,
        "cost_usd": 0.0,
        "status_checks": 0,
        "resolved": 0,
        "void": 0,
        "errors": [],
    }

    # 1. discover
    listed: dict[str, Question] = {}
    listing_ok: dict[str, bool] = {}
    for v in cfg.venues:
        try:
            qs = adapters[v].list_open(http, cfg, clock())
            listing_ok[v] = True
        except Exception as exc:  # a venue outage must not stop the other steps
            listing_ok[v] = False
            summary["errors"].append(f"list {v}: {type(exc).__name__}: {exc}")
            qs = []
        summary["listed"][v] = len(qs)
        for q in qs:
            listed.setdefault(q.id, q)
    now = clock()
    for qid, q in listed.items():
        t = state.tracked.get(qid)
        if t is not None and t.state in ACTIVE and q.close_at != t.anchor_now:
            state.update(t, "anchor_change", now, old=t.anchor_now, new=q.close_at)
            summary["anchor_changes"] += 1
    admitted, reasons = _admit(state, list(listed.values()), cfg, now)
    if admitted:
        state.append(state.tracked_path, admitted)
        for q in admitted:
            state.tracked[q.id] = Tracked(q, anchor_now=q.close_at)
    summary["admitted"] = len(admitted)
    summary["admission"] = dict(reasons)

    # 2. horizons: due -> snapshot; window passed -> missed (never backfilled)
    for t in sorted(state.tracked.values(), key=lambda t: t.q.id):
        for h in cfg.horizons:
            key = (t.q.id, h)
            if key in state.horizon_log or key in state.snapshots:
                continue
            now = clock()
            target = t.anchor - HORIZONS[h]
            lo, hi = target - cfg.tolerance, target + cfg.tolerance
            base = {"question_id": t.q.id, "horizon": h, "target": to_iso(target)}
            if now > hi or (t.state not in ACTIVE and now >= lo):
                reason = _missed_reason(t, h, state, cfg)
                if t.state != OPEN and reason == "no_pass_in_window":
                    reason = f"market_{t.state}"
                state.log_horizon(
                    {**base, "status": "missed", "time": to_iso(now), "reason": reason}
                )
                summary["missed"] += 1
                continue
            if now < lo or t.state != OPEN:
                continue
            summary["due"].append(f"{t.q.id}|{h}")
            try:
                snap = adapters[t.q.venue].snapshot(http, t.q)
                err = "" if snap is not None and _finite_prob(snap.mid) else "no usable book"
            except Exception as exc:  # retried next pass while the window is open
                snap, err = None, f"{type(exc).__name__}: {exc}"
            if err or snap is None:
                state.log_horizon({**base, "status": "error", "time": to_iso(now), "reason": err})
                summary["snapshot_errors"] += 1
                continue
            snap = replace(snap, question_id=t.q.id, horizon=h)
            state.append(state.snapshots_path, [snap])
            state.snapshots[key] = snap
            lag = (parse_time(snap.as_of) - target).total_seconds() / 60
            state.log_horizon(
                {**base, "status": "taken", "time": to_iso(now), "as_of": snap.as_of,
                 "lag_min": round(lag, 1), "mid": snap.mid}
            )  # fmt: skip
            summary["taken"] += 1

    # recent snapshots (this pass, or an earlier one that crashed) still worth acting on
    now = clock()
    recent = [
        s
        for s in state.snapshots.values()
        if now - parse_time(s.as_of) <= cfg.grace and s.question_id in state.tracked
    ]
    qmap = {qid: t.q for qid, t in state.tracked.items()}

    # 3. news as of each snapshot (provably public before it)
    need_news = [s for s in recent if (s.question_id, s.horizon) not in state.news_keys]
    if need_news:
        try:
            fn = news_fn or _default_news(cfg, cache)
            summary["news"] = fn(
                [qmap[s.question_id] for s in need_news], need_news, state.news_path
            )
            state.news_keys |= {(s.question_id, s.horizon) for s in need_news}
        except Exception as exc:
            summary["errors"].append(f"news: {type(exc).__name__}: {exc}")

    # 4. forecasts (budgeted), and forecasts that can no longer be made in time
    _forecast(cfg, state, recent, qmap, now, dry_run, provider_factory, cache, summary)

    # 5. resolution
    _resolve(cfg, state, http, adapters, listed, listing_ok, clock, summary)

    # 6. views for `pricedin evaluate`
    write_views(state)
    summary["tracked_active"] = sum(t.state in ACTIVE for t in state.tracked.values())
    summary["tracked_total"] = len(state.tracked)
    summary["finished"] = to_iso(clock())
    summary["cost_usd"] = round(summary["cost_usd"], 6)
    state.append(state.passes_path, [{k: v for k, v in summary.items() if k != "due"}])
    return summary


def _default_news(cfg: LiveConfig, cache: Path) -> NewsFn:
    def fn(questions: list[Question], snaps: list[Snapshot], out: Path) -> dict[str, Any]:
        from pricedin.news import collect_news

        return collect_news(
            questions,
            snaps,
            sources=tuple(cfg.news_sources),
            out_path=out,
            cache_dir=cache / "news",
            k=cfg.news_k,
            window_days=cfg.news_window_days,
        )

    return fn


def forecast_file(state: State, spec: str, condition: str) -> Path:
    from pricedin.cli import forecast_slug

    return state.raw_dir / f"{forecast_slug(spec)}__{condition}.jsonl"


def _forecast(
    cfg: LiveConfig,
    state: State,
    recent: list[Snapshot],
    qmap: dict[str, Question],
    now: datetime,
    dry_run: bool,
    provider_factory: ProviderFactory | None,
    cache: Path,
    summary: dict[str, Any],
) -> None:
    from pricedin.forecast.runner import load_forecasts, run_forecasts

    recent_keys = {(s.question_id, s.horizon) for s in recent}
    budget = cfg.max_forecasts_per_pass
    news = None
    for spec in cfg.forecasters:
        for cond in cfg.conditions:
            path = forecast_file(state, spec, cond)
            done = {k for k, f in load_forecasts(path).items() if f.prob is not None}
            # snapshots whose forecast window has passed: logged once, never backfilled
            for (qid, h), _s in sorted(state.snapshots.items()):
                if (qid, h) in done or (qid, h) in recent_keys:
                    continue
                if (qid, h, spec, cond) in state.forecast_missed:
                    continue
                state.log_horizon(
                    {"question_id": qid, "horizon": h, "status": "forecast_missed",
                     "forecaster": spec, "condition": cond, "time": to_iso(now)}
                )  # fmt: skip
                summary["forecast_missed"] += 1
            todo = sorted(
                (s for s in recent if (s.question_id, s.horizon) not in done),
                key=lambda s: (s.as_of, s.question_id, s.horizon),
            )
            if dry_run or not todo:
                continue
            if budget <= 0:
                summary["errors"].append(f"forecast budget spent; {spec}/{cond} deferred")
                continue
            if cond == "news" and news is None:
                news = [from_dict(NewsContext, r) for r in _rows(state.news_path)]
            provider = provider_factory(spec) if provider_factory else None
            try:
                res = run_forecasts(
                    [qmap[s.question_id] for s in todo],
                    todo,
                    news if cond == "news" else None,
                    provider_spec=spec,
                    condition=cond,
                    out_path=path,
                    workers=cfg.workers,
                    limit=budget,
                    cache_dir=cache / "llm",
                    provider=provider,
                )
            except Exception as exc:
                summary["errors"].append(f"forecast {spec}/{cond}: {type(exc).__name__}: {exc}")
                continue
            budget -= res["tasks"]
            summary["forecast_tasks"] += res["tasks"]
            summary["cost_usd"] += res.get("cost_usd") or 0.0
            row = {
                "time": to_iso(now),
                "forecaster": spec,
                "condition": cond,
                **{k: res.get(k) for k in ("tasks", "done", "failed", "cached", "cost_usd")},
                "usage": res.get("usage", {}),
                "aborted": res.get("aborted"),
            }
            state.append(state.ledger_path, [row])
            summary["forecasts"].append(row)


def _resolve(
    cfg: LiveConfig,
    state: State,
    http: HttpClient,
    adapters: Mapping[str, Adapter],
    listed: Mapping[str, Question],
    listing_ok: Mapping[str, bool],
    clock: Callable[[], datetime],
    summary: dict[str, Any],
) -> None:
    for t in sorted(state.tracked.values(), key=lambda t: t.q.id):
        if t.state not in ACTIVE:
            continue
        now = clock()
        vanished = listing_ok.get(t.q.venue, False) and t.q.id not in listed
        if not (now >= t.anchor or t.state == CLOSED or vanished):
            continue
        try:
            st = adapters[t.q.venue].status(http, t.q, cfg)
        except Exception as exc:
            summary["errors"].append(f"status {t.q.id}: {type(exc).__name__}: {exc}")
            continue
        summary["status_checks"] += 1
        if st.state == RESOLVED and st.outcome in (0, 1):
            state.update(
                t, RESOLVED, now, outcome=st.outcome, actual_close=st.actual_close,
                settled_at=st.settled_at, volume=st.volume, detail=st.detail,
            )  # fmt: skip
            summary["resolved"] += 1
        elif st.state == VOID:
            state.update(t, VOID, now, actual_close=st.actual_close, detail=st.detail)
            summary["void"] += 1
        elif st.state == CLOSED and t.state == OPEN:
            state.update(t, CLOSED, now, actual_close=st.actual_close, detail=st.detail)
        elif st.state == OPEN and st.anchor and st.anchor != t.anchor_now:
            state.update(t, "anchor_change", now, old=t.anchor_now, new=st.anchor)
            summary["anchor_changes"] += 1
        if t.state in ACTIVE and now > t.anchor + timedelta(days=cfg.give_up_days):
            state.update(t, UNRESOLVED, now, detail=f"unsettled {cfg.give_up_days} d after anchor")


def resolved_question(t: Tracked) -> Question:
    """The frozen question with its outcome and post-close facts in ``extra``."""
    q = t.q
    extra = dict(q.extra)
    extra["live"] = True
    extra["actual_close"] = t.actual_close or q.close_at
    extra["settled_at"] = t.settled_at
    if t.final_volume is not None:
        extra["final_volume"] = t.final_volume
    gap_h = (q.close - parse_time(extra["actual_close"])).total_seconds() / 3600
    extra["early_close"] = gap_h > 1.0
    extra["close_gap_h"] = round(gap_h, 3)
    if t.anchor_changes:
        extra["anchor_changes"] = t.anchor_changes
    return replace(q, outcome=int(t.outcome or 0), extra=extra)


def write_views(state: State) -> dict[str, bool]:
    """Rewrite ``questions.jsonl`` (resolved only) and ``forecasts/*.jsonl.gz`` (best row per
    question and horizon) when their content changed. Returns which files were written."""
    from pricedin.forecast.runner import load_forecasts

    changed: dict[str, bool] = {}
    resolved = sorted(
        (resolved_question(t) for t in state.tracked.values() if t.state == RESOLVED),
        key=lambda q: q.id,
    )
    text = "".join(
        json.dumps(to_dict(q), ensure_ascii=False, sort_keys=True) + "\n" for q in resolved
    )
    changed["questions.jsonl"] = _write_if_changed(state.questions_path, text)
    for raw in sorted(state.raw_dir.glob("*.jsonl")):
        rows: list[Forecast] = sorted(
            load_forecasts(raw).values(), key=lambda f: (f.question_id, f.horizon)
        )
        text = "".join(
            json.dumps(to_dict(f), ensure_ascii=False, sort_keys=True) + "\n" for f in rows
        )
        name = raw.name + ".gz"
        changed[name] = _write_if_changed(state.view_dir / name, text, gz=True)
    return changed


def status_report(data_dir: str | Path) -> dict[str, Any]:
    """Counts by state and horizon outcome, for a quick look at a live study."""
    state = State(Path(data_dir))
    state.load()
    by_state = Counter(t.state for t in state.tracked.values())
    by_h: dict[str, Counter[str]] = defaultdict(Counter)
    for (_, h), r in state.horizon_log.items():
        by_h[h][r["status"]] += 1
    return {
        "tracked": dict(by_state),
        "horizons": {h: dict(c) for h, c in by_h.items()},
        "snapshots": len(state.snapshots),
    }


# --- entry point ------------------------------------------------------------------------


def main(args: Any) -> int:
    """``pricedin live``: one pass with ``--once``, else a pass every ``interval_minutes``.

    Optional args (used when the CLI defines them): ``data_dir``, ``forecaster`` (list, replaces
    the configured forecasters), ``max_forecasts``, ``max_tracked``.
    """
    from pricedin.universe import load_study

    study = load_study(args.study)
    cfg = LiveConfig.from_study(
        study,
        forecasters=getattr(args, "forecaster", None) or None,
        max_forecasts_per_pass=getattr(args, "max_forecasts", None),
        max_tracked=getattr(args, "max_tracked", None),
    )
    data_dir = Path(getattr(args, "data_dir", None) or study.data_dir)
    while True:
        t0 = time.monotonic()
        try:
            summary = run_pass(cfg, data_dir, dry_run=bool(args.dry_run))
        except LockBusy as exc:
            _log(str(exc))
            if args.once:
                return 2
            summary = None
        if summary is not None:
            print(json.dumps(summary, indent=1, default=str), flush=True)
            if args.once:
                aborted = any(f.get("aborted") for f in summary["forecasts"])
                return 1 if aborted else 0
        time.sleep(max(0.0, cfg.interval_minutes * 60 - (time.monotonic() - t0)))
