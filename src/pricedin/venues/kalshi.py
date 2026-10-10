"""Kalshi: resolved binary markets, historical snapshots from 1-minute candles, live order books.

Public REST API, no auth: ``https://api.elections.kalshi.com/trade-api/v2``. Markets settled before
the moving ``/historical/cutoff`` live only under ``/historical/...`` (same market field names,
candles without the ``_dollars``/``_fp`` suffixes); later ones under ``/markets``. Markets carry no
category, so it comes from ``/series`` (category, tags, frequency, fee terms) and the event title /
``mutually_exclusive`` from ``/events``.

Close times (checked 2026-10-10). ``close_time`` is overwritten with the actual close when a market
closes early. ``expected_expiration_time`` and ``latest_expiration_time`` are not: in
KXTRUMPMEETING-27JAN01 the still-open markets close 2027-01-01T14:59Z and markets that resolved YES
months earlier keep expected 2027-01-01T15:00Z / latest 2027-01-08T15:00Z beside their real close
(same for KXTRUMPSAYNICKNAME-27JAN01-POCA: closed 2026-10-02, expected 2027-01-01). Across 230
multi-market events whose markets closed >6 h apart, 223 share one expected_expiration_time; the
other 7 are date ladders where each market has its own deadline. No sampled market had an
expected_expiration_time that tracked an early close (of 7,790 markets with expected within 10 min
of close, the 234 with non-round times are scheduled closes such as 03:59:59 or game times).

But ``expected_expiration_time`` is usually the event time plus a settlement buffer that is a
constant of the series template (mention markets +14 d, Billboard +9 d, CPI +1.5 h). Anchoring on
it raw put as_of after the actual close for 61-73 % of eligible markets. So each question gets a
per-series buffer estimated only from what was known when it was listed. Most series close
every market at the event, whatever the outcome (mention markets close when the speech ends);
only "by <date>" ladders close early, and only on one outcome. So prior markets of the same
series are split by result, the median of max(0, expected - close_time) is taken within each
result, and the buffer is the smaller of the two: the outcome that runs to the deadline shows
the template gap, the early-closing one only adds to it. A prior counts once its result was
public before this market's ``created_at`` (``settlement_ts``, else the later of close and
expected expiration) and it closed within the 180 days before (at least 3 priors, else 0). Then
``close_at`` = ``expected_expiration_time`` - buffer, and horizons never depend on when (or
whether) this market closed early. ``scheduled_close_at`` is left empty: the estimate is not shown
to the model, which reads the deadline from the rules text. Markets without expected_expiration_time
are dropped. ``extra`` keeps ``scheduled_end`` (raw expected), ``buffer_h``, ``n_prior``,
``actual_close`` (``close_time``), ``early_close`` (actual close more than 1 h before ``close_at``)
and ``close_gap_h`` (close_at - actual close). For a few ladder events or delayed games expected is
earlier than the market's real deadline. ``latest_expiration_time`` is a far fallback
(deadline + 7 d to + 2 y) and ``expiration_time`` equals it.

Snapshots are None when as_of >= actual close - 60 s (the market had stopped trading). Otherwise
they read 1-minute candles over [as_of - max_stale, as_of]. A candle ending at T covers
(T - 1 min, T] and its closes are the state at T, so any candle with end <= as_of is usable.
Candles are emitted when the book or trades change, quiet minutes are omitted. We do not use
``include_latest_before_start``: on the live tier it adds a synthetic candle stamped at the window
start that carries older state, and the historical tier ignores the flag, so using it would make
the two tiers (early vs late study periods) select differently. A quote counts only when
0 < bid <= ask < 1: bid 0 means no bids and ask 1 means no asks, and averaging an empty side would
manufacture 0.5. Without a two-sided quote the mid is the last trade price clamped into the one
side that exists (a lone bid of 0.99 beats a stale 0.87 trade), else None.
"""

from __future__ import annotations

import bisect
import logging
import math
import re
import statistics
import threading
import weakref
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pricedin.http import HttpClient, HttpError
from pricedin.schema import HORIZONS, Question, Snapshot, parse_time, to_iso

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HOST = "api.elections.kalshi.com"
HOST_INTERVAL = 0.1  # ~10 req/s; the basic tier allows ~20 and 429s are retried by HttpClient
CANDLE_SOURCE = "kalshi-candles-1m"
PAGE = 1000
MAX_RETRIES = 10  # 429s come in bursts when other scans share the IP
BUFFER_LOOKBACK = timedelta(days=180)
BUFFER_MIN_PRIOR = 3

# Series whose markets live at most a few days (sampled 2026-10-10: daily max 3.99 d open,
# hourly 0.04 d, fifteen_min 0.01 d) can never pass a min_open_days >= 4 filter.
SHORT_FREQUENCIES = frozenset({"fifteen_min", "hourly", "daily"})
SKIP_CATEGORIES = frozenset({"Exotics"})  # multivariate combos / parlays

_CATEGORY_MAP = {
    "Politics": "politics",
    "Elections": "politics",
    "Economics": "economics",
    "Financials": "finance",
    "Companies": "finance",
    "Commodities": "finance",
    "Business": "finance",
    "Crypto": "crypto",
    "Climate and Weather": "weather",
    "Sports": "sports",
    "Entertainment": "culture",
    "Social": "culture",
    "Culture": "culture",
    "World": "world",
    "Health": "science",
}
_SCIENCE_TAGS = frozenset({"Space", "Public Health", "Medicine", "Physics & Math", "Energy"})
_MENTION_TAGS = {
    "Earnings": "finance",
    "Business": "finance",
    "CEOs": "finance",
    "Politicians": "politics",
    "Trump": "politics",
    "SOTU": "politics",
    "Sports": "sports",
    "Entertainment": "culture",
}


# -- plumbing ------------------------------------------------------------------------------------


def configure(client: HttpClient) -> HttpClient:
    """Give ``client`` the Kalshi host spacing unless the caller already chose one."""
    client.host_intervals.setdefault(HOST, HOST_INTERVAL)
    return client


def make_client(**kwargs: Any) -> HttpClient:
    """An HttpClient tuned for Kalshi (host spacing, patient retries); kwargs override.

    ``max_retries`` is per client in HttpClient, so a client shared with other hosts keeps its
    own setting; ``configure`` only adds the host spacing.
    """
    kwargs.setdefault("max_retries", MAX_RETRIES)
    kwargs.setdefault("backoff_max", 60.0)
    kwargs["host_intervals"] = {HOST: HOST_INTERVAL, **(kwargs.get("host_intervals") or {})}
    return HttpClient(**kwargs)


def _get(client: HttpClient, path: str, params: Mapping[str, Any] | None = None, *, cache=True):
    return client.get_json(f"{BASE}/{path}", params, use_cache=cache)


_memo: weakref.WeakKeyDictionary[HttpClient, dict[str, Any]] = weakref.WeakKeyDictionary()
_memo_lock = threading.Lock()


def _memoized(client: HttpClient, key: str, fn: Callable[[], Any]) -> Any:
    # Per-client, per-process memo for mutable listings that must not come from the disk cache.
    with _memo_lock:
        slot = _memo.setdefault(client, {})
        if key not in slot:
            slot[key] = fn()
        return slot[key]


def fetch_series(client: HttpClient) -> dict[str, dict[str, Any]]:
    """All series by ticker (~15k: category, tags, frequency, fee_type, fee_multiplier, title)."""

    def load() -> dict[str, dict[str, Any]]:
        body = _get(client, "series", {"limit": 20000}, cache=False)
        return {s["ticker"]: s for s in body.get("series") or []}

    return _memoized(client, "series", load)


def historical_cutoff(client: HttpClient) -> datetime:
    """Markets settled before this are served only by the ``/historical`` endpoints."""
    body = _memoized(client, "cutoff", lambda: _get(client, "historical/cutoff", cache=False))
    return parse_time(body["market_settled_ts"])


def _ts(t: datetime) -> int:
    return int(t.timestamp())


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _iso_or_blank(value: Any) -> str:
    t = _time(value)
    return to_iso(t) if t else ""


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        t = parse_time(value)
    except (TypeError, ValueError):
        return None
    return None if t.year < 2000 else t  # "0001-01-01T00:00:00Z" placeholders


# -- market -> Question --------------------------------------------------------------------------


def map_category(category: str, tags: Iterable[str] = ()) -> str:
    """Kalshi series category (+ tags) -> one of ``schema.CATEGORIES``."""
    tags = list(tags or ())
    if category == "Science and Technology":
        return "science" if any(t in _SCIENCE_TAGS for t in tags) else "tech"
    if category == "Mentions":
        for t in tags:
            if t in _MENTION_TAGS:
                return _MENTION_TAGS[t]
        return "other"
    return _CATEGORY_MAP.get(category, "other")


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("**", "")).strip()


def question_title(market: Mapping[str, Any]) -> str:
    """Market title, plus the YES outcome label when the title alone does not name it.

    Multi-market events often share one title ("Who will win the Nobel Peace Prize?") and put the
    outcome in ``yes_sub_title``; appending it makes each question self-contained.
    """
    title = _clean(market.get("title"))
    sub = _clean(market.get("yes_sub_title"))
    if sub and sub.lower() not in title.lower():
        return f"{title} — {sub}" if title else sub
    return title


def question_rules(market: Mapping[str, Any]) -> str:
    parts = [(market.get(k) or "").strip() for k in ("rules_primary", "rules_secondary")]
    return "\n\n".join(p for p in parts if p)


def fee_terms(series: Mapping[str, Any] | None) -> dict[str, Any]:
    s = series or {}
    mult = _num(s.get("fee_multiplier"))
    return {
        "kind": "kalshi",
        "fee_type": s.get("fee_type") or "quadratic",
        "multiplier": 1.0 if mult is None else mult,
    }


def taker_fee(price: float, contracts: float = 1.0, multiplier: float = 1.0) -> float:
    """Kalshi taker fee in dollars: ceil to the cent of 0.07 * m * C * P * (1 - P)."""
    raw = 0.07 * multiplier * contracts * price * (1.0 - price)
    return math.ceil(round(raw * 100, 9)) / 100


def scheduled_end(market: Mapping[str, Any]) -> datetime | None:
    """The listing-time deadline (``expected_expiration_time``), unaffected by early closes."""
    return _time(market.get("expected_expiration_time"))


class BufferIndex:
    """Per-series settlement buffers, using only markets settled before a question was listed."""

    def __init__(self, markets: Iterable[Mapping[str, Any]]):
        rows = []
        for m in markets:
            close, expected = _time(m.get("close_time")), scheduled_end(m)
            result = m.get("result")
            if close is None or expected is None or result not in ("yes", "no"):
                continue
            known = _time(m.get("settlement_ts")) or max(close, expected)
            gap = max(0.0, (expected - close).total_seconds())
            rows.append((known.timestamp(), close.timestamp(), gap, result))
        rows.sort()
        self._rows = rows
        self._known = [r[0] for r in rows]

    def buffer(
        self,
        listed_at: datetime,
        *,
        lookback: timedelta = BUFFER_LOOKBACK,
        min_count: int = BUFFER_MIN_PRIOR,
    ) -> tuple[float, int]:
        """(buffer hours, n priors): the smaller per-result median gap of markets whose result
        was public before ``listed_at`` and that closed in the ``lookback`` before it."""
        t = listed_at.timestamp()
        hi = bisect.bisect_left(self._known, t)  # result public strictly before listing
        lo_close = t - lookback.total_seconds()
        gaps: dict[str, list[float]] = {"yes": [], "no": []}
        for _, close, gap, result in self._rows[:hi]:
            if close >= lo_close:
                gaps[result].append(gap)
        n = len(gaps["yes"]) + len(gaps["no"])
        if n < min_count:
            return 0.0, n
        return min(statistics.median(g) for g in gaps.values() if g) / 3600, n


def settlement_buffer(
    prior: Iterable[Mapping[str, Any]],
    listed_at: datetime,
    *,
    lookback: timedelta = BUFFER_LOOKBACK,
    min_count: int = BUFFER_MIN_PRIOR,
) -> tuple[float, int]:
    """One-off form of ``BufferIndex(prior).buffer(listed_at)``."""
    return BufferIndex(prior).buffer(listed_at, lookback=lookback, min_count=min_count)


def market_to_question(
    market: Mapping[str, Any],
    series: Mapping[str, Any] | None = None,
    event: Mapping[str, Any] | None = None,
    *,
    buffer_h: float = 0.0,
    n_prior: int = 0,
) -> Question | None:
    """Map one raw market (live or historical object) to a Question; None if not a clean binary.

    ``buffer_h`` (from ``BufferIndex``) is subtracted from expected_expiration_time to give
    close_at; it must come from markets that closed before this one was listed.
    """
    result = market.get("result")
    if market.get("market_type") != "binary" or result not in ("yes", "no"):
        return None
    scheduled = scheduled_end(market)
    actual = _time(market.get("close_time"))
    created = _time(market.get("created_time"))
    opened = _time(market.get("open_time")) or created
    if scheduled is None or actual is None or created is None or opened is None:
        return None
    s = series or {}
    series_ticker = s.get("ticker") or (event or {}).get("series_ticker") or ""
    close_at = scheduled - timedelta(hours=buffer_h)
    gap_h = (close_at - actual).total_seconds() / 3600
    extra: dict[str, Any] = {
        "series": series_ticker,
        "actual_close": to_iso(actual),
        "scheduled_end": to_iso(scheduled),
        "buffer_h": round(buffer_h, 4),
        "n_prior": n_prior,
        "latest_expiration": _iso_or_blank(market.get("latest_expiration_time")),
        "settled_at": _iso_or_blank(market.get("settlement_ts")),
        "early_close": gap_h > 1.0,
        "close_gap_h": round(gap_h, 3),
        "can_close_early": bool(market.get("can_close_early")),
        "fee": fee_terms(series),
        "volume": _num(market.get("volume_fp")) or 0.0,
        "open_interest": _num(market.get("open_interest_fp")),
        "frequency": s.get("frequency", ""),
        "tags": list(s.get("tags") or []),
    }
    if event is not None:
        extra["mutually_exclusive"] = event.get("mutually_exclusive")
        extra["event_title"] = _clean(event.get("title"))
        extra["event_sub_title"] = _clean(event.get("sub_title"))
    venue_category = s.get("category") or (event or {}).get("category") or ""
    return Question(
        id=f"kalshi:{market['ticker']}",
        venue="kalshi",
        event_id=f"kalshi:{market.get('event_ticker') or market['ticker']}",
        title=question_title(market),
        rules=question_rules(market),
        category=map_category(venue_category, s.get("tags") or ()),
        created_at=to_iso(created),
        open_at=to_iso(opened),
        close_at=to_iso(close_at),
        scheduled_close_at="",  # the buffer is an estimate; the rules state the deadline
        outcome=1 if result == "yes" else 0,
        volume=extra["volume"],
        url=f"https://kalshi.com/markets/{series_ticker.lower()}" if series_ticker else "",
        venue_category=venue_category,
        extra=extra,
    )


def is_eligible(
    q: Question,
    *,
    created_after: datetime,
    closed_before: datetime,
    min_open_days: float,
    min_volume: float,
) -> bool:
    """Listing-time filters plus "finished trading by closed_before" (both closes)."""
    close = parse_time(q.close_at)
    return (
        parse_time(q.created_at) >= created_after
        and close <= closed_before
        and parse_time(q.extra["actual_close"]) <= closed_before
        and parse_time(q.open_at) <= close - timedelta(days=min_open_days)
        and q.volume >= min_volume
    )


# -- listing -------------------------------------------------------------------------------------


def _pages(client: HttpClient, path: str, params: dict[str, Any], key: str) -> Iterator[list[dict]]:
    cursor = ""
    while True:
        body = _get(client, path, {**params, "cursor": cursor} if cursor else params, cache=False)
        items = body.get(key) or []
        yield items
        cursor = body.get("cursor") or ""
        if not cursor or not items:
            return


def series_markets(
    client: HttpClient, series_ticker: str, created_after: datetime, cutoff: datetime
) -> list[dict[str, Any]]:
    """Settled markets of one series that closed at or after ``created_after``, both tiers.

    Live tier filters on settlement time, so a few markets closed slightly earlier come along.
    """
    out: dict[str, dict[str, Any]] = {}
    live = {
        "status": "settled",
        "series_ticker": series_ticker,
        "min_settled_ts": _ts(created_after),
        "limit": PAGE,
    }
    for page in _pages(client, "markets", live, "markets"):
        for m in page:
            out.setdefault(m["ticker"], m)
    if created_after < cutoff:
        hist = {"series_ticker": series_ticker, "limit": PAGE}
        for page in _pages(client, "historical/markets", hist, "markets"):
            closes = []
            for m in page:
                out.setdefault(m["ticker"], m)
                if t := _time(m.get("close_time")):
                    closes.append(t)
            # newest first by close: once a whole page closed before created_after, stop
            if closes and max(closes) < created_after:
                break
    return list(out.values())


def series_events(
    client: HttpClient, series_ticker: str, created_after: datetime
) -> dict[str, dict[str, Any]]:
    params = {"series_ticker": series_ticker, "min_close_ts": _ts(created_after), "limit": 200}
    out: dict[str, dict[str, Any]] = {}
    for page in _pages(client, "events", params, "events"):
        for e in page:
            out[e["event_ticker"]] = e
    return out


def scan_series(
    series_tickers: Iterable[str], series: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Series worth scanning: drop short-lived frequencies and combo categories."""
    keep = []
    for t in series_tickers:
        s = series.get(t, {})
        if s.get("frequency") in SHORT_FREQUENCIES or s.get("category") in SKIP_CATEGORIES:
            continue
        if t.startswith("KXMVE"):
            continue
        keep.append(t)
    return keep


def list_candidates(
    client: HttpClient,
    *,
    created_after: datetime,
    closed_before: datetime,
    min_open_days: float = 8,
    min_volume: float,
    series: Iterable[str] | None = None,
    workers: int = 8,
    progress: Callable[[int, int, int], None] | None = None,
    skip_series: Collection[str] = (),
    on_series: Callable[[str, list[Question]], None] | None = None,
) -> Iterator[Question]:
    """Every eligible resolved binary Kalshi market, one series at a time.

    ``skip_series`` are left out (already scanned); ``on_series(ticker, questions)`` is called
    once per finished series, before its questions are yielded, so a caller can checkpoint.

    ``series`` restricts the scan to those series tickers (as given, no frequency filter);
    otherwise every series from ``/series`` minus short-lived frequencies and combos.
    ``progress(done_series, total_series, n_yielded)`` is called after each series. Series that
    fail after HttpClient's retries are logged and skipped. Listing responses are never cached
    (they change as markets settle and move to the historical tier). Series with markets listed
    since ``created_after`` are fetched again back to ``BUFFER_LOOKBACK`` earlier for the
    settlement buffers. A full scan is ~2 requests per series plus ~3 per candidate series
    (~32k), so roughly an hour at 10 req/s. Use ``make_client`` for patient 429 retries.
    """
    configure(client)
    all_series = fetch_series(client)
    tickers = list(series) if series is not None else scan_series(sorted(all_series), all_series)
    skip = set(skip_series)
    tickers = [t for t in tickers if t not in skip]
    cutoff = historical_cutoff(client)
    kw = dict(
        created_after=created_after,
        closed_before=closed_before,
        min_open_days=min_open_days,
        min_volume=min_volume,
    )

    def work(ticker: str) -> list[Question]:
        s = all_series.get(ticker) or {"ticker": ticker}
        markets = series_markets(client, ticker, created_after, cutoff)
        listed = [m for m in markets if (t := _time(m.get("created_time"))) and t >= created_after]
        if not listed:
            return []
        # Buffers need the series' markets closed up to BUFFER_LOOKBACK before created_after.
        # They only set each question's anchor; they are never questions themselves.
        markets = series_markets(client, ticker, created_after - BUFFER_LOOKBACK, cutoff)
        index = BufferIndex(markets)
        qs = []
        for m in listed:
            buf, n_prior = index.buffer(_time(m["created_time"]))
            q = market_to_question(m, s, buffer_h=buf, n_prior=n_prior)
            if q is not None and is_eligible(q, **kw):
                qs.append(q)
        if qs:
            events = series_events(client, ticker, created_after)
            for q in qs:
                ev = events.get(q.event_id.removeprefix("kalshi:"))
                if ev is not None:
                    q.extra["mutually_exclusive"] = ev.get("mutually_exclusive")
                    q.extra["event_title"] = _clean(ev.get("title"))
                    q.extra["event_sub_title"] = _clean(ev.get("sub_title"))
        return qs

    seen: set[str] = set()
    done = 0
    n = 0
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        it = iter(tickers)
        pending: dict[Future[list[Question]], str] = {}
        for t in it:  # keep a bounded number in flight so early exit is cheap
            pending[pool.submit(work, t)] = t
            if len(pending) >= 4 * workers:
                break
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                ticker = pending.pop(fut)
                try:
                    qs = fut.result()
                except HttpError as exc:
                    log.warning("kalshi: skipping series %s: %s", ticker, exc)
                    qs = None
                if qs is None:
                    qs = []
                elif on_series is not None:
                    on_series(ticker, qs)
                for q in qs:
                    if q.id not in seen:
                        seen.add(q.id)
                        n += 1
                        yield q
                done += 1
                if progress is not None:
                    progress(done, len(tickers), n)
                nxt = next(it, None)
                if nxt is not None:
                    pending[pool.submit(work, nxt)] = nxt
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# -- historical snapshot -------------------------------------------------------------------------


@dataclass
class Observation:
    ts: int  # candle end (unix seconds); its closes are the state at this time
    bid: float | None
    ask: float | None
    last: float | None


def _side(block: Any, name: str) -> float | None:
    if not isinstance(block, Mapping):
        return None
    # live candles suffix "_dollars", historical ones do not
    v = block.get(f"{name}_dollars", block.get(name))
    return _num(v)


def parse_candle(c: Mapping[str, Any]) -> Observation:
    """One candle (live or historical naming) -> end time, YES bid/ask close, last trade."""
    price = c.get("price") or {}
    last = _side(price, "close")
    if last is None:
        last = _side(price, "previous")  # no trade in this minute: the last one before it
    return Observation(
        ts=int(c["end_period_ts"]),
        bid=_side(c.get("yes_bid"), "close"),
        ask=_side(c.get("yes_ask"), "close"),
        last=last,
    )


def select_observation(
    candles: Iterable[Mapping[str, Any]], as_of: datetime, max_stale: timedelta
) -> Observation | None:
    """The latest candle with as_of - max_stale <= end <= as_of, or None."""
    hi = as_of.timestamp()
    lo = hi - max_stale.total_seconds()
    best: Observation | None = None
    for c in candles:
        obs = parse_candle(c)
        if lo <= obs.ts <= hi and (best is None or obs.ts > best.ts):
            best = obs
    return best


def book_mid(bid: float | None, ask: float | None, last: float | None) -> float | None:
    """Mid of a two-sided book, else the last trade clamped into the side that exists, else None.

    ``bid``/``ask`` must already be None for empty sides.
    """
    if bid is not None and ask is not None and bid <= ask:
        return round((bid + ask) / 2, 6)
    if last is None or not 0.0 < last < 1.0:
        return None
    lo = bid if bid is not None else 0.0
    hi = ask if ask is not None else 1.0
    return min(max(last, lo), hi)


def quote(obs: Observation) -> tuple[float, float | None, float | None] | None:
    """(mid, bid, ask) from an observation; empty book sides become None. None if no price."""
    bid = obs.bid if obs.bid is not None and 0.0 < obs.bid < 1.0 else None
    ask = obs.ask if obs.ask is not None and 0.0 < obs.ask < 1.0 else None
    mid = book_mid(bid, ask, obs.last)
    return None if mid is None else (mid, bid, ask)


def _ticker(q: Question | str) -> str:
    return q.id.removeprefix("kalshi:") if isinstance(q, Question) else q.removeprefix("kalshi:")


def _candles(
    client: HttpClient, q: Question, start: int, end: int, cutoff: datetime
) -> list[dict[str, Any]]:
    ticker = _ticker(q)
    series = q.extra.get("series") or ticker.split("-")[0]
    settled = _time(q.extra.get("settled_at")) or parse_time(q.close_at)
    params = {"period_interval": 1, "start_ts": start, "end_ts": end}
    live = (f"series/{series}/markets/{ticker}/candlesticks", params)
    hist = (f"historical/markets/{ticker}/candlesticks", params)
    order = [hist, live] if settled < cutoff else [live, hist]
    for i, (path, p) in enumerate(order):
        try:
            return _get(client, path, p).get("candlesticks") or []
        except HttpError as exc:
            if exc.status != 404 or i == len(order) - 1:
                raise
    return []


def snapshot(
    client: HttpClient,
    question: Question,
    horizon: str,
    *,
    max_stale: timedelta = timedelta(hours=6),
) -> Snapshot | None:
    """Market state at close_at - HORIZONS[horizon] from 1-minute candles.

    None when the market had already closed by as_of (as_of >= actual close - 60 s), or when no
    usable candle ends within ``max_stale`` before as_of.
    Uses only candles ending at or before as_of. Responses are cached (settled data is final).
    """
    configure(client)
    as_of = parse_time(question.close_at) - HORIZONS[horizon]
    end = math.floor(as_of.timestamp())
    start = end - int(max_stale.total_seconds())
    actual = _time(question.extra.get("actual_close"))
    if actual is not None and as_of >= actual - timedelta(seconds=60):
        return None  # trading had stopped by as_of
    if as_of < parse_time(question.open_at):
        return None
    candles = _candles(client, question, start, end, historical_cutoff(client))
    obs = select_observation(candles, as_of, max_stale)
    if obs is None:
        return None
    q = quote(obs)
    if q is None:
        return None
    mid, bid, ask = q
    return Snapshot(
        question_id=question.id,
        horizon=horizon,
        as_of=to_iso(as_of),
        mid=mid,
        bid=bid,
        ask=ask,
        last=obs.last,
        price_at=to_iso(datetime.fromtimestamp(obs.ts, tz=UTC)),
        source=CANDLE_SOURCE,
    )


# -- live ----------------------------------------------------------------------------------------


def parse_orderbook(body: Mapping[str, Any], depth: int = 10) -> dict[str, list[list[float]]]:
    """Kalshi book (YES and NO bids, ascending) -> YES-terms {"bids", "asks"}, best first."""
    ob = body.get("orderbook_fp") or {}

    def levels(key: str) -> list[list[float]]:
        out = []
        for row in ob.get(key) or []:
            p, size = _num(row[0]), _num(row[1])
            if p is not None and size is not None and size > 0:
                out.append([p, size])
        return sorted(out, key=lambda r: -r[0])[:depth]

    bids = levels("yes_dollars")
    asks = [[round(1.0 - p, 6), size] for p, size in levels("no_dollars")]
    return {"bids": bids, "asks": asks}


def live_snapshot(
    client: HttpClient,
    question_or_market_id: Question | str,
    *,
    depth: int = 10,
    horizon: str = "live",
) -> Snapshot | None:
    """Current top-of-book for the prospective collector (never cached).

    A one-sided book falls back to the last trade clamped into the side that exists, and the
    result is None when there is neither a two-sided book nor a trade.
    """
    configure(client)
    ticker = _ticker(question_or_market_id)
    now = datetime.now(UTC)
    book = parse_orderbook(
        _get(client, f"markets/{ticker}/orderbook", {"depth": depth}, cache=False), depth
    )
    bid = book["bids"][0][0] if book["bids"] else None
    ask = book["asks"][0][0] if book["asks"] else None
    last = None
    if bid is None or ask is None or bid > ask:
        market = _get(client, f"markets/{ticker}", cache=False).get("market") or {}
        last = _num(market.get("last_price_dollars"))
    mid = book_mid(bid, ask, last)
    if mid is None:
        return None
    return Snapshot(
        question_id=f"kalshi:{ticker}",
        horizon=horizon,
        as_of=to_iso(now),
        mid=mid,
        bid=bid,
        ask=ask,
        last=last,
        price_at=to_iso(now),
        source="live-book",
        book=book,
    )
