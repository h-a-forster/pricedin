"""Polymarket: resolved binary markets (Gamma API) and their price history (CLOB API).

Candidates come from a keyset scan of ``gamma-api /markets/keyset`` (closed markets, filtered by
``startDate`` and volume server-side, the rest client-side). Historical prices come from
``clob /prices-history`` at 1-minute fidelity; its ``p`` looks like the YES midpoint and there is
no historical bid/ask. The live order book comes from ``clob /book``.

Leakage notes, from a survey of ~22k markets listed Jun-Aug 2026 (see the build report):

- ``endDate`` is the nominal end set at listing and is not overwritten by an early resolution
  (it only equals ``closedTime`` for short crypto markets that close automatically), so it is
  ``close_at``: horizons never depend on when (or whether) the market resolved early. Markets
  with no endDate, or one before ``startDate``/``createdAt``, are dropped. Snapshots whose
  ``as_of`` falls after the actual close are not observable and return None.
- ``description`` is the rules text. No post-resolution "Update:"/"Clarification" paragraphs were
  found, but a conservative detector flags (and strips or drops) them in case they appear.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pricedin.http import HttpClient, HttpError
from pricedin.schema import HORIZONS, Question, Snapshot, parse_time, to_iso

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# Gamma /markets allows 300 req/10 s and CLOB /prices-history 1000 req/10 s; stay below both.
HOST_INTERVALS: dict[str, float] = {
    "gamma-api.polymarket.com": 0.04,
    "clob.polymarket.com": 0.015,
}

PAGE_LIMIT = 100
SCAN_WINDOW = timedelta(days=7)  # wide sparse keyset scans time out server-side (HTTP 500)
MIN_SCAN_WINDOW = timedelta(hours=6)
MAX_STALE = timedelta(hours=6)
# A prices-history point stamped t is a sample taken a few seconds into a minute. We cannot rule
# out that it summarises the minute that follows, so only points with t <= as_of - GUARD count.
HISTORY_GUARD = timedelta(seconds=60)
BOOK_LEVELS = 10

# Tag slug -> category, checked in this order; the first tag present wins. Specific beats broad:
# a Fed market tagged politics+economy is economics, an Iran deal tagged politics+geopolitics is
# world, a stock market tagged tech+finance is finance, anything tagged sports is sports.
TAG_PRECEDENCE: tuple[tuple[str, str], ...] = (
    ("sports", "sports"),
    ("esports", "sports"),
    ("crypto", "crypto"),
    ("weather", "weather"),
    ("economy", "economics"),
    ("finance", "finance"),
    ("tech", "tech"),
    ("science", "science"),
    ("pop-culture", "culture"),
    ("culture", "culture"),
    ("entertainment", "culture"),
    ("geopolitics", "world"),
    ("politics", "politics"),
    ("elections", "politics"),
    ("world", "world"),
)


# --- small parsers ---------------------------------------------------------------------------


def make_client(**kwargs: Any) -> HttpClient:
    """An ``HttpClient`` with Polymarket host intervals (other kwargs pass through)."""
    intervals = {**HOST_INTERVALS, **kwargs.pop("host_intervals", {})}
    return HttpClient(host_intervals=intervals, **kwargs)


def _tune(client: HttpClient) -> None:
    # A generic client defaults to 1 req/s for unknown hosts; keep any explicit override.
    for host, interval in HOST_INTERVALS.items():
        client.host_intervals.setdefault(host, interval)


def parse_pm_time(value: str | None) -> datetime | None:
    """Gamma timestamps: ISO with ``Z`` or ``closedTime`` style ``2026-09-16 20:02:43+00``."""
    if not value:
        return None
    s = value.strip().replace(" ", "T", 1)
    if re.search(r"[+-]\d{2}$", s):
        s += ":00"
    try:
        return parse_time(s)
    except ValueError:
        return None


def _json_list(value: Any) -> list[Any]:
    # outcomes / outcomePrices / clobTokenIds are JSON arrays encoded as strings.
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            out = json.loads(value)
        except ValueError:
            return []
        return out if isinstance(out, list) else []
    return []


def yes_no_tokens(market: dict[str, Any]) -> tuple[str, str] | None:
    """(YES token, NO token) matched by outcome name, not position; None unless Yes/No binary."""
    outcomes = [str(o).strip().lower() for o in _json_list(market.get("outcomes"))]
    tokens = [str(t) for t in _json_list(market.get("clobTokenIds"))]
    if sorted(outcomes) != ["no", "yes"] or len(tokens) != 2:
        return None
    return tokens[outcomes.index("yes")], tokens[outcomes.index("no")]


def resolved_outcome(market: dict[str, Any]) -> int | None:
    """1 if YES paid 1, 0 if NO paid 1; None for 50/50, unresolved or non-Yes/No markets."""
    outcomes = [str(o).strip().lower() for o in _json_list(market.get("outcomes"))]
    try:
        prices = [float(p) for p in _json_list(market.get("outcomePrices"))]
    except (TypeError, ValueError):
        return None
    if sorted(outcomes) != ["no", "yes"] or len(prices) != 2:
        return None
    pay = dict(zip(outcomes, prices, strict=True))
    if pay["yes"] == 1.0 and pay["no"] == 0.0:
        return 1
    if pay["yes"] == 0.0 and pay["no"] == 1.0:
        return 0
    return None


def tag_slugs(market: dict[str, Any]) -> list[str]:
    return [str(t.get("slug")) for t in market.get("tags") or [] if t.get("slug")]


def map_category(tags: list[str]) -> tuple[str, str]:
    """(category, raw tag that decided it) by ``TAG_PRECEDENCE``; ("other", "") if none match."""
    present = set(tags)
    for tag, cat in TAG_PRECEDENCE:
        if tag in present:
            return cat, tag
    return "other", ""


def fee_terms(market: dict[str, Any]) -> dict[str, Any]:
    """``extra["fee"]``: the market's own fee schedule (fixed at creation)."""
    sched = market.get("feeSchedule") or {}
    enabled = bool(market.get("feesEnabled")) and bool(sched)
    return {
        "kind": "polymarket",
        "enabled": enabled,
        "fee_type": market.get("feeType"),
        "rate": float(sched.get("rate") or 0.0) if enabled else 0.0,
        "exponent": float(sched.get("exponent") or 1) if enabled else 1.0,
        "taker_only": bool(sched.get("takerOnly", True)),
        "rebate_rate": float(sched.get("rebateRate") or 0.0) if enabled else 0.0,
    }


def taker_fee(fee: dict[str, Any], shares: float, price: float) -> float:
    """USDC taker fee: shares x rate x (p(1-p))^exponent, rounded to 5 dp (makers pay nothing).

    The docs give ``C x feeRate x p x (1 - p)`` and describe ``exponent`` as applied to the price
    component; every market seen so far has exponent 1, where both readings agree.
    """
    if not fee.get("enabled") or fee.get("rate", 0.0) <= 0:
        return 0.0
    core = (price * (1.0 - price)) ** float(fee.get("exponent", 1.0))
    return round(shares * float(fee["rate"]) * core, 5)


# --- rules text ------------------------------------------------------------------------------

# A paragraph that opens like a moderator note appended after listing.
_APPENDIX = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:update[sd]?|clarification|edit(?:ed)?|additional context|"
    r"resolution (?:update|note))\b[^\n:]{0,40}[:\-–—]"
)
# Statements that only make sense once the outcome is known.
_RESOLVED_STATEMENT = re.compile(
    r"(?i)\b(?:this market (?:has|was) (?:been )?resolved|has (?:already )?resolved|"
    r"was resolved (?:to|as)|resolved (?:to|as) (?:\"|')?(?:yes|no)\b|"
    r"market (?:will be|has been) resolved (?:to|as) (?:\"|')?(?:yes|no)\b)"
)
# A note naming the specific item that appeared after listing (e.g. MrBeast's "next video").
_POST_LISTING_REFERENCE = re.compile(r"(?i)\bnote:\s*this market refers to the [^\n]{0,40}titled")


def check_rules(text: str) -> tuple[str, list[str], bool]:
    """Conservative leak check on a description: (cleaned text, flags, drop).

    An appended Update/Clarification paragraph is stripped (flag ``stripped_appendix``); text
    stating the market has resolved means drop (flag ``resolution_statement``); a note naming an
    item that appeared after listing is kept but flagged ``post_listing_reference``.
    """
    flags: list[str] = []
    cleaned = text
    m = _APPENDIX.search(text)
    if m and m.start() > 0:
        cleaned = text[: m.start()].rstrip()
        flags.append("stripped_appendix")
    elif m:  # the whole description opens like an update: keep it but flag it
        flags.append("appendix_at_start")
    drop = bool(_RESOLVED_STATEMENT.search(text))
    if drop:
        flags.append("resolution_statement")
    if _POST_LISTING_REFERENCE.search(cleaned):
        flags.append("post_listing_reference")
    return cleaned, flags, drop


_COMMON_CAPS = {"Will", "What", "Who", "Which", "When", "How", "The", "A", "An", "In", "By", "On"}


def self_contained_title(question: str, event_title: str) -> str:
    """Append the event title when it names things the question leaves out.

    E.g. "Will Henan FC win on 2026-07-17?" in event "Henan FC vs. Qingdao Hainiu FC" needs the
    opponent; "Will Bitcoin dip to $52,500 in July?" does not need "What price will Bitcoin hit in
    July?". The test: does the event title carry a capitalised word (not its first word) that the
    question lacks?
    """
    q, ev = question.strip(), (event_title or "").strip()
    if not ev or ev.lower() in q.lower():
        return q
    words = re.findall(r"[^\W\d_][\w'.-]*", ev)
    q_low = q.lower()
    missing = [
        w for w in words[1:] if w[0].isupper() and w not in _COMMON_CAPS and w.lower() not in q_low
    ]
    return f"{q} ({ev})" if missing else q


# --- title deadline --------------------------------------------------------------------------

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    )
}
# "by|on|before [Weekday,] <Month> <day>[st][, <year>]"; a day range "July 2-3" keeps its end.
# Period forms like "in July 2026" are deliberately not matched (monthly data is released later).
_TITLE_DEADLINE = re.compile(
    r"(?i)\b(?:by|on|before)\s+(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s+)?"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?!\d)"
    r"(?:st|nd|rd|th)?(?:\s*[-\u2013]\s*(\d{1,2})(?!\d))?(?:,?\s+(\d{4})(?!\d))?"
)
# End of the named day in US Eastern. tzdata is not guaranteed on Windows, so use EST (UTC-5)
# year-round: 23:59 EST is the later of the two readings, so it never excludes more.
_ET_CONSERVATIVE = timedelta(hours=-5)
# A title deadline this far before endDate means horizons would fall after the question was
# effectively decided.
TITLE_DEADLINE_SLACK = timedelta(days=1)


def title_deadline(title: str, close: datetime, created: datetime) -> datetime | None:
    """End (23:59 ET, as UTC) of the specific day the title names as its deadline, or None.

    Listing-time only: uses the title, endDate (``close``) and createdAt. The first
    "by|on|before <Month> <day>[, <year>]" wins. Without a year, the day is the first one on or
    after the creation date (in ET), so a December listing asking "by January 5" means next year.
    """
    m = _TITLE_DEADLINE.search(title)
    if not m:
        return None
    month = _MONTHS[m.group(1).lower()[:3]]
    day = int(m.group(3) or m.group(2))
    created_et = (created + _ET_CONSERVATIVE).date()
    if m.group(4):
        years = [int(m.group(4))]
    else:
        years = list(range(created_et.year, max(created_et.year, close.year) + 2))
    for y in years:
        try:
            named = date(y, month, day)
        except ValueError:  # e.g. February 30, or February 29 outside a leap year
            continue
        if m.group(4) or named >= created_et:
            return datetime(y, month, day, 23, 59, tzinfo=UTC) - _ET_CONSERVATIVE
    return None


def question_title_deadline(q: Question) -> datetime | None:
    return title_deadline(q.title, q.close, parse_time(q.created_at))


# --- market -> Question ----------------------------------------------------------------------


def parse_market(market: dict[str, Any]) -> tuple[Question | None, str]:
    """Raw Gamma market -> (Question, "") or (None, reason it is not eligible).

    Venue-level checks only (binary Yes/No, clean resolution, times present, rules leak check);
    date and duration windows are applied in ``list_candidates``.
    """
    if not market.get("closed", True):
        return None, "open"
    tokens = yes_no_tokens(market)
    if tokens is None:
        return None, "not_yes_no"
    outcome = resolved_outcome(market)
    if outcome is None:
        return None, "unclean_resolution"
    status = market.get("umaResolutionStatus")
    if status not in (None, "resolved"):
        return None, f"uma_{status}"
    slug = str(market.get("slug") or "")
    if "parlay" in slug or "up-or-down" in slug:
        return None, "combo_or_updown"
    created = parse_pm_time(market.get("createdAt"))
    opened = parse_pm_time(market.get("startDate")) or created
    actual = parse_pm_time(market.get("closedTime"))
    scheduled = parse_pm_time(market.get("endDate"))
    if created is None or opened is None or actual is None:
        return None, "missing_times"
    # Horizons hang off the listing-time endDate, never the actual close (which depends on the
    # outcome). Missing or impossible endDates are listing-time attributes, so dropping is safe.
    if scheduled is None:
        return None, "no_end_date"
    if scheduled <= max(opened, created):
        return None, "end_date_before_open"

    rules, flags, drop = check_rules(str(market.get("description") or ""))
    if drop:
        return None, "rules_leak"

    events = market.get("events") or [{}]
    ev = events[0] or {}
    tags = tag_slugs(market)
    category, cat_tag = map_category(tags)
    cid = str(market.get("conditionId") or "")
    if not cid:
        return None, "no_condition_id"
    extra: dict[str, Any] = {
        "market_id": str(market.get("id") or ""),
        "condition_id": cid,
        "slug": slug,
        "event_slug": ev.get("slug"),
        "event_title": ev.get("title"),
        "series": ev.get("seriesSlug"),
        "group_item": market.get("groupItemTitle"),
        "actual_close": to_iso(actual),
        "scheduled_end": to_iso(scheduled),
        "early_close": actual < scheduled,
        "close_gap_h": round((actual - scheduled).total_seconds() / 3600, 2),
        "yes_token": tokens[0],
        "no_token": tokens[1],
        "neg_risk": bool(market.get("negRisk")),
        "neg_risk_id": market.get("negRiskMarketID"),
        "tick_size": _float_or_none(market.get("orderPriceMinTickSize")),
        "fee": fee_terms(market),
        "tags": tags,
    }
    if flags:
        extra["rules_flag"] = flags
    title = self_contained_title(str(market.get("question") or ""), str(ev.get("title") or ""))
    deadline = title_deadline(title, scheduled, created)
    if deadline is not None:
        extra["title_deadline"] = to_iso(deadline)
    q = Question(
        id=f"polymarket:{cid}",
        venue="polymarket",
        event_id=f"polymarket:{ev.get('id') or cid}",
        title=title,
        rules=rules,
        category=category,
        created_at=to_iso(created),
        open_at=to_iso(opened),
        close_at=to_iso(scheduled),
        scheduled_close_at=to_iso(scheduled),
        outcome=outcome,
        volume=float(market.get("volumeNum") or 0.0),
        url=f"https://polymarket.com/event/{ev.get('slug')}" if ev.get("slug") else "",
        venue_category=cat_tag,
        extra=extra,
    )
    return q, ""


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def eligible(
    q: Question,
    *,
    created_after: datetime,
    closed_before: datetime,
    min_open_days: float,
    min_volume: float,
) -> str:
    """ "" if ``q`` passes the study windows, else the reason it does not."""
    if parse_time(q.created_at) < created_after:
        return "created_before"
    actual = parse_time(q.extra["actual_close"])
    if q.close > closed_before or actual > closed_before:
        return "closed_after"
    if parse_time(q.open_at) > q.close - timedelta(days=min_open_days):
        return "too_short"
    if q.volume < min_volume:
        return "low_volume"
    # Recomputed from the Question (not extra) so saved candidates are re-checked on --reuse.
    deadline = question_title_deadline(q)
    if deadline is not None and q.close - deadline > TITLE_DEADLINE_SLACK:
        return "title_deadline_before_end"
    return ""


# --- listing ---------------------------------------------------------------------------------


def _scan_window(
    client: HttpClient, start: datetime, end: datetime, min_volume: float
) -> Iterator[dict[str, Any]]:
    params: dict[str, Any] = {
        "closed": "true",
        "limit": PAGE_LIMIT,
        "start_date_min": to_iso(start),
        "start_date_max": to_iso(end),
        "include_tag": "true",
    }
    if min_volume > 0:
        params["volume_num_min"] = min_volume
    cursor = None
    while True:
        p = dict(params, after_cursor=cursor) if cursor else params
        page = client.get_json(f"{GAMMA}/markets/keyset", p)
        markets = page.get("markets") or []
        yield from markets
        cursor = page.get("next_cursor")
        if not cursor or not markets:
            return


def iter_markets(
    client: HttpClient,
    start: datetime,
    end: datetime,
    *,
    min_volume: float = 0.0,
    window: timedelta = SCAN_WINDOW,
    progress: Callable[[str], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Raw closed markets with ``start <= startDate <= end``, each once.

    Scans in ``window``-sized slices; a slice that keeps failing (the keyset endpoint returns 500
    on wide sparse scans) is split in half down to ``MIN_SCAN_WINDOW``.
    """
    _tune(client)
    seen: set[str] = set()
    stack = []
    a = start
    while a < end:
        b = min(a + window, end)
        stack.append((a, b))
        a = b
    stack.reverse()
    pages_before = client.stats["requests"]
    while stack:
        a, b = stack.pop()
        try:
            for m in _scan_window(client, a, b, min_volume):
                mid = str(m.get("id"))
                if mid not in seen:
                    seen.add(mid)
                    yield m
        except HttpError:
            if b - a <= MIN_SCAN_WINDOW:
                raise
            half = a + (b - a) / 2
            stack.extend([(half, b), (a, half)])
            continue
        if progress:
            n = client.stats["requests"] - pages_before
            progress(f"startDate {to_iso(a)}..{to_iso(b)}: {len(seen)} markets, {n} requests")


def list_candidates(
    client: HttpClient,
    *,
    created_after: datetime,
    closed_before: datetime,
    min_open_days: float = 8,
    min_volume: float = 0.0,
    progress: Callable[[str], None] | None = None,
    reasons: dict[str, int] | None = None,
) -> Iterator[Question]:
    """Every eligible resolved Yes/No Polymarket market (see the venue module contract).

    ``createdAt <= startDate``, so ``start_date_min=created_after`` is a safe server-side bound
    (``createdAt`` is re-checked here); ``startDate <= closed_before - min_open_days`` bounds the
    other end. Both the scheduled end (``close_at``) and the actual close must be
    ``<= closed_before``. Pass a dict as ``reasons`` to collect drop counts.
    """
    stop = closed_before - timedelta(days=min_open_days)
    for m in iter_markets(client, created_after, stop, min_volume=min_volume, progress=progress):
        q, why = parse_market(m)
        if q is not None:
            why = eligible(
                q,
                created_after=created_after,
                closed_before=closed_before,
                min_open_days=min_open_days,
                min_volume=min_volume,
            )
        if reasons is not None:
            reasons[why or "ok"] = reasons.get(why or "ok", 0) + 1
        if q is not None and not why:
            yield q


# --- prices ----------------------------------------------------------------------------------


def pick_point(
    history: list[dict[str, Any]],
    as_of: datetime,
    *,
    max_stale: timedelta = MAX_STALE,
    guard: timedelta = HISTORY_GUARD,
) -> tuple[datetime, float] | None:
    """The last (t, p) with t <= as_of - guard and t >= as_of - max_stale, else None.

    Malformed points and prices that are not finite or not in [0, 1] are skipped.
    """
    latest = as_of - guard
    best: tuple[float, float] | None = None
    for pt in history:
        try:
            t, p = float(pt["t"]), float(pt["p"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(t) and _is_price(p)):
            continue
        if t <= latest.timestamp() and (best is None or t > best[0]):
            best = (t, p)
    if best is None:
        return None
    when = datetime.fromtimestamp(best[0], tz=UTC)
    if as_of - when > max_stale:
        return None
    return when, best[1]


def price_history(
    client: HttpClient, token: str, start: datetime, end: datetime, *, fidelity: int = 1
) -> list[dict[str, Any]]:
    """``/prices-history`` points in [start, end] (window must be under ~16 days)."""
    _tune(client)
    body = client.get_json(
        f"{CLOB}/prices-history",
        {
            "market": token,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "fidelity": fidelity,
        },
    )
    return list((body or {}).get("history") or [])


def snapshot(
    client: HttpClient, question: Question, horizon: str, *, max_stale: timedelta = MAX_STALE
) -> Snapshot | None:
    """YES mid at ``close_at - HORIZONS[horizon]`` from 1-minute price history, or None.

    None also when the market had already closed by ``as_of`` (within ``HISTORY_GUARD``): that
    row was not observable in real time.
    """
    as_of = question.close - HORIZONS[horizon]
    token = question.extra.get("yes_token")
    if not token or as_of <= parse_time(question.open_at):
        return None
    actual = question.extra.get("actual_close")
    if actual and as_of >= parse_time(actual) - HISTORY_GUARD:
        return None
    hist = price_history(client, str(token), as_of - max_stale, as_of)
    hit = pick_point(hist, as_of, max_stale=max_stale)
    if hit is None:
        return None
    when, p = hit
    return Snapshot(
        question_id=question.id,
        horizon=horizon,
        as_of=to_iso(as_of),
        mid=p,
        price_at=to_iso(when),
        source="polymarket-prices-history",
    )


def _is_price(p: float) -> bool:
    return math.isfinite(p) and 0.0 <= p <= 1.0


def _levels(raw: list[dict[str, Any]], *, best_high: bool) -> list[list[float]]:
    """Book levels as [price, size], best first; malformed, out-of-range or empty levels dropped."""
    out = []
    for lv in raw or []:
        try:
            price, size = float(lv["price"]), float(lv["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if _is_price(price) and math.isfinite(size) and size > 0:
            out.append([price, size])
    out.sort(key=lambda x: x[0], reverse=best_high)
    return out[:BOOK_LEVELS]


def book_snapshot(body: dict[str, Any], question_id: str, now: datetime | None = None) -> Snapshot:
    """``/book`` response for the YES token -> Snapshot (best level first on both sides)."""
    bids = _levels(body.get("bids") or [], best_high=True)
    asks = _levels(body.get("asks") or [], best_high=False)
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2
    else:
        last = _float_or_none(body.get("last_trade_price"))
        mid = (last if last is not None and _is_price(last) else None) or bid or ask or math.nan
    ms = _float_or_none(body.get("timestamp"))
    if ms is not None and math.isfinite(ms) and ms > 0:
        when = datetime.fromtimestamp(ms / 1000, tz=UTC)
    else:
        when = now or datetime.now(UTC)
    return Snapshot(
        question_id=question_id,
        horizon="live",
        as_of=to_iso(when),
        mid=mid,
        bid=bid,
        ask=ask,
        last=_float_or_none(body.get("last_trade_price")),
        price_at=to_iso(when),
        source="live-book",
        book={"bids": bids, "asks": asks, "tick_size": _float_or_none(body.get("tick_size"))},
    )


def _resolve_live(client: HttpClient, ref: Question | str) -> tuple[str, str]:
    """(question id, YES token) from a Question, a token id, or a (prefixed) condition id."""
    if isinstance(ref, Question):
        return ref.id, str(ref.extra["yes_token"])
    s = ref.removeprefix("polymarket:")
    if s.isdigit():
        return f"polymarket:token:{s}", s
    markets = client.get_json(f"{GAMMA}/markets", {"condition_ids": s}, use_cache=False)
    for m in markets or []:
        tokens = yes_no_tokens(m)
        if tokens:
            return f"polymarket:{m.get('conditionId') or s}", tokens[0]
    raise ValueError(f"no Yes/No Polymarket market for {ref!r}")


def live_snapshot(client: HttpClient, question_or_market_id: Question | str) -> Snapshot:
    """Current YES order book (top ``BOOK_LEVELS`` per side) for the prospective collector.

    Accepts a Question, ``"polymarket:<conditionId>"``/a bare condition id, or a YES token id.
    """
    _tune(client)
    qid, token = _resolve_live(client, question_or_market_id)
    body = client.get_json(f"{CLOB}/book", {"token_id": token}, use_cache=False)
    return book_snapshot(body, qid)
