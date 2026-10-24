"""Forecast prompts, built deterministically from a question as of a snapshot time.

The model sees the question, its resolution rules, the venue name, the snapshot time (stated as
the current date) and the scheduled close, plus, in the ``news`` condition, dated headlines
published before the snapshot. It never sees the market price, volume, the actual close time
(an early close would leak the outcome) or anything dated after ``as_of``.

Any change to the wording below must bump ``PROMPT_VERSION``: it is stored with each forecast
and feeds the response cache key through the prompt text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import datetime, timedelta

from pricedin.schema import NewsItem, Question, parse_time

PROMPT_VERSION = "v1"
CONDITIONS = ("base", "news")
PROMPT_HASH_CHARS = 16

MAX_RULES_CHARS = 4000
MAX_HEADLINES = 40

_VENUES = {"kalshi": "Kalshi", "polymarket": "Polymarket"}
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

SYSTEM_TEMPLATE = """\
You are an expert forecaster who works like a superforecaster: you give careful, calibrated \
probabilities for binary questions about future events.

The current date is {now}. Your training data may end well before this date, so events you do \
not know about may have happened since. Reason as of the current date, using the information \
given in the question{news_clause} together with your background knowledge. If any other \
date appears in your context, ignore it: this forecast is made as of the current date above.

How to work:
1. Read the resolution criteria closely. The question resolves exactly as they are worded, \
including edge cases, deadlines and the source used to settle it.
2. Start from a base rate: how often do comparable events happen in a comparable time window?
3. Account for the time remaining{close_clause}. Ask what would have to happen before the \
deadline and how likely it is; over short windows the status quo usually persists.
4. Adjust for the specific evidence, avoiding both overconfidence and needless hedging toward 50%.

Keep your reasoning brief: at most about 150 words. Always give a probability, even when you are \
uncertain or the question is unfamiliar; do not refuse and do not give a range. End your reply \
with one final line in exactly this form:
PROBABILITY: <number between 0 and 1>
"""

_NEWS_CLAUSE = " and the dated headlines provided"

USER_TEMPLATE = """\
Question: {title}
Venue: {venue}
Current date: {now}
{close_line}
Resolution criteria:
{rules}
{news}
What is the probability that this question resolves YES?"""

_NEWS_HEADER = (
    "Recent headlines, oldest first. All were published before the current date; they come "
    "from an automated news search and may be incomplete or irrelevant."
)
_NO_NEWS = "A news search found no headlines for this question before the current date."


def format_time(t: datetime) -> str:
    """``Saturday, 15 August 2026, 14:00 UTC`` (locale-independent)."""
    return (
        f"{_WEEKDAYS[t.weekday()]}, {t.day} {_MONTHS[t.month - 1]} {t.year}, "
        f"{t.hour:02d}:{t.minute:02d} UTC"
    )


def format_duration(d: timedelta) -> str:
    """Largest two units, e.g. ``6 days 23 hours``, ``1 hour``, ``45 minutes``."""
    minutes = max(0, int(d.total_seconds() // 60))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    parts = [(days, "day"), (hours, "hour"), (mins, "minute")]
    out = [f"{n} {unit}{'' if n == 1 else 's'}" for n, unit in parts if n]
    return " ".join(out[:2]) if out else "less than a minute"


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_rules(rules: str, limit: int = MAX_RULES_CHARS) -> str:
    """Normalise line endings and cut very long rules at a word boundary."""
    text = re.sub(r"\n{3,}", "\n\n", rules.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text:
        return "(No resolution criteria were published beyond the question title.)"
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = max(cut.rfind(" "), cut.rfind("\n"))
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " [...] (criteria truncated)"


def select_headlines(
    items: Iterable[NewsItem], as_of: datetime, limit: int = MAX_HEADLINES
) -> list[NewsItem]:
    """Headlines seen strictly before ``as_of``, deduplicated by title, the ``limit`` most
    recent, oldest first. Ties are broken by domain and title so the order is deterministic."""
    kept: list[tuple[datetime, NewsItem]] = []
    for it in items:
        try:
            seen = parse_time(it.seen_at)
        except (TypeError, ValueError):
            continue
        if seen < as_of and _one_line(it.title):
            kept.append((seen, it))
    kept.sort(key=lambda x: (x[0], x[1].domain, _one_line(x[1].title)))
    # Keep the earliest copy of a repeated title (syndicated stories).
    seen_titles: set[str] = set()
    unique: list[NewsItem] = []
    for _, it in kept:
        key = _one_line(it.title).casefold()
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(it)
    return unique[-limit:] if limit > 0 else []


def format_headlines(items: list[NewsItem]) -> str:
    if not items:
        return _NO_NEWS
    lines = [_NEWS_HEADER]
    for it in items:
        day = parse_time(it.seen_at).strftime("%Y-%m-%d")
        src = f" ({it.domain})" if it.domain else ""
        lines.append(f"- {day}{src}: {_one_line(it.title)}")
    return "\n".join(lines)


def close_time(question: Question) -> datetime | None:
    """The scheduled close known at listing, or None.

    Never falls back to ``close_at``: an early actual close would reveal the outcome.
    """
    return parse_time(question.scheduled_close_at) if question.scheduled_close_at else None


def build_prompt(
    question: Question,
    as_of: str | datetime,
    condition: str = "base",
    news: Iterable[NewsItem] | None = None,
) -> tuple[str, str, str]:
    """Return ``(system, user, PROMPT_VERSION)`` for one question at one snapshot time.

    ``condition`` is ``"base"`` (question only) or ``"news"`` (question plus headlines from
    ``news``; ``None`` or empty gives a "no headlines found" note).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    now = parse_time(as_of)
    close = close_time(question)
    if close is not None:
        close_clause = (
            f": the question is scheduled to close on {format_time(close)}, "
            f"{format_duration(close - now)} after the current date"
        )
        close_line = f"Scheduled close: {format_time(close)}\n"
    else:
        close_clause = ": the deadline is set out in the resolution criteria"
        close_line = ""
    system = SYSTEM_TEMPLATE.format(
        now=format_time(now),
        close_clause=close_clause,
        news_clause=_NEWS_CLAUSE if condition == "news" else "",
    )
    news_block = ""
    if condition == "news":
        news_block = "\n" + format_headlines(select_headlines(news or (), now)) + "\n"
    user = USER_TEMPLATE.format(
        title=_one_line(question.title),
        venue=_VENUES.get(question.venue, question.venue),
        now=format_time(now),
        close_line=close_line,
        rules=truncate_rules(question.rules),
        news=news_block,
    )
    return system, user, PROMPT_VERSION


def prompt_hash(system: str, user: str) -> str:
    """Hex sha256 prefix of the exact prompt pair sent to the model.

    ``PROMPT_VERSION`` names the template; this identifies the rendered text, so two rows with the
    same hash saw byte-identical prompts.
    """
    payload = json.dumps([system, user], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:PROMPT_HASH_CHARS]
