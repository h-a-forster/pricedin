"""Record types shared by every stage, and JSONL helpers.

A study is four append-only JSONL tables, each keyed by ``question_id`` (and ``horizon``):

- ``questions.jsonl``  one resolved binary question (``Question``)
- ``snapshots.jsonl``  the market state at each horizon before close (``Snapshot``)
- ``news.jsonl``       headlines published before each snapshot time (``NewsContext``)
- ``forecasts/*.jsonl`` one model probability per question, horizon and condition (``Forecast``)

Times are timezone-aware UTC and serialised as ISO 8601 with a ``Z`` suffix.
Probabilities are floats in [0, 1] for the YES side.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

Venue = Literal["kalshi", "polymarket"]

# Horizon label -> time before the question's close.
HORIZONS: dict[str, timedelta] = {
    "7d": timedelta(days=7),
    "1d": timedelta(days=1),
    "1h": timedelta(hours=1),
}

# Normalised categories. Venue taxonomies are mapped onto these in ``pricedin.venues``.
CATEGORIES = (
    "politics",
    "economics",
    "finance",
    "crypto",
    "tech",
    "science",
    "sports",
    "culture",
    "weather",
    "world",
    "other",
)


def to_iso(t: datetime) -> str:
    if t.tzinfo is None:
        raise ValueError(f"naive datetime: {t!r}")
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str | int | float | datetime) -> datetime:
    """ISO 8601 string (``Z`` or offset) or unix seconds -> aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"naive datetime: {value!r}")
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return t.astimezone(UTC)


@dataclass
class Question:
    id: str  # "kalshi:<market ticker>" or "polymarket:<condition id>"
    venue: Venue
    event_id: str  # cluster for resampling: markets in one event share news and outcomes
    title: str
    rules: str  # resolution criteria as published by the venue
    category: str  # one of CATEGORIES
    created_at: str  # when the question was listed (ISO, UTC)
    open_at: str  # when trading opened
    # Horizons are measured back from close_at: the scheduled end known at listing. It is
    # never the actual close, because an early close (a "by <date>" market resolving YES)
    # depends on the outcome. The actual close is kept in extra["actual_close"].
    close_at: str
    scheduled_close_at: str  # the deadline shown to the model ("" if none was known)
    outcome: int  # 1 = YES, 0 = NO
    volume: float  # venue units: contracts on Kalshi, USDC on Polymarket
    url: str = ""
    venue_category: str = ""  # the venue's own label, kept for audit
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def close(self) -> datetime:
        return parse_time(self.close_at)


@dataclass
class Snapshot:
    question_id: str
    horizon: str  # key of HORIZONS
    as_of: str  # the snapshot time: close_at - HORIZONS[horizon]
    mid: float  # the market probability used for scoring
    bid: float | None = None  # best YES bid, if known
    ask: float | None = None  # best YES ask, if known
    last: float | None = None  # last trade price at or before as_of, if known
    price_at: str | None = None  # time of the observation actually used (<= as_of)
    source: str = ""  # e.g. "kalshi-candles-1m", "polymarket-prices-history", "live-book"
    book: dict[str, Any] | None = None  # {"bids": [[p, size], ...], "asks": [...]} live only


@dataclass
class NewsItem:
    title: str
    seen_at: str  # first time the source saw the article (ISO, UTC); always <= as_of
    domain: str = ""
    url: str = ""
    source: str = ""  # e.g. "gdelt"


@dataclass
class NewsContext:
    question_id: str
    horizon: str
    as_of: str
    query: str
    items: list[NewsItem] = field(default_factory=list)
    source: str = ""
    error: str | None = None


@dataclass
class Forecast:
    question_id: str
    horizon: str
    as_of: str
    forecaster: str  # e.g. "claude-cli:claude-opus-5-5@low"
    condition: str  # e.g. "base" (question only) or "news" (question + dated headlines)
    prob: float | None  # None when the reply could not be parsed or the call failed
    prompt_version: str = ""
    prompt_hash: str = ""  # sha256 prefix of the exact (system, user) prompt; see prompt_hash()
    model: str = ""  # model id the endpoint reported, e.g. a dated snapshot of the alias asked for
    harness: str = ""  # client and version that made the call, e.g. "claude-cli 2.1.287"
    rationale: str = ""  # the model's short reasoning, truncated
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    latency_s: float = 0.0
    error: str | None = None


T = TypeVar("T")

_NESTED: dict[type, dict[str, type]] = {NewsContext: {"items": NewsItem}}


def to_dict(rec: Any) -> dict[str, Any]:
    return asdict(rec)


def from_dict(cls: type[T], d: dict[str, Any]) -> T:
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kwargs = {k: v for k, v in d.items() if k in names}
    for name, sub in _NESTED.get(cls, {}).items():
        if name in kwargs and kwargs[name] is not None:
            kwargs[name] = [from_dict(sub, x) for x in kwargs[name]]
    return cls(**kwargs)


def _open(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return open(path, mode, encoding="utf-8", newline="\n")


def read_jsonl(path: str | Path, cls: type[T] | None = None) -> Iterator[Any]:
    """Yield records from a ``.jsonl`` or ``.jsonl.gz`` file (as ``cls`` if given)."""
    p = Path(path)
    with _open(p, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield from_dict(cls, obj) if cls is not None else obj


def write_jsonl(path: str | Path, records: Iterable[Any], *, append: bool = False) -> int:
    """Write dataclass records or dicts, one JSON object per line. Returns the count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(p, "a" if append else "w") as fh:
        for rec in records:
            obj = rec if isinstance(rec, dict) else to_dict(rec)
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n
