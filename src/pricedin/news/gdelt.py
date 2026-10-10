"""GDELT DOC 2.0 headlines, one query per event, cut per snapshot time.

GDELT allows one request per 5 s per IP, so we query once per event over
``[earliest as_of - window, latest as_of]`` and then, for each snapshot, keep only articles with
``seendate`` strictly before that snapshot's ``as_of``. ``seendate`` is when GDELT first saw the
article, which is the leakage-relevant time. The DOC API only searches roughly the last three
months, so old windows come back empty. English sources only.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from pricedin.http import HttpClient
from pricedin.news.relevance import STOPWORDS, proper_phrases, tokenize
from pricedin.schema import NewsItem, parse_time, to_iso

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
SOURCE = "gdelt"
MAX_RECORDS = 250
MAX_TERMS = 4


def _fmt(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def parse_seendate(s: str) -> datetime:
    """``20260810T123000Z`` -> aware UTC datetime."""
    return datetime.strptime(s.strip(), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def build_query(title: str, rules: str = "", max_terms: int = MAX_TERMS) -> str:
    """Deterministic GDELT query: multi-word proper nouns (quoted) first, then the longest
    distinct content words, OR-ed together, English sources only."""
    terms: list[str] = []
    for ph in proper_phrases(title) + proper_phrases(rules[:600]):
        if len(ph) <= 4:
            q = f'"{" ".join(ph)}"'
            if q not in terms:
                terms.append(q)
    covered = {w for t in terms for w in t.strip('"').split()}
    words = sorted(
        {w for w in tokenize(title) if len(w) >= 4 and not w.isdigit() and w not in STOPWORDS},
        key=lambda w: (-len(w), w),
    )
    for w in words:
        if len(terms) >= max_terms:
            break
        if w not in covered:
            terms.append(w)
    terms = terms[:max_terms]
    if not terms:
        raise ValueError(f"no usable query terms in {title!r}")
    core = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    return f"{core} sourcelang:english"


def fetch_articles(
    client: HttpClient,
    query: str,
    start: datetime,
    end: datetime,
    *,
    max_records: int = MAX_RECORDS,
    api_url: str = API_URL,
) -> list[dict[str, Any]]:
    """Raw article dicts (url, title, seendate, domain, language, ...) for ``query``."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(min(max_records, MAX_RECORDS)),
        "sort": "hybridrel",
        "startdatetime": _fmt(start),
        "enddatetime": _fmt(end),
    }
    body = client.get_json(api_url, params)  # cached: the window is fully in the past
    if not isinstance(body, dict):
        raise RuntimeError(f"unexpected GDELT response: {str(body)[:200]}")
    return list(body.get("articles") or [])


def to_items(articles: Iterable[dict[str, Any]]) -> list[NewsItem]:
    """English articles with a title and a parseable ``seendate``; duplicates by URL dropped."""
    out: list[NewsItem] = []
    seen_urls: set[str] = set()
    for a in articles:
        lang = str(a.get("language", "English") or "English")
        title = " ".join(str(a.get("title") or "").split())
        url = str(a.get("url") or "")
        if lang.lower() != "english" or not title or url in seen_urls:
            continue
        try:
            seen = parse_seendate(str(a.get("seendate", "")))
        except ValueError:
            continue
        seen_urls.add(url)
        out.append(
            NewsItem(
                title=title,
                seen_at=to_iso(seen),
                domain=str(a.get("domain") or ""),
                url=url,
                source=SOURCE,
            )
        )
    return out


def items_for_event(
    client: HttpClient,
    title: str,
    rules: str,
    as_ofs: Iterable[datetime],
    *,
    days: int = 14,
) -> tuple[str, dict[datetime, list[NewsItem]]]:
    """One query covering every snapshot of an event; per ``as_of`` only items seen before it."""
    as_ofs = sorted(set(as_ofs))
    query = build_query(title, rules)
    if not as_ofs:
        return query, {}
    items = to_items(fetch_articles(client, query, as_ofs[0] - timedelta(days=days), as_ofs[-1]))
    out = {}
    for t in as_ofs:
        lo = t - timedelta(days=days)
        out[t] = [it for it in items if lo <= parse_time(it.seen_at) < t]
    return query, out
