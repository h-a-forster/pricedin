"""Wikipedia "Portal:Current events" daily pages, read as they stood at a past moment.

Each day has a page (``Portal:Current events/2026 August 10``) of one-line news bullets grouped
by section and topic. Pages keep being edited for weeks, so for a snapshot time ``as_of`` we use
the last revision whose timestamp is ``<= as_of``: everything in it was provably public by then.
A page with no such revision did not exist yet and contributes nothing.

Plan per run: (1) fetch each needed page's revision list (ids + timestamps), (2) pick the
revision per (page, as_of), (3) fetch the distinct revision texts in batches of 50 and cache them
by revid (immutable), (4) parse bullets into ``NewsItem``s whose ``seen_at`` is that revision's
timestamp. Revision lists change as pages are edited, so a cached list is reused only for
``as_of`` values well before the moment it was fetched.
"""

from __future__ import annotations

import bisect
import hashlib
import html
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pricedin.http import HttpClient, atomic_write_json
from pricedin.schema import NewsItem, parse_time, to_iso

API_URL = "https://en.wikipedia.org/w/api.php"
DOMAIN = "en.wikipedia.org"
SOURCE = "wikipedia-current-events"
MAX_REVIDS_PER_REQUEST = 50  # MediaWiki limit for prop=revisions&revids= (non-bot users)
MAX_TITLE_CHARS = 400
# A cached revision list is trusted for as_of up to fetched_at minus this (replication lag).
REVLIST_SAFETY = timedelta(minutes=30)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December")  # fmt: skip


@dataclass(frozen=True, order=True)
class Revision:
    timestamp: datetime
    revid: int


@dataclass(frozen=True)
class Bullet:
    path: tuple[str, ...]  # section heading, then parent topic bullets
    text: str


def page_title(day: date) -> str:
    return f"Portal:Current events/{day.year} {_MONTHS[day.month - 1]} {day.day}"


def window_days(as_of: datetime, days: int = 14) -> list[date]:
    """The daily pages that can hold news public by ``as_of``: from ``as_of - days`` to as_of."""
    end = as_of.astimezone(UTC).date()
    start = (as_of - timedelta(days=days)).astimezone(UTC).date()
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def select_revision(revisions: Iterable[Revision], as_of: datetime) -> Revision | None:
    """The latest revision with ``timestamp <= as_of`` (highest revid on a tie), or None."""
    revs = sorted(revisions)
    i = bisect.bisect_right(revs, Revision(as_of, 2**62))
    return revs[i - 1] if i else None


# -- wikitext -------------------------------------------------------------------------------

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref\b[^>/]*/>|<ref\b[^>]*>.*?</ref>", re.S | re.I)
_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")
_FILE_LINK = re.compile(r"\[\[(?:File|Image):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
_WIKILINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_EXTLINK = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
_BARE_URL = re.compile(r"https?://\S+")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_HEADING = re.compile(r"^\s*(?:'''(.+?)'''\s*|;\s*(.+?)\s*)$")
_BULLET = re.compile(r"^(\*+)\s*(.*)$")


def _template(body: str) -> str:
    parts = [p.strip() for p in body.split("|")]
    name = parts[0].lower().replace("_", " ")
    args = [p for p in parts[1:] if "=" not in p]
    if not args:
        return ""
    if name in ("ill", "interlanguage link", "nowrap", "nobr", "small", "abbr", "w", "sic"):
        return args[0]
    if name == "lang" and len(args) >= 2:
        return args[1]
    if name in ("cvt", "convert") and len(args) >= 2:
        return f"{args[0]} {args[1]}"
    if name == "m" and len(args) >= 2:  # {{M|w|7.4}} renders "Mw 7.4"
        return f"M{args[0]} {args[1]}"
    if name in ("usd", "us$"):
        return f"US${args[0]}"
    return ""


def strip_markup(text: str) -> str:
    """Wikitext -> plain text: links to their labels, external links to their source name,
    refs, comments, unknown templates and HTML tags removed."""
    s = _COMMENT.sub("", text)
    s = _REF.sub("", s)
    for _ in range(10):  # innermost templates first, until none are left
        s, n = _TEMPLATE.subn(lambda m: _template(m.group(1)), s)
        if not n:
            break
    s = _FILE_LINK.sub("", s)
    s = _WIKILINK.sub(lambda m: (m.group(2) or m.group(1).split("#")[0]).lstrip(":"), s)
    s = _EXTLINK.sub(lambda m: m.group(1) or "", s)
    s = _BARE_URL.sub("", s)
    s = _TAG.sub("", s)
    s = s.replace("'''", "").replace("''", "")
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def parse_bullets(wikitext: str) -> list[Bullet]:
    """News bullets of a daily page with their topic path. A bullet that has deeper bullets
    under it is a topic heading, not an item."""
    text = _REF.sub("", _COMMENT.sub("", wikitext))
    rows: list[tuple[str, int, str]] = []  # (section, depth, cleaned text)
    section = ""
    for line in text.splitlines():
        if line.lstrip().startswith("{{Current events"):
            continue
        m = _BULLET.match(line)
        if m:
            cleaned = strip_markup(m.group(2))
            if cleaned:
                rows.append((section, len(m.group(1)), cleaned))
            continue
        h = _HEADING.match(line)
        if h:
            section = strip_markup(h.group(1) or h.group(2))
    out: list[Bullet] = []
    stack: list[str] = []
    for i, (sec, depth, cleaned) in enumerate(rows):
        stack = stack[: depth - 1]
        stack += [""] * (depth - 1 - len(stack))  # tolerate skipped levels
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt is not None and nxt[0] == sec and nxt[1] > depth:
            stack.append(cleaned)
            continue
        path = tuple(p for p in (sec, *stack) if p)
        out.append(Bullet(path, cleaned))
        stack.append(cleaned)
    return out


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    cut = s[: limit - 3]
    space = cut.rfind(" ")
    if space > limit * 0.7:
        cut = cut[:space]
    return cut.rstrip(" ,;>") + "..."


def bullet_title(day: date, bullet: Bullet, limit: int = MAX_TITLE_CHARS) -> str:
    """``[2026-08-10] Section > Topic > ... > text``; the topic path is shortened first."""
    prefix = f"[{day.isoformat()}] "
    path = " > ".join(bullet.path)
    room = limit - len(prefix) - len(bullet.text) - 3
    if path and room < len(path):
        path = _truncate(path, max(room, 60))
    return _truncate(prefix + (f"{path} > " if path else "") + bullet.text, limit)


# -- API ------------------------------------------------------------------------------------


class CurrentEvents:
    """Reads daily Current-events pages as of past timestamps, with an on-disk cache."""

    def __init__(
        self,
        client: HttpClient,
        cache_dir: str | Path | None = None,
        *,
        api_url: str = API_URL,
        now: datetime | None = None,
    ):
        self.client = client
        self.api_url = api_url
        self.dir = Path(cache_dir) / "wikipedia" if cache_dir is not None else None
        self._now = now  # fixed "now" for tests
        self._revlists: dict[str, tuple[datetime, list[Revision]]] = {}
        self._texts: dict[int, str] = {}
        self._parsed: dict[int, list[Bullet]] = {}

    def _query(self, params: dict[str, Any]) -> dict[str, Any]:
        base = {"action": "query", "format": "json", "formatversion": "2"}
        body = self.client.get_json(self.api_url, {**base, **params}, use_cache=False)
        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"MediaWiki error {err.get('code')}: {err.get('info')}")
        return body

    # revision lists ------------------------------------------------------------------------
    def _revlist_path(self, title: str) -> Path | None:
        if self.dir is None:
            return None
        h = hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
        return self.dir / "revlists" / f"{h}.json"

    def revisions(self, title: str, needed_as_of: datetime) -> list[Revision]:
        """All revisions of ``title`` (empty if the page is missing), from a cache that was
        fetched late enough to be complete up to ``needed_as_of``."""
        hit = self._revlists.get(title)
        if hit is None and (path := self._revlist_path(title)) is not None and path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                revs = [Revision(parse_time(t), int(r)) for r, t in d["revisions"]]
                hit = (parse_time(d["fetched_at"]), revs)
            except (OSError, ValueError, KeyError):
                hit = None
        if hit is not None and needed_as_of <= hit[0] - REVLIST_SAFETY:
            self._revlists[title] = hit
            return hit[1]

        fetched_at = self._now or datetime.now(UTC)
        revs: list[Revision] = []
        cont: dict[str, Any] = {}
        while True:
            body = self._query(
                {
                    "prop": "revisions",
                    "titles": title,
                    "rvprop": "ids|timestamp",
                    "rvlimit": "max",
                    **cont,
                }
            )
            for page in body.get("query", {}).get("pages", []):
                for r in page.get("revisions", []):
                    revs.append(Revision(parse_time(r["timestamp"]), int(r["revid"])))
            if "continue" not in body:
                break
            cont = body["continue"]
        revs.sort()
        self._revlists[title] = (fetched_at, revs)
        if (path := self._revlist_path(title)) is not None:
            atomic_write_json(
                path,
                {
                    "title": title,
                    "fetched_at": to_iso(fetched_at),
                    "revisions": [[r.revid, to_iso(r.timestamp)] for r in revs],
                },
            )
        return revs

    # revision texts ------------------------------------------------------------------------
    def _text_path(self, revid: int) -> Path | None:
        return None if self.dir is None else self.dir / "revs" / f"{revid}.json"

    def texts(self, revids: Iterable[int]) -> dict[int, str]:
        """Wikitext of each revision (``""`` if hidden or deleted), batched and cached."""
        want = sorted(set(revids))
        out: dict[int, str] = {}
        missing: list[int] = []
        for rid in want:
            if rid in self._texts:
                out[rid] = self._texts[rid]
                continue
            path = self._text_path(rid)
            if path is not None and path.exists():
                try:
                    out[rid] = self._texts[rid] = json.loads(path.read_text(encoding="utf-8"))[
                        "content"
                    ]
                    continue
                except (OSError, ValueError, KeyError):
                    pass
            missing.append(rid)
        for i in range(0, len(missing), MAX_REVIDS_PER_REQUEST):
            batch = missing[i : i + MAX_REVIDS_PER_REQUEST]
            got: dict[int, tuple[str, str, str]] = {}
            cont: dict[str, Any] = {}
            while True:  # big batches can come back in several parts
                body = self._query(
                    {
                        "prop": "revisions",
                        "revids": "|".join(map(str, batch)),
                        "rvprop": "ids|timestamp|content",
                        "rvslots": "main",
                        **cont,
                    }
                )
                for page in body.get("query", {}).get("pages", []):
                    for r in page.get("revisions", []):
                        content = r.get("slots", {}).get("main", {}).get("content")
                        if content is not None or int(r["revid"]) not in got:
                            got[int(r["revid"])] = (page.get("title", ""), r["timestamp"], content)
                if "continue" not in body:
                    break
                cont = body["continue"]
            for rid in batch:
                title, ts, content = got.get(rid, ("", "", None))
                text = content or ""
                out[rid] = self._texts[rid] = text
                if content is not None and (path := self._text_path(rid)) is not None:
                    atomic_write_json(
                        path, {"revid": rid, "title": title, "timestamp": ts, "content": text}
                    )
        return out

    def _bullets(self, revid: int, text: str) -> list[Bullet]:
        if revid not in self._parsed:
            self._parsed[revid] = parse_bullets(text)
        return self._parsed[revid]

    # public --------------------------------------------------------------------------------
    def items_as_of(
        self, as_ofs: Iterable[datetime], *, days: int = 14
    ) -> dict[datetime, list[NewsItem]]:
        """For each ``as_of``, items from the daily pages in its window, each page read at its
        last revision ``<= as_of``. One revision-list request per page, then batched texts."""
        as_ofs = sorted(set(as_ofs))
        pages: dict[date, datetime] = {}  # day -> latest as_of that needs it
        for t in as_ofs:
            for d in window_days(t, days):
                pages[d] = max(pages.get(d, t), t)
        chosen: dict[tuple[datetime, date], Revision] = {}
        for d in sorted(pages):
            revs = self.revisions(page_title(d), pages[d])
            for t in as_ofs:
                if d in window_days(t, days) and (rev := select_revision(revs, t)) is not None:
                    chosen[(t, d)] = rev
        texts = self.texts(r.revid for r in chosen.values())

        out: dict[datetime, list[NewsItem]] = {t: [] for t in as_ofs}
        for (t, d), rev in sorted(chosen.items()):
            if rev.timestamp > t:  # cannot happen by construction; never let it through
                raise AssertionError(f"revision {rev.revid} at {rev.timestamp} is after {t}")
            seen = to_iso(rev.timestamp)
            url = f"https://{DOMAIN}/w/index.php?oldid={rev.revid}"
            for b in self._bullets(rev.revid, texts.get(rev.revid, "")):
                out[t].append(
                    NewsItem(
                        title=bullet_title(d, b),
                        seen_at=seen,
                        domain=DOMAIN,
                        url=url,
                        source=SOURCE,
                    )
                )
        return out
