from __future__ import annotations

from datetime import UTC, date, datetime

import httpx

from pricedin.http import HttpClient
from pricedin.news.wikipedia import (
    CurrentEvents as CE,
)
from pricedin.news.wikipedia import (
    Revision,
    bullet_title,
    page_title,
    parse_bullets,
    select_revision,
    strip_markup,
    window_days,
)
from pricedin.schema import parse_time

FIXTURE = """{{Current events|year=2026|month=08|day=10|top=yes}}
<!-- All news items below this line -->
'''Armed conflicts and attacks'''
*[[2026 Iran war]]
**[[2026 Strait of Hormuz crisis]]
***[[2026 United States naval blockade of Iran|US blockade]]
****The [[United States Central Command]] says it has redirected 55 vessels linked to \
[[Iran]].<ref>{{cite web|url=https://x.test|title=Cite}}</ref> \
[https://www.middleeasteye.net/live (''Middle East Eye'')]
**{{ill|Fawzi al-Mansouri|ar|x}} is [[Assassination|assassinated]] in [[Benghazi]]. \
[https://www.aljazeera.com/a (Al Jazeera)]
'''Disasters and accidents'''
*A {{M|w|7.4}} earthquake strikes [[Chocó Department]], [[Colombia]], burning \
{{cvt|550|ha|acres}}.<ref name="r1" /> [https://www.reuters.com/x (Reuters)] <!-- note -->
'''Business and economy'''
*[[Federal Reserve]]
**The [[Federal Reserve|Fed]] holds its &quot;key&quot; rate at <small>4.25%</small>. \
[https://www.bbc.com/news/x BBC News]
<!-- All news items above this line -->
{{Current events|year=2026|month=08|day=10|bottom=yes}}
"""


def dt(s: str) -> datetime:
    return parse_time(s)


def test_strip_markup():
    assert strip_markup("[[a|b]] and [[c]] [[d#e|f]]") == "b and c f"
    assert strip_markup("x<ref>gone</ref> [https://u.test (''Src'')]") == "x (Src)"
    assert strip_markup("{{ill|Name|ar|y}} {{unknown|z}} '''bold'''") == "Name bold"
    assert strip_markup("{{M|w|7.4}} &amp; [https://u.test]") == "Mw 7.4 &"


def test_parse_bullets_nested_paths():
    bs = parse_bullets(FIXTURE)
    assert [b.path for b in bs] == [
        (
            "Armed conflicts and attacks",
            "2026 Iran war",
            "2026 Strait of Hormuz crisis",
            "US blockade",
        ),
        ("Armed conflicts and attacks", "2026 Iran war"),
        ("Disasters and accidents",),
        ("Business and economy", "Federal Reserve"),
    ]
    texts = [b.text for b in bs]
    assert texts[0] == (
        "The United States Central Command says it has redirected 55 vessels linked to Iran. "
        "(Middle East Eye)"
    )
    assert texts[1] == "Fawzi al-Mansouri is assassinated in Benghazi. (Al Jazeera)"
    assert (
        texts[2]
        == "A Mw 7.4 earthquake strikes Chocó Department, Colombia, burning 550 ha. (Reuters)"
    )
    assert texts[3] == 'The Fed holds its "key" rate at 4.25%. BBC News'
    assert not any("cite" in t.lower() or "http" in t or "{{" in t for t in texts)


def test_bullet_title_prefix_and_truncation():
    b = parse_bullets(FIXTURE)[0]
    t = bullet_title(date(2026, 8, 10), b)
    assert t.startswith("[2026-08-10] Armed conflicts and attacks > 2026 Iran war > ")
    assert t.endswith("(Middle East Eye)")
    long = bullet_title(date(2026, 8, 10), b, limit=120)
    assert len(long) <= 120 and long.endswith("...")


def test_page_title_and_window():
    assert page_title(date(2026, 8, 1)) == "Portal:Current events/2026 August 1"
    days = window_days(dt("2026-08-11T06:00:00Z"), 14)
    assert days[0] == date(2026, 7, 28) and days[-1] == date(2026, 8, 11) and len(days) == 15


def test_select_revision_edges():
    revs = [
        Revision(dt("2026-08-10T01:00:00Z"), 1),
        Revision(dt("2026-08-10T05:00:00Z"), 2),
        Revision(dt("2026-08-10T05:00:00Z"), 3),
        Revision(dt("2026-08-11T06:00:00Z"), 4),
        Revision(dt("2026-08-11T06:00:01Z"), 5),
    ]
    assert select_revision(revs, dt("2026-08-11T06:00:00Z")).revid == 4  # exactly at as_of
    assert select_revision(revs, dt("2026-08-11T05:59:59Z")).revid == 3  # tie -> higher revid
    assert select_revision(reversed(revs), dt("2026-08-10T04:00:00Z")).revid == 1
    assert select_revision(revs, dt("2026-08-10T00:59:59Z")) is None  # page not created yet
    assert select_revision([], dt("2026-08-10T00:00:00Z")) is None


# -- fake MediaWiki API -----------------------------------------------------------------------

AUG10 = page_title(date(2026, 8, 10))
AUG11 = page_title(date(2026, 8, 11))
AUG12 = page_title(date(2026, 8, 12))
REVS = {
    AUG10: [
        (101, "2026-08-10T02:00:00Z"),
        (102, "2026-08-11T06:00:00Z"),
        (103, "2026-08-12T09:00:00Z"),
    ],
    AUG11: [(201, "2026-08-11T07:00:00Z")],  # created after the first as_of
    AUG12: [(301, "2026-08-12T01:00:00Z")],
}
TEXTS = {
    101: "'''Business'''\n*The [[Federal Reserve]] meets.",
    102: "'''Business'''\n*The [[Federal Reserve]] meets.\n*[[Strait of Hormuz]] reopens.",
    103: "'''Business'''\n*LATER EDIT that must not leak.",
    201: "'''World'''\n*Day eleven news.",
    301: "'''World'''\n*Day twelve news.",
}


class FakeWiki:
    def __init__(self, page_size: int = 500):
        self.calls: list[dict[str, str]] = []
        self.page_size = page_size

    def __call__(self, req: httpx.Request) -> httpx.Response:
        p = dict(req.url.params)
        self.calls.append(p)
        if "titles" in p:
            title = p["titles"]
            revs = sorted(REVS.get(title, []), key=lambda r: -r[0])
            start = int(p.get("rvcontinue", 0))
            chunk = revs[start : start + self.page_size]
            page = {"title": title, "ns": 100}
            if title in REVS:
                page["revisions"] = [{"revid": r, "timestamp": t} for r, t in chunk]
            else:
                page["missing"] = True
            body = {"query": {"pages": [page]}}
            if start + self.page_size < len(revs):
                body["continue"] = {"rvcontinue": str(start + self.page_size), "continue": "||"}
            return httpx.Response(200, json=body)
        ids = [int(x) for x in p["revids"].split("|")]
        assert len(ids) <= 50
        pages = {}
        for rid in ids:
            title = next(t for t, rs in REVS.items() if any(r == rid for r, _ in rs))
            ts = dict(REVS[title])[rid]
            pages.setdefault(title, []).append(
                {"revid": rid, "timestamp": ts, "slots": {"main": {"content": TEXTS[rid]}}}
            )
        body = {"query": {"pages": [{"title": t, "revisions": r} for t, r in pages.items()]}}
        return httpx.Response(200, json=body)


def client_for(fake) -> HttpClient:
    return HttpClient(
        transport=httpx.MockTransport(fake),
        min_interval=0.0,
        host_intervals={"en.wikipedia.org": 0.0},
    )


NOW = dt("2026-10-01T00:00:00Z")


def test_items_as_of_multiple_pages_and_times(tmp_path):
    fake = FakeWiki(page_size=2)  # forces rvcontinue pagination
    ce = CE(client_for(fake), tmp_path, now=NOW)
    t1, t2 = dt("2026-08-11T06:00:00Z"), dt("2026-08-12T05:00:00Z")
    got = ce.items_as_of([t1, t2], days=2)

    titles1 = [it.title for it in got[t1]]
    assert titles1 == [
        "[2026-08-10] Business > The Federal Reserve meets.",
        "[2026-08-10] Business > Strait of Hormuz reopens.",
    ]  # rev 102 sits exactly at t1; Aug 11 page did not exist yet
    assert {it.seen_at for it in got[t1]} == {"2026-08-11T06:00:00Z"}
    assert got[t1][0].url.endswith("oldid=102")
    assert got[t1][0].source == "wikipedia-current-events"

    titles2 = [it.title for it in got[t2]]
    assert "[2026-08-11] World > Day eleven news." in titles2
    assert "[2026-08-12] World > Day twelve news." in titles2
    assert not any("LATER EDIT" in t for t in titles2)  # rev 103 is after t2
    for t, items in got.items():
        assert all(parse_time(it.seen_at) <= t for it in items)

    rev_calls = [c for c in fake.calls if "revids" in c]
    assert len(rev_calls) == 1  # all distinct revisions in one batch
    assert sorted(map(int, rev_calls[0]["revids"].split("|"))) == [102, 201, 301]

    # Second run from the disk cache: no requests at all.
    fake2 = FakeWiki()
    ce2 = CE(client_for(fake2), tmp_path, now=NOW)
    assert ce2.items_as_of([t1, t2], days=2) == got
    assert fake2.calls == []


def test_missing_page_and_stale_revlist_cache(tmp_path):
    t = dt("2026-08-13T00:00:00Z")
    fake = FakeWiki()
    # A revision list fetched *before* as_of must not be trusted (it could miss later edits).
    ce = CE(client_for(fake), tmp_path, now=dt("2026-08-12T00:00:00Z"))
    ce.revisions(AUG12, t)
    n = len(fake.calls)
    ce2 = CE(client_for(fake), tmp_path, now=NOW)
    ce2.revisions(AUG12, t)
    assert len(fake.calls) == n + 1

    missing = page_title(date(2026, 8, 13))
    assert ce2.revisions(missing, t) == []
    got = ce2.items_as_of([t], days=0)
    assert got == {t: []}


def test_items_as_of_is_utc_aware():
    t = datetime(2026, 8, 11, 6, tzinfo=UTC)
    fake = FakeWiki()
    got = CE(client_for(fake), None, now=NOW).items_as_of([t], days=1)
    assert len(got[t]) == 2
