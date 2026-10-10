from __future__ import annotations

import httpx
import pytest

from pricedin.http import HttpClient
from pricedin.news import gdelt
from pricedin.schema import parse_time

ARTICLES = [
    {
        "url": "https://a.test/1",
        "title": "Tankers  halt in Strait of Hormuz",
        "seendate": "20260810T120000Z",
        "domain": "a.test",
        "language": "English",
    },
    {
        "url": "https://a.test/2",
        "title": "Hormuz reopens",
        "seendate": "20260811T060000Z",  # exactly at the first as_of: excluded there
        "domain": "a.test",
        "language": "English",
    },
    {
        "url": "https://b.test/3",
        "title": "Hormuz nachrichten",
        "seendate": "20260810T120000Z",
        "domain": "b.test",
        "language": "German",
    },
    {"url": "https://a.test/1", "title": "dup", "seendate": "20260810T130000Z"},
    {"url": "https://a.test/4", "title": "bad date", "seendate": "yesterday"},
]


def test_build_query_deterministic():
    q = gdelt.build_query("Will the Strait of Hormuz be closed by Iran before September 1?")
    assert q.startswith('("strait of hormuz" OR ')
    assert q.endswith(" sourcelang:english")
    assert q == gdelt.build_query("Will the Strait of Hormuz be closed by Iran before September 1?")
    assert gdelt.build_query("Bitcoin") == "bitcoin sourcelang:english"
    with pytest.raises(ValueError):
        gdelt.build_query("Will it?")


def test_parse_seendate():
    assert gdelt.parse_seendate("20260810T120000Z") == parse_time("2026-08-10T12:00:00Z")


def test_items_for_event_one_request_cut_per_as_of(tmp_path):
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(dict(req.url.params))
        return httpx.Response(200, json={"articles": ARTICLES})

    client = HttpClient(
        transport=httpx.MockTransport(handler),
        host_intervals={"api.gdeltproject.org": 0.0},
        cache_dir=tmp_path,
    )
    t1, t2 = parse_time("2026-08-11T06:00:00Z"), parse_time("2026-08-12T00:00:00Z")
    query, got = gdelt.items_for_event(client, "Strait of Hormuz closure", "", [t2, t1], days=14)
    assert len(calls) == 1
    p = calls[0]
    assert p["mode"] == "artlist" and p["format"] == "json" and p["maxrecords"] == "250"
    assert p["startdatetime"] == "20260728060000" and p["enddatetime"] == "20260812000000"
    assert p["query"] == query
    assert [it.url for it in got[t1]] == ["https://a.test/1"]  # strict < as_of, English only
    assert [it.url for it in got[t2]] == ["https://a.test/1", "https://a.test/2"]
    assert got[t1][0].title == "Tankers halt in Strait of Hormuz"
    assert got[t1][0].seen_at == "2026-08-10T12:00:00Z" and got[t1][0].source == "gdelt"
    # Cached: a re-run makes no request.
    gdelt.items_for_event(client, "Strait of Hormuz closure", "", [t1, t2], days=14)
    assert len(calls) == 1
