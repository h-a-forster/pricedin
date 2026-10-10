from __future__ import annotations

import pytest

from pricedin.http import HttpClient
from pricedin.news import collect as collect_mod
from pricedin.news.collect import LeakageError, check_no_leakage, collect_news
from pricedin.schema import NewsContext, NewsItem, Question, Snapshot, read_jsonl


def question(qid: str, title: str, event: str = "e1") -> Question:
    return Question(
        id=qid,
        venue="kalshi",
        event_id=event,
        title=title,
        rules="",
        category="world",
        created_at="2026-08-01T00:00:00Z",
        open_at="2026-08-01T00:00:00Z",
        close_at="2026-08-12T06:00:00Z",
        scheduled_close_at="2026-08-12T06:00:00Z",
        outcome=1,
        volume=1.0,
    )


def snap(qid: str, horizon: str, as_of: str) -> Snapshot:
    return Snapshot(question_id=qid, horizon=horizon, as_of=as_of, mid=0.5)


class FakeCE:
    """Stands in for CurrentEvents; ``leak`` injects an item dated after as_of."""

    calls = 0
    leak = False
    fail = False

    def __init__(self, client, cache_dir):
        pass

    def items_as_of(self, as_ofs, *, days=14):
        FakeCE.calls += 1
        if FakeCE.fail:
            raise RuntimeError("wiki down")
        out = {}
        for t in as_ofs:
            seen = t.strftime("%Y-%m-%dT%H:%M:%SZ")
            items = [
                NewsItem(title="Strait of Hormuz tanker traffic resumes", seen_at=seen),
                NewsItem(title="Federal Reserve holds rates", seen_at="2026-08-01T00:00:00Z"),
                NewsItem(title="Cricket scores", seen_at="2026-08-01T00:00:00Z"),
            ]
            if FakeCE.leak:
                items.append(
                    NewsItem(title="Strait of Hormuz future", seen_at="2026-12-31T00:00:00Z")
                )
            out[t] = items
        return out


@pytest.fixture
def fake_wiki(monkeypatch):
    FakeCE.calls, FakeCE.leak, FakeCE.fail = 0, False, False
    monkeypatch.setattr(collect_mod, "CurrentEvents", FakeCE)
    return FakeCE


QS = [
    question("q1", "Will the Strait of Hormuz reopen?"),
    question("q2", "Will the Federal Reserve cut rates?", event="e2"),
]
SNAPS = [
    snap("q1", "1d", "2026-08-11T06:00:00Z"),
    snap("q1", "1h", "2026-08-12T05:00:00Z"),
    snap("q2", "1d", "2026-08-11T06:00:00Z"),
    snap("qX", "1d", "2026-08-11T06:00:00Z"),
]


def test_collect_writes_and_resumes(tmp_path, fake_wiki):
    out = tmp_path / "news.jsonl"
    s = collect_news(QS, SNAPS, out_path=out, cache_dir=tmp_path / "c", k=5, client=HttpClient())
    assert s["rows_written"] == 3 and s["skipped_unknown_question"] == 1
    rows = list(read_jsonl(out, NewsContext))
    assert {(r.question_id, r.horizon) for r in rows} == {("q1", "1d"), ("q1", "1h"), ("q2", "1d")}
    q1 = next(r for r in rows if r.question_id == "q1" and r.horizon == "1d")
    assert [it.title for it in q1.items] == ["Strait of Hormuz tanker traffic resumes"]
    q2 = next(r for r in rows if r.question_id == "q2")
    assert [it.title for it in q2.items] == ["Federal Reserve holds rates"]
    assert all(r.error is None and r.source == "wikipedia" for r in rows)
    assert fake_wiki.calls == 2  # once per event

    s2 = collect_news(QS, SNAPS, out_path=out, cache_dir=tmp_path / "c", client=HttpClient())
    assert s2["rows_written"] == 0 and s2["skipped_existing"] == 3
    assert len(list(read_jsonl(out))) == 3


def test_source_errors_are_recorded_and_retryable(tmp_path, fake_wiki):
    out = tmp_path / "news.jsonl"
    fake_wiki.fail = True
    s = collect_news(QS, SNAPS[:1], out_path=out, cache_dir=None, client=HttpClient())
    assert s["rows_with_error"] == 1
    (row,) = read_jsonl(out, NewsContext)
    assert row.items == [] and "wiki down" in row.error

    fake_wiki.fail = False
    s = collect_news(
        QS, SNAPS[:1], out_path=out, cache_dir=None, client=HttpClient(), retry_errors=True
    )
    assert s["rows_written"] == 1
    (row,) = read_jsonl(out, NewsContext)
    assert row.error is None and row.items


def test_leakage_guard_blocks_write(tmp_path, fake_wiki):
    fake_wiki.leak = True
    out = tmp_path / "news.jsonl"
    with pytest.raises(LeakageError, match="after as_of"):
        collect_news(QS[:1], SNAPS[:1], out_path=out, cache_dir=None, client=HttpClient())
    assert not out.exists() or not list(read_jsonl(out))


def test_check_no_leakage_boundary():
    ok = NewsContext("q", "1d", "2026-08-11T06:00:00Z", "", [NewsItem("a", "2026-08-11T06:00:00Z")])
    check_no_leakage(ok)  # equal is allowed
    bad = NewsContext(
        "q", "1d", "2026-08-11T06:00:00Z", "", [NewsItem("a", "2026-08-11T06:00:01Z")]
    )
    with pytest.raises(LeakageError):
        check_no_leakage(bad)
    with pytest.raises(LeakageError):
        check_no_leakage(NewsContext("q", "1d", "2026-08-11T06:00:00Z", "", [NewsItem("a", "?")]))


def test_unknown_source_rejected(tmp_path):
    with pytest.raises(ValueError):
        collect_news([], [], sources=("bing",), out_path=tmp_path / "n.jsonl", cache_dir=None)
