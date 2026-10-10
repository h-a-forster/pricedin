from __future__ import annotations

from pricedin.news.relevance import (
    bm25_scores,
    proper_phrases,
    question_query,
    rank,
    rank_news,
    tokenize,
)
from pricedin.schema import NewsItem


def item(title: str, seen: str = "2026-08-10T00:00:00Z") -> NewsItem:
    return NewsItem(title=title, seen_at=seen)


def test_tokenize_normalises():
    assert tokenize("The Fed's RATES, cuts & policies!") == ["fed", "rate", "cut", "policy"]
    assert tokenize("Will the market resolve Yes?") == []


def test_proper_phrases():
    ph = proper_phrases("Will the Strait of Hormuz reopen before the Federal Reserve meets?")
    assert ("strait", "of", "hormuz") in ph
    assert ("federal", "reserve") in ph


def test_bm25_prefers_relevant_and_rare_terms():
    q, _ = question_query("Federal Reserve rate cut")
    docs = [
        "Federal Reserve cuts interest rate by 25 basis points",
        "Earthquake in Colombia",
        "Rate of inflation rises",
    ]
    s = bm25_scores(q, docs)
    assert s[0] > s[2] > s[1] == 0.0


def test_phrase_bonus():
    q, ph = question_query("Will the Strait of Hormuz reopen?")
    a, b = bm25_scores(
        q, ["Iran closes the Strait of Hormuz to tankers", "Hormuz strait, a long note"], phrases=ph
    )
    assert a > b


def test_rank_top_k_positive_and_chronological():
    items = [
        item("Fed holds rates steady", "2026-08-05T00:00:00Z"),
        item("Federal Reserve chair speaks on rates", "2026-08-01T00:00:00Z"),
        item("Football results", "2026-08-02T00:00:00Z"),
        item("Federal Reserve raises rates; Federal Reserve statement", "2026-08-09T00:00:00Z"),
    ]
    got = rank_news("Will the Federal Reserve cut rates in September?", "", items, k=2)
    assert [g.seen_at for g in got] == sorted(g.seen_at for g in got)
    assert all("Federal Reserve" in g.title for g in got)
    # Without the coverage rule, "Fed holds rates" still fails the subject rule (no
    # "Federal"/"Reserve"); only "Football results" scored 0 before either rule.
    assert len(rank("Federal Reserve rates", items, k=10, min_coverage=0.0)) == 2
    assert len(rank_news("Federal Reserve rates", "", items, k=10)) == 2  # "Fed ... rates" weak
    assert rank_news("Federal Reserve", "", items, k=0) == []


def test_rank_deterministic_tie_break():
    items = [
        item("Hormuz news b", "2026-08-01T00:00:00Z"),
        item("Hormuz news a", "2026-08-03T00:00:00Z"),
    ]
    items += [item("Hormuz news c", "2026-08-02T00:00:00Z")]
    a = rank("Hormuz", items, k=2)
    b = rank("Hormuz", list(reversed(items)), k=2)
    assert a == b
    # Equal scores: newest kept first.
    assert [x.title for x in a] == ["Hormuz news c", "Hormuz news a"]


def test_rules_contribute():
    # Rules terms re-order items that match the title; rules-only matches are too weak.
    items = [item("Oil prices fall in Asia"), item("Oil prices: Brent crude jumps"), item("Brent")]
    rules = "Resolves Yes if Brent crude closes above $100."
    got = rank("Will oil prices spike?", items, rules=rules, k=1)
    assert [g.title for g in got] == ["Oil prices: Brent crude jumps"]
    assert len(rank("Will oil prices spike?", items, rules=rules)) == 2


def test_dates_do_not_match_and_weak_matches_are_dropped():
    assert tokenize("[2026-08-10] On 3 Aug 2026") == []
    items = [
        item("[2026-07-29] Russia's Federal Security Service issues a warrant"),
        item("[2026-07-29] Federal Reserve holds interest rates at its meeting"),
        item("[2026-07-30] Sports > 2026 FIFA World Cup > Argentina wins"),
    ]
    got = rank_news(
        "Will the Federal Reserve cut interest rates at its 2026 FOMC meeting?", "", items
    )
    assert [g.title for g in got] == [
        "[2026-07-29] Federal Reserve holds interest rates at its meeting"
    ]
    # With the coverage filter off, the subject rule still drops the weak "Federal" match:
    # "Reserve" is the rarer, heavier entity term.
    assert len(rank("Will the Federal Reserve cut interest rates?", items, min_coverage=0.0)) == 1


def test_items_must_name_the_subject() -> None:
    from pricedin.news.relevance import entity_terms, rank

    assert entity_terms("Will Venezuela win on 2026-09-28?") == {"venezuela"}
    items = [
        NewsItem(title="In tennis, Zverev wins the US Open final", seen_at="2026-09-13T00:00:00Z"),
        NewsItem(
            title="Venezuela beats Peru in a World Cup qualifier", seen_at="2026-09-12T00:00:00Z"
        ),
    ]
    out = rank("Will Venezuela win on 2026-09-28?", items)
    assert [it.title for it in out] == [items[1].title]
