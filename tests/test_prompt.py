from __future__ import annotations

from datetime import timedelta

import pytest

from pricedin.forecast.prompt import (
    MAX_HEADLINES,
    PROMPT_VERSION,
    build_prompt,
    format_duration,
    select_headlines,
    truncate_rules,
)
from pricedin.schema import NewsItem, Question, parse_time


def make_question(**kw) -> Question:
    base = dict(
        id="kalshi:TEST-1",
        venue="kalshi",
        event_id="TEST",
        title="Will the test pass?",
        rules="Resolves YES if the test passes before close.",
        category="tech",
        created_at="2026-08-01T00:00:00Z",
        open_at="2026-08-01T00:00:00Z",
        close_at="2026-08-20T12:00:00Z",
        scheduled_close_at="2026-08-22T12:00:00Z",
        outcome=1,
        volume=12345.0,
    )
    base.update(kw)
    return Question(**base)


AS_OF = "2026-08-15T12:00:00Z"


def test_base_prompt_content() -> None:
    q = make_question(extra={"last_price": 0.42})
    system, user, version = build_prompt(q, AS_OF, "base")
    assert version == PROMPT_VERSION
    assert "The current date is Saturday, 15 August 2026, 12:00 UTC." in system
    assert "PROBABILITY: <number between 0 and 1>" in system
    # Scheduled close, not the (possibly early) actual close.
    assert "Saturday, 22 August 2026, 12:00 UTC" in system
    assert "7 days after the current date" in system
    assert "20 August" not in system + user
    assert "Will the test pass?" in user and "Kalshi" in user
    assert "Resolves YES if the test passes" in user
    for leak in ("12345", "0.42", "price", "volume", "headline"):
        assert leak not in (system + user).lower()


def test_prompt_is_deterministic() -> None:
    q = make_question()
    items = [NewsItem(title=f"t{i}", seen_at="2026-08-10T00:00:00Z") for i in range(5)]
    a = build_prompt(q, AS_OF, "news", items)
    b = build_prompt(q, parse_time(AS_OF), "news", list(reversed(items)))
    assert a == b


def test_news_condition_filters_orders_and_caps() -> None:
    q = make_question()
    items = [
        NewsItem(title="Later story", seen_at="2026-08-14T09:00:00Z", domain="b.com"),
        NewsItem(title="Earlier story", seen_at="2026-08-02T09:00:00Z", domain="a.com"),
        NewsItem(title="From the future", seen_at="2026-08-15T12:00:00Z", domain="x.com"),
        NewsItem(title="earlier   STORY", seen_at="2026-08-03T09:00:00Z", domain="c.com"),
    ]
    system, user, _ = build_prompt(q, AS_OF, "news", items)
    assert "dated headlines" in system
    assert "- 2026-08-02 (a.com): Earlier story" in user
    assert user.index("Earlier story") < user.index("Later story")
    assert "From the future" not in user
    assert "c.com" not in user  # duplicate title dropped
    assert "published before the current date" in user

    many = [
        NewsItem(title=f"n{i:03d}", seen_at=f"2026-08-{1 + i % 14:02d}T00:00:00Z")
        for i in range(100)
    ]
    kept = select_headlines(many, parse_time(AS_OF))
    assert len(kept) == MAX_HEADLINES
    assert [k.seen_at for k in kept] == sorted(k.seen_at for k in kept)
    assert kept[-1].seen_at == "2026-08-14T00:00:00Z"


def test_news_condition_without_items() -> None:
    _, user, _ = build_prompt(make_question(), AS_OF, "news", None)
    assert "found no headlines" in user


def test_long_rules_truncated() -> None:
    rules = "word " * 2000
    out = truncate_rules(rules)
    assert len(out) < 4100 and out.endswith("(criteria truncated)")
    assert truncate_rules("   ").startswith("(No resolution criteria")
    _, user, _ = build_prompt(make_question(rules=rules), AS_OF)
    assert "(criteria truncated)" in user


def test_rules_with_braces_are_safe() -> None:
    _, user, _ = build_prompt(make_question(rules="Uses {placeholder} text"), AS_OF)
    assert "{placeholder}" in user


def test_format_duration() -> None:
    assert format_duration(timedelta(days=7)) == "7 days"
    assert format_duration(timedelta(hours=1)) == "1 hour"
    assert format_duration(timedelta(days=1, hours=2, minutes=5)) == "1 day 2 hours"
    assert format_duration(timedelta(seconds=10)) == "less than a minute"


def test_bad_condition() -> None:
    with pytest.raises(ValueError):
        build_prompt(make_question(), AS_OF, "market")
