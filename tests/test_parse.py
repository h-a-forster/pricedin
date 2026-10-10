from __future__ import annotations

import pytest

from pricedin.forecast.parse import MAX_RATIONALE_CHARS, extract_rationale, parse_probability


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Reasoning.\nPROBABILITY: 0.37", 0.37),
        ("PROBABILITY: .37", 0.37),
        ("PROBABILITY: 37%", 0.37),
        ("PROBABILITY: 37 %", 0.37),
        ("PROBABILITY: 0.37.", 0.37),
        ("**PROBABILITY:** 0.2", 0.2),
        ("probability: 1", 1.0),
        ("PROBABILITY: 0", 0.0),
        ("PROBABILITY = 0.4", 0.4),
        ("Final probability: 0.9", 0.9),
        ("PROBABILITY: 0.1\nmore text\nPROBABILITY: 0.8", 0.8),
        ("PROBABILITY: 0.3 (roughly 30%)", 0.3),
        ("Thoughts...\nFinal answer: 0.25", 0.25),
        ("Thoughts...\n**0.6**", 0.6),
    ],
)
def test_accepts(text: str, expected: float) -> None:
    assert parse_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "PROBABILITY: 37",  # bare number above 1: ambiguous
        "PROBABILITY: -0.2",
        "PROBABILITY: nan",
        "PROBABILITY: 150%",
        "PROBABILITY: <number between 0 and 1>",
        "PROBABILITY: 0.1\nPROBABILITY: unclear",  # the last labelled line wins
        "I think it is likely, maybe 0.7 or so.",
        "There were 3 such events in 2024.\nSo it is unlikely.",
    ],
)
def test_rejects(text: str) -> None:
    assert parse_probability(text) is None


def test_rationale_drops_probability_line_and_truncates() -> None:
    assert extract_rationale("Base rate is low.\n\nPROBABILITY: 0.1") == "Base rate is low."
    long = "word " * 500 + "\nPROBABILITY: 0.5"
    r = extract_rationale(long)
    assert len(r) <= MAX_RATIONALE_CHARS
    assert r.endswith("...")
    assert "PROBABILITY" not in r
