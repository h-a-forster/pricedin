"""Deterministic BM25 ranking of candidate news items against a question.

The query is the question title plus the start of its resolution rules; title terms count
double, and multi-word proper nouns from the question ("Strait of Hormuz", "Federal Reserve")
add a bonus when they appear as a phrase in an item. IDF comes from the candidate set itself,
so scores are comparable only within one call. No randomness: ties break on time, then title.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from pricedin.schema import NewsItem

T = TypeVar("T")

RULES_CHARS = 600
TITLE_WEIGHT = 2.0
PHRASE_BONUS = 1.5  # per matched proper-noun phrase, scaled by the mean term IDF
# An item must match this share of the title terms' IDF mass (or one of its proper-noun
# phrases); keeps "Russia's Federal Security Service" out of a Federal Reserve question.
MIN_COVERAGE = 0.25
# ...and half the IDF mass of the title's capitalised words ("Braun" in "Will John Braun earn
# the most votes in the WA-03 Primary?"), so generic capitalised words cannot carry a match.
MIN_ENTITY_COVERAGE = 0.5

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each else ever every few
    for from further had has have having he her here hers him his how i if in into is it its
    itself just me more most my no nor not now of off on once only or other our out over own
    same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom why will
    with would you your yours also per via may might must shall upon within without yes market
    markets resolve resolves resolved resolution question contract contracts outcome according
    otherwise listed source sources official date time et pt utc jan feb mar apr jun jul aug sep
    sept oct nov dec january february march april june july august september october november
    december monday tuesday wednesday thursday friday saturday sunday
    """.split()  # noqa: SIM905
)

_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_DATEISH = re.compile(r"\d{1,2}|(?:19|20)\d\d")  # days, months, years: in every dated item
_CAPWORD = re.compile(r"\b[A-Z][\w'’.-]+")
# Capitalised words too generic to identify a topic on their own.
_GENERIC = frozenset(
    """
    exact score fc cf sc united city club team game match final party yes no top best new
    jan feb mar apr may jun jul aug sep oct nov dec january february march april june july
    august september october november december monday tuesday wednesday thursday friday
    saturday sunday
    """.split()  # noqa: SIM905
)
_PROPER = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+(?:of|the|de|and|for|on|in)?\s*[A-Z][\w'.-]*)+")


def _stem(tok: str) -> str:
    # Crude plural/possessive stripping; enough to match "rates" with "rate", "Iran's" with "Iran".
    tok = re.sub(r"['’]s?$", "", tok)
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and tok[-3] in "sxz":
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens with punctuation dropped and plurals stripped."""
    out = []
    for raw in _WORD.findall(text.lower().replace("’", "'")):
        tok = _stem(raw)
        if len(tok) < 2 or (not keep_stopwords and _DATEISH.fullmatch(tok)):
            continue
        if keep_stopwords or (tok not in STOPWORDS and raw not in STOPWORDS):
            out.append(tok)
    return out


def proper_phrases(text: str) -> list[tuple[str, ...]]:
    """Multi-word capitalised sequences ("Strait of Hormuz") as token tuples, first-seen order."""
    seen: dict[tuple[str, ...], None] = {}
    for m in _PROPER.finditer(text):
        toks = tokenize(m.group(0), keep_stopwords=True)
        while toks and toks[0] in STOPWORDS:  # "Will the Federal Reserve" -> federal reserve
            toks.pop(0)
        while toks and toks[-1] in STOPWORDS:
            toks.pop()
        if sum(t not in STOPWORDS for t in toks) >= 2:
            seen.setdefault(tuple(toks), None)
    return list(seen)


def entity_terms(title: str) -> set[str]:
    """Tokens of capitalised words in the title, minus stopwords and generic words:
    "Will Venezuela win?" -> {"venezuela"}. Used to require that an item names the subject."""
    out: set[str] = set()
    for m in _CAPWORD.finditer(title):
        for tok in tokenize(m.group(0)):
            if tok not in _GENERIC and not tok.isdigit():
                out.add(tok)
    return out


def question_query(title: str, rules: str = "") -> tuple[Counter[str], list[tuple[str, ...]]]:
    """Weighted query terms and proper-noun phrases from a question's title and rules.
    Title terms carry weight ``>= TITLE_WEIGHT``."""
    rules = rules[:RULES_CHARS]
    weights: Counter[str] = Counter()
    for tok in set(tokenize(title)):
        weights[tok] += TITLE_WEIGHT
    for tok in set(tokenize(rules)):
        weights[tok] += 1.0
    # Phrases come from the title only: a phrase match bypasses the coverage rules, and the
    # rules text names incidental places and sources ("Los Angeles", "Parcl Labs").
    return weights, proper_phrases(title)


def _has_phrase(doc: list[str], phrase: tuple[str, ...]) -> bool:
    n = len(phrase)
    return any(tuple(doc[i : i + n]) == phrase for i in range(len(doc) - n + 1))


def bm25_scores(
    query: Counter[str],
    docs: Sequence[str],
    *,
    phrases: Sequence[tuple[str, ...]] = (),
    focus: Iterable[str] = (),
    min_coverage: float = 0.0,
    entities: Iterable[str] = (),
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """BM25 score of each document for a weighted bag of query terms.

    Documents matching less than ``min_coverage`` of the IDF mass of the ``focus`` terms, and
    none of the ``phrases``, score 0. When ``entities`` is given, a document must also match
    ``MIN_ENTITY_COVERAGE`` of their IDF mass (or a phrase), so rare names count most.
    """
    entities = set(entities)
    toks = [tokenize(d) for d in docs]
    n = len(toks)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in toks) / n or 1.0
    df: Counter[str] = Counter()
    for t in toks:
        df.update(set(t))
    idf = {q: math.log(1.0 + (n - df[q] + 0.5) / (df[q] + 0.5)) for q in query}
    full = [tokenize(d, keep_stopwords=True) for d in docs] if phrases else []
    focus_idf = {q: math.log(1.0 + (n - df[q] + 0.5) / (df[q] + 0.5)) for q in set(focus)}
    focus_mass = sum(focus_idf.values())
    entity_idf = {q: math.log(1.0 + (n - df[q] + 0.5) / (df[q] + 0.5)) for q in entities}
    entity_mass = sum(entity_idf.values())
    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = 0.0
        for q, w in query.items():
            f = tf.get(q, 0)
            if f:
                s += w * idf[q] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl))
        if s > 0:
            matched = [ph for ph in phrases if _has_phrase(full[i], ph)]
            for ph in matched:
                terms = [idf.get(x, 0.0) for x in ph if x in idf] or [1.0]
                s += PHRASE_BONUS * sum(terms) / len(terms)
            if not matched and focus_mass > 0:
                cov = sum(v for q, v in focus_idf.items() if q in tf) / focus_mass
                if cov < min_coverage:
                    s = 0.0
            if not matched and entity_mass > 0:
                ecov = sum(v for q, v in entity_idf.items() if q in tf) / entity_mass
                if ecov < MIN_ENTITY_COVERAGE:
                    s = 0.0
        scores.append(s)
    return scores


def rank(
    query_text: str,
    items: Sequence[T],
    *,
    rules: str = "",
    k: int = 25,
    min_coverage: float = MIN_COVERAGE,
    text: Callable[[T], str] = lambda it: it.title,  # type: ignore[attr-defined]
    when: Callable[[T], str] = lambda it: it.seen_at,  # type: ignore[attr-defined]
) -> list[T]:
    """Top-``k`` items with a positive score, returned oldest first.

    Items must match ``min_coverage`` of the title terms' IDF mass or a proper-noun phrase.

    Selection breaks score ties by newer first, then text; the output is sorted by
    (``when``, text) so the prompt reads chronologically.
    """
    if k <= 0 or not items:
        return []
    weights, phrases = question_query(query_text, rules)
    focus = [q for q, w in weights.items() if w >= TITLE_WEIGHT]
    scores = bm25_scores(
        weights,
        [text(it) for it in items],
        phrases=phrases,
        focus=focus,
        min_coverage=min_coverage,
        entities=entity_terms(query_text),
    )
    order = sorted(
        (i for i in range(len(items)) if scores[i] > 0),
        key=lambda i: (-round(scores[i], 9), _neg(when(items[i])), text(items[i])),
    )
    top = [items[i] for i in order[:k]]
    return sorted(top, key=lambda it: (when(it), text(it)))


def _neg(s: str) -> tuple[int, ...]:
    # Sort key that orders ISO timestamps newest first.
    return tuple(-ord(c) for c in s)


def rank_news(title: str, rules: str, items: Sequence[NewsItem], k: int = 25) -> list[NewsItem]:
    """``rank`` for ``NewsItem``s against a question's title and rules."""
    return rank(title, items, rules=rules, k=k)
