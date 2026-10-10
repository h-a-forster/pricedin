"""Leakage-safe news context: dated items provably public before each snapshot time.

Sources: Wikipedia "Portal:Current events" daily pages read at their last revision before
``as_of`` (primary) and GDELT DOC 2.0 headlines (optional). Items are ranked against the
question with BM25. Entry point: ``collect_news``.
"""

from __future__ import annotations

from pricedin.news.collect import LeakageError, check_no_leakage, collect_news
from pricedin.news.relevance import rank_news

__all__ = ["LeakageError", "check_no_leakage", "collect_news", "rank_news"]
