from __future__ import annotations

import numpy as np
import pytest

from pricedin.eval.panel import (
    build_panel,
    split_at_quantile,
    split_masks,
    to_dt64,
    walk_forward,
)
from pricedin.schema import Forecast, Question, Snapshot

KEY = "m1|base"


def _q(i: int, close: str, outcome: int = 1, event: str | None = None) -> Question:
    return Question(
        id=f"kalshi:Q{i}",
        venue="kalshi",
        event_id=event or f"E{i}",
        title=f"q{i}",
        rules="",
        category="politics",
        created_at="2025-01-01T00:00:00Z",
        open_at="2025-01-01T00:00:00Z",
        close_at=close,
        scheduled_close_at=close,
        outcome=outcome,
        volume=100.0,
    )


def _s(i: int, h: str, as_of: str, mid: float = 0.6) -> Snapshot:
    return Snapshot(question_id=f"kalshi:Q{i}", horizon=h, as_of=as_of, mid=mid)


def _f(i: int, h: str, as_of: str, prob, cond: str = "base") -> Forecast:
    return Forecast(
        question_id=f"kalshi:Q{i}",
        horizon=h,
        as_of=as_of,
        forecaster="m1",
        condition=cond,
        prob=prob,
    )


def test_build_panel_joins_and_counts_drops():
    qs = [_q(i, "2025-03-10T00:00:00Z") for i in range(5)]
    a = "2025-03-09T00:00:00Z"
    snaps = [_s(0, "1d", a), _s(1, "1d", a), _s(2, "1d", a, mid=1.0), _s(3, "1d", a)]
    fcs = [
        _f(0, "1d", a, 0.7),
        _f(0, "1d", a, None),  # a later failed retry must not override a parsed forecast
        _f(1, "1d", a, None),
        _f(2, "1d", a, 0.0),
        _f(3, "1d", "2025-03-01T00:00:00Z", 0.5),
    ]
    p = build_panel(qs, snaps, fcs, [KEY], horizons=["1d"], eps=0.01)
    assert p.question_id.tolist() == ["kalshi:Q0", "kalshi:Q2"]
    assert p.forecasts[KEY].tolist() == pytest.approx([0.7, 0.01])
    assert p.market.tolist() == pytest.approx([0.6, 0.99])  # clipped
    assert p.dropped == {
        "as_of_mismatch:m1|base": 1,
        "missing_snapshot": 1,
        "unparsed_forecast:m1|base": 1,
    }
    assert p.close_at.dtype == np.dtype("datetime64[s]")
    assert p.keys == [KEY]


def test_build_panel_default_keys_and_missing_forecast():
    qs = [_q(0, "2025-03-10T00:00:00Z"), _q(1, "2025-03-10T00:00:00Z")]
    a = "2025-03-09T00:00:00Z"
    snaps = [_s(0, "1d", a), _s(1, "1d", a)]
    fcs = [_f(0, "1d", a, 0.4), _f(0, "1d", a, 0.3, cond="news"), _f(1, "1d", a, 0.2)]
    p = build_panel(qs, snaps, fcs, horizons=["1d"])
    assert p.keys == ["m1|base", "m1|news"]
    assert len(p) == 1
    assert p.dropped == {"missing_forecast:m1|news": 1}


def _toy_panel():
    # close times on the 10th of Jan..Jun; 1d horizon
    qs, snaps, fcs = [], [], []
    for i in range(60):
        month = 1 + i % 6
        close = f"2025-{month:02d}-10T00:00:00Z"
        as_of = f"2025-{month:02d}-09T00:00:00Z"
        qs.append(_q(i, close, outcome=i % 2))
        snaps.append(_s(i, "1d", as_of))
        fcs.append(_f(i, "1d", as_of, 0.5))
    return build_panel(qs, snaps, fcs, [KEY], horizons=["1d"])


def test_split_with_embargo():
    p = _toy_panel()
    # split at 2025-03-09T12:00: March rows have as_of before, close after -> embargoed
    tr, te = split_masks(p, "2025-03-09T12:00:00Z")
    assert not np.any(tr & te)
    assert np.all(p.close_at[tr] < to_dt64("2025-03-09T12:00:00Z"))
    assert np.all(p.as_of[te] >= to_dt64("2025-03-09T12:00:00Z"))
    assert tr.sum() == 20 and te.sum() == 30
    embargoed = ~tr & ~te
    assert embargoed.sum() == 10
    assert set(p.close_at[embargoed].astype(str)) == {"2025-03-10T00:00:00"}


def test_subset_and_quantile_split():
    p = _toy_panel()
    s = p.subset(p.y == 1)
    assert len(s) == 30 and np.all(s.y == 1) and len(s.forecasts[KEY]) == 30
    t = split_at_quantile(p, 0.5)
    assert to_dt64("2025-03-10T00:00:00Z") <= t <= to_dt64("2025-04-10T00:00:00Z")


def test_walk_forward_folds():
    p = _toy_panel()
    folds = walk_forward(p, min_train=15)
    starts = [str(f.start) for f in folds]
    assert starts[0] == "2025-03-01T00:00:00"  # Feb fold has only 10 closed rows
    assert len(folds) == 4
    tested = np.zeros(len(p), dtype=int)
    for f in folds:
        assert np.all(p.close_at[f.train] < f.start)
        assert np.all(p.as_of[f.test] >= f.start) and np.all(p.as_of[f.test] < f.end)
        tested += f.test
    assert tested.max() == 1


def test_sensitivity_clusters_series_and_resolution_date():
    qs = [_q(i, "2025-03-10T00:00:00Z") for i in range(3)]
    qs[0].extra["series"] = "KXS"
    qs[1].extra["series"] = "KXS"
    qs[2].extra["actual_close"] = "2025-03-08T23:30:00Z"  # resolved early, a day before
    a = "2025-03-07T00:00:00Z"
    snaps = [_s(i, "1d", a) for i in range(3)]
    fcs = [_f(i, "1d", a, 0.5) for i in range(3)]
    p = build_panel(qs, snaps, fcs, [KEY], horizons=["1d"])
    cl = p.sensitivity_clusters()
    assert cl["series"].tolist() == ["kalshi:series:KXS", "kalshi:series:KXS", "E2"]  # no series
    assert cl["resolution_date"].tolist() == ["2025-03-10", "2025-03-10", "2025-03-08"]
    sub = p.subset(np.array([False, True, True]))
    assert sub.sensitivity_clusters()["series"].tolist() == ["kalshi:series:KXS", "E2"]
