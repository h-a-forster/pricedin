"""Analysis panel: one row per (question, horizon) with the outcome, the market price and each
forecaster's probability, plus leakage-safe train/test splits.

Splits use an embargo on information windows. A row's window runs from its snapshot time
(``as_of``) to the question's close (``close_at``, when the outcome becomes known). Training rows
must have closed before the split; test rows must have been forecast after it; rows whose window
straddles the split belong to neither.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import datetime

import numpy as np

from ..schema import HORIZONS, Forecast, Question, Snapshot, parse_time


def resolved_at(q: Question) -> str:
    """When the outcome became known: the actual close if recorded, else ``close_at``.

    Venue collectors anchor ``close_at`` on the scheduled end, and trading can run past it
    (in-play sports) or stop before it (early resolution). Training rows must have resolved
    before the split, so the split uses this time, not the anchor.
    """
    return str(q.extra.get("actual_close") or q.close_at)


def forecaster_key(forecaster: str, condition: str) -> str:
    return f"{forecaster}|{condition}"


def to_dt64(t: str | int | float | datetime | np.datetime64) -> np.datetime64:
    """Any supported time -> naive UTC ``datetime64[s]``."""
    if isinstance(t, np.datetime64):
        return t.astype("datetime64[s]")
    return np.datetime64(parse_time(t).replace(tzinfo=None), "s")


@dataclass
class Panel:
    """Column arrays, one entry per (question_id, horizon) row.

    ``forecasts`` maps forecaster keys (``"forecaster|condition"``) to probability arrays.
    Probabilities are clipped to [eps, 1 - eps]. ``dropped`` counts rows excluded while building
    the panel, by first failing reason.
    """

    question_id: np.ndarray
    horizon: np.ndarray
    y: np.ndarray
    market: np.ndarray
    forecasts: dict[str, np.ndarray]
    event_id: np.ndarray
    category: np.ndarray
    venue: np.ndarray
    close_at: np.ndarray  # datetime64[s], naive UTC
    as_of: np.ndarray  # datetime64[s], naive UTC
    eps: float = 0.01
    dropped: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.y.size)

    @property
    def keys(self) -> list[str]:
        return list(self.forecasts)

    def columns(self) -> dict[str, np.ndarray]:
        """Probability columns by name: ``market`` plus each forecaster key."""
        return {"market": self.market, **self.forecasts}

    def subset(self, mask) -> Panel:
        """Rows selected by a boolean mask or index array (metadata carried over)."""
        mask = np.asarray(mask)
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                kw[f.name] = v[mask]
        kw["forecasts"] = {k: v[mask] for k, v in self.forecasts.items()}
        return replace(self, **kw)


def build_panel(
    questions: Iterable[Question],
    snapshots: Iterable[Snapshot],
    forecasts: Iterable[Forecast],
    forecaster_keys: Sequence[str] | None = None,
    *,
    horizons: Sequence[str] | None = None,
    eps: float = 0.01,
) -> Panel:
    """Join records into a panel.

    The candidate rows are every question x horizon (default: all of ``HORIZONS``). A row is
    kept when it has a snapshot with a finite mid and a parsed probability for every requested
    forecaster key (default: every key seen in ``forecasts``). Drop reasons, counted once per
    row in this order: ``bad_outcome``, ``missing_snapshot``, ``missing_market``,
    ``missing_forecast:<key>``, ``unparsed_forecast:<key>``, ``as_of_mismatch:<key>`` (the
    forecast was made for a different snapshot time). If a key has several records for a row,
    the last parsed one wins.
    """
    horizons = list(horizons or HORIZONS)
    qs = {q.id: q for q in questions}
    snaps = {(s.question_id, s.horizon): s for s in snapshots}
    fc: dict[tuple[str, str, str], Forecast] = {}
    for f in forecasts:
        key = (f.question_id, f.horizon, forecaster_key(f.forecaster, f.condition))
        if f.prob is not None or key not in fc or fc[key].prob is None:
            fc[key] = f
    keys = list(forecaster_keys) if forecaster_keys is not None else sorted({k[2] for k in fc})

    dropped: Counter[str] = Counter()
    rows: list[tuple[Question, str, Snapshot, list[float]]] = []
    for qid in sorted(qs):
        q = qs[qid]
        for h in horizons:
            reason, probs = _check_row(q, h, snaps.get((qid, h)), fc, keys)
            if reason:
                dropped[reason] += 1
            else:
                rows.append((q, h, snaps[(qid, h)], probs))

    def col(values, dtype=None) -> np.ndarray:
        return np.array(list(values), dtype=dtype)

    return Panel(
        question_id=col((r[0].id for r in rows), object),
        horizon=col((r[1] for r in rows), object),
        y=col((r[0].outcome for r in rows), int),
        market=np.clip(col((r[2].mid for r in rows), float), eps, 1 - eps),
        forecasts={
            k: np.clip(col((r[3][j] for r in rows), float), eps, 1 - eps)
            for j, k in enumerate(keys)
        },
        event_id=col((r[0].event_id for r in rows), object),
        category=col((r[0].category for r in rows), object),
        venue=col((r[0].venue for r in rows), object),
        close_at=col((to_dt64(resolved_at(r[0])) for r in rows), "datetime64[s]"),
        as_of=col((to_dt64(r[2].as_of) for r in rows), "datetime64[s]"),
        eps=eps,
        dropped=dict(sorted(dropped.items())),
    )


def _check_row(q, h, snap, fc, keys) -> tuple[str | None, list[float]]:
    if q.outcome not in (0, 1):
        return "bad_outcome", []
    if snap is None:
        return "missing_snapshot", []
    if snap.mid is None or not np.isfinite(snap.mid):
        return "missing_market", []
    probs = []
    as_of = to_dt64(snap.as_of)
    for k in keys:
        f = fc.get((q.id, h, k))
        if f is None:
            return f"missing_forecast:{k}", []
        if f.prob is None or not np.isfinite(f.prob):
            return f"unparsed_forecast:{k}", []
        if f.as_of and to_dt64(f.as_of) != as_of:
            return f"as_of_mismatch:{k}", []
        probs.append(float(f.prob))
    return None, probs


# --- splits -------------------------------------------------------------------------------------


def split_masks(panel: Panel, split_at) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) boolean masks for a time split with an embargo.

    train: ``close_at < split_at``; test: ``as_of >= split_at``. Rows with
    ``as_of < split_at <= close_at`` are in neither.
    """
    t = to_dt64(split_at)
    return panel.close_at < t, panel.as_of >= t


def split_at_quantile(panel: Panel, q: float = 0.5) -> np.datetime64:
    """A split time at the ``q`` quantile of close times (a convenient default)."""
    secs = np.sort(panel.close_at.astype("int64"))
    i = min(len(secs) - 1, max(0, int(round(q * (len(secs) - 1)))))
    return np.datetime64(int(secs[i]), "s")


@dataclass
class Fold:
    start: np.datetime64
    end: np.datetime64
    train: np.ndarray
    test: np.ndarray


def walk_forward(panel: Panel, *, min_train: int = 100) -> list[Fold]:
    """Monthly expanding-window folds.

    For a month [start, end): train on rows with ``close_at < start``; test on rows with
    ``start <= as_of < end`` (all of which have ``as_of >= start``, so the embargo holds). The
    first fold is the first month whose training set has at least ``min_train`` rows; months with
    no test rows are skipped. Each row is tested in at most one fold.
    """
    if len(panel) == 0:
        return []
    first = panel.as_of.min().astype("datetime64[M]")
    last = panel.as_of.max().astype("datetime64[M]")
    folds = []
    m = first
    while m <= last:
        start, end = m.astype("datetime64[s]"), (m + np.timedelta64(1, "M")).astype("datetime64[s]")
        train = panel.close_at < start
        test = (panel.as_of >= start) & (panel.as_of < end)
        if train.sum() >= min_train and test.any():
            folds.append(Fold(start, end, train, test))
        m = m + np.timedelta64(1, "M")
    return folds
