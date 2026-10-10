"""Extract the final probability and a short rationale from a model reply.

The prompt asks for a last line ``PROBABILITY: <number between 0 and 1>``. The parser takes the
last such line and accepts ``0.37``, ``.37``, ``37%`` and ``37 %``. A bare number above 1 is
rejected (``37`` might mean 37% or a misread scale), as are negatives and NaN. Without a
``PROBABILITY`` line it falls back only to a last line that is clearly a final answer
(``Final answer: 0.37``, ``**0.37**``); anything else gives ``None`` so the task is retried.
"""

from __future__ import annotations

import math
import re

MAX_RATIONALE_CHARS = 600

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_LABELLED = re.compile(
    r"^-?\s*(?:final\s+)?probability(?:\s+of\s+yes)?\s*[:=]\s*(.*)$", re.IGNORECASE
)
_VALUE = re.compile(rf"^({_NUM})\s*(%)?(?!\d|\.\d)")
_FINAL = re.compile(
    rf"^(?:(?:final\s+)?(?:answer|probability|forecast|estimate)\s*[:=\-]?\s*)?({_NUM})\s*(%)?$",
    re.IGNORECASE,
)
_MARKUP = str.maketrans("", "", "*_`#>")


def _clean(line: str) -> str:
    return line.translate(_MARKUP).strip()


def _value(number: str, percent: str | None) -> float | None:
    try:
        v = float(number)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v) or v < 0:
        return None
    if percent:
        return v / 100.0 if v <= 100 else None
    return v if v <= 1 else None


def parse_probability(text: str) -> float | None:
    """The final probability in ``text``, or ``None`` when there is no clear answer."""
    lines = [_clean(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if x]
    for line in reversed(lines):
        m = _LABELLED.match(line)
        if m:
            v = _VALUE.match(m.group(1).strip())
            return _value(v.group(1), v.group(2)) if v else None
    if lines:
        m = _FINAL.match(lines[-1])
        if m:
            return _value(m.group(1), m.group(2))
    return None


def extract_rationale(text: str, limit: int = MAX_RATIONALE_CHARS) -> str:
    """The reply without the ``PROBABILITY`` line, whitespace collapsed, at most ``limit`` chars."""
    kept = [x for x in (text or "").splitlines() if not _LABELLED.match(_clean(x))]
    flat = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3].rstrip() + "..."
