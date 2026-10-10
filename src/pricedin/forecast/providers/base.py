"""Provider interface: one stateless text completion per call, plus shared retry logic."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@dataclass
class Completion:
    """One model reply. ``usage`` keys: input_tokens, output_tokens, cache_read_tokens,
    cache_write_tokens (missing when the provider does not report them)."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = None
    model: str = ""
    latency_s: float = 0.0
    stop_reason: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": dict(self.usage),
            "cost_usd": self.cost_usd,
            "model": self.model,
            "latency_s": self.latency_s,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Completion:
        return cls(
            text=d["text"],
            usage={k: int(v) for k, v in (d.get("usage") or {}).items()},
            cost_usd=d.get("cost_usd"),
            model=d.get("model", ""),
            latency_s=float(d.get("latency_s", 0.0)),
            stop_reason=d.get("stop_reason"),
        )


@runtime_checkable
class Provider(Protocol):
    """A text-in, text-out model endpoint. Implementations must be thread-safe.

    ``name`` and ``model`` identify the endpoint; ``options`` holds every setting other than
    the prompt and ``max_tokens`` that changes the output (used as part of the cache key).
    """

    name: str
    model: str
    options: dict[str, Any]

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion: ...


class ProviderError(Exception):
    """A failure that retrying will not fix (login, unknown model, missing SDK or key).

    ``fatal`` errors mean every further call will fail the same way, so a run should stop.
    """

    def __init__(self, message: str, *, fatal: bool = True):
        super().__init__(message)
        self.fatal = fatal


class TransientError(Exception):
    """A failure worth retrying (rate limit, overload, timeout, connection drop)."""


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "provider",
) -> T:
    """Call ``fn``; retry on ``TransientError`` with exponential backoff and jitter.

    Other exceptions propagate at once. After ``attempts`` transient failures a non-fatal
    ``ProviderError`` is raised (the next run retries the task).
    """
    attempts = max(1, attempts)
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except TransientError as exc:
            last = exc
            if i + 1 < attempts:
                delay = min(max_delay, base_delay * 2**i)
                sleep(delay * (0.5 + 0.5 * random.random()))
    raise ProviderError(f"{what}: gave up after {attempts} attempts: {last}", fatal=False) from last


def is_sdk_error(exc: BaseException) -> bool:
    """True for exceptions from the ``anthropic`` SDK HTTP layer (``APIError``)."""
    return any(c.__name__ == "APIError" for c in type(exc).__mro__)


def classify_sdk_error(exc: Exception, what: str) -> Exception:
    """Map an ``anthropic`` SDK exception to TransientError or ProviderError."""
    status = getattr(exc, "status_code", None)
    msg = f"{what}: {type(exc).__name__}: {exc}"
    if isinstance(status, int):
        if status in (408, 409, 429) or status >= 500:
            return TransientError(msg)
        # 401/403/404: account or configuration problems, the same for every call.
        # Other 4xx (e.g. 400 for one oversized prompt) only fail this task.
        return ProviderError(msg, fatal=status in (401, 403, 404))
    if type(exc).__name__ in ("APIConnectionError", "APITimeoutError") or isinstance(
        exc, (TimeoutError, ConnectionError)
    ):
        return TransientError(msg)
    return ProviderError(msg, fatal=False)
