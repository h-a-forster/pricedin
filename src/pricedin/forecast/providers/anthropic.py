"""Anthropic Messages API via the official ``anthropic`` SDK (``pip install pricedin[anthropic]``).

The API key is read from the environment (``ANTHROPIC_API_KEY`` by default) and never stored.
No tools are offered, so the model cannot search the web. No sampling parameters are sent
(current models reject them). Thinking is left at the model default; ``effort`` is sent as
``output_config.effort`` only when set. Prompts are not cached: every system prompt carries its
own as-of date, so there is no shared prefix worth caching.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from pricedin.forecast.providers.base import (
    Completion,
    ProviderError,
    classify_sdk_error,
    is_sdk_error,
    with_retries,
)

EFFORTS = ("low", "medium", "high", "xhigh", "max")

# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
# List prices; models not listed get cost_usd = None.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25, 12.5),
    "claude-fable-5": (10.0, 50.0, 1.0, 12.5),
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-opus-5": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-8": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-7": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-6": (5.0, 25.0, 0.50, 6.25),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
    "claude-sonnet-5": (2.0, 10.0, 0.20, 2.5),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30, 3.75),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25),
}


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    price = PRICES.get(model)
    if price is None:
        # Dated snapshot ids (claude-haiku-4-5-20251001) map to their base id.
        price = next((p for m, p in PRICES.items() if model.startswith(m + "-")), None)
    if price is None:
        return None
    p_in, p_out, p_read, p_write = price
    return (
        usage.get("input_tokens", 0) * p_in
        + usage.get("output_tokens", 0) * p_out
        + usage.get("cache_read_tokens", 0) * p_read
        + usage.get("cache_write_tokens", 0) * p_write
    ) / 1e6


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        effort: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout: float = 600.0,
        retries: int = 4,
        client: Any = None,
    ):
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}, got {effort!r}")
        self.model = model
        self.effort = effort
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.retries = retries
        self.options: dict[str, Any] = {"effort": effort}
        self._client = client
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                try:
                    import anthropic
                except ImportError as exc:
                    raise ProviderError(
                        "the anthropic provider needs the anthropic SDK: "
                        "pip install 'pricedin[anthropic]'"
                    ) from exc
                key = os.environ.get(self.api_key_env)
                if not key:
                    raise ProviderError(
                        f"anthropic: environment variable {self.api_key_env} is not set"
                    )
                # Retries are handled here (with_retries), not by the SDK.
                self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=0)
            return self._client

    def request(self, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        """The ``messages.create`` keyword arguments for one call."""
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort is not None:
            req["output_config"] = {"effort": self.effort}
        return req

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        client = self._get_client()
        req = self.request(system, user, max_tokens)

        def call() -> Any:
            try:
                return client.messages.create(**req)
            except Exception as exc:
                if not is_sdk_error(exc):
                    raise
                raise classify_sdk_error(exc, "anthropic") from exc

        started = time.perf_counter()
        resp = with_retries(call, attempts=self.retries, what="anthropic")
        latency = time.perf_counter() - started
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        u = resp.usage
        usage = {
            "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
            "cache_read_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
            "cache_write_tokens": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        }
        model = getattr(resp, "model", None) or self.model
        return Completion(
            text=text,
            usage=usage,
            cost_usd=estimate_cost(model, usage),
            model=model,
            latency_s=round(latency, 3),
            stop_reason=getattr(resp, "stop_reason", None),
        )
