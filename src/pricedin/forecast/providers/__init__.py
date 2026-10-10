"""Model providers and the ``make_provider`` factory. SDK-backed providers import lazily.

Provider specs (the spec string is also stored as ``Forecast.forecaster``):

- ``claude-cli:<model>[@effort]``  Claude Code headless (``claude -p``), local login, no key
- ``anthropic:<model>[@effort]``   Anthropic Messages API, ``ANTHROPIC_API_KEY``
- ``constant:<p>``                 always answers ``p`` (baseline, tests)
- ``echo`` or ``echo:<text>``      returns one fixed reply text (tests, dry runs)
"""

from __future__ import annotations

from typing import Any

from pricedin.forecast.providers.base import (
    Completion,
    Provider,
    ProviderError,
    TransientError,
    with_retries,
)
from pricedin.forecast.providers.scripted import ConstantProvider, EchoProvider, ScriptedProvider

__all__ = [
    "Completion",
    "ConstantProvider",
    "EchoProvider",
    "Provider",
    "ProviderError",
    "ScriptedProvider",
    "TransientError",
    "make_provider",
    "parse_spec",
    "with_retries",
]


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """``"claude-cli:claude-opus-5-5@low"`` -> ``("claude-cli", "claude-opus-5-5", "low")``."""
    kind, sep, rest = spec.strip().partition(":")
    if kind == "echo":
        return kind, rest, None
    if not sep or not rest:
        raise ValueError(f"provider spec must look like '<kind>:<model>[@effort]', got {spec!r}")
    model, _, effort = rest.partition("@")
    if not model:
        raise ValueError(f"provider spec has no model: {spec!r}")
    return kind, model, effort or None


def make_provider(spec: str, **kwargs: Any) -> Provider:
    """Build a provider from a spec string; ``kwargs`` go to its constructor (timeout, retries).

    The scripted providers (constant, echo) ignore ``kwargs``.
    """
    kind, model, effort = parse_spec(spec)
    if kind == "claude-cli":
        from pricedin.forecast.providers.claude_cli import ClaudeCLIProvider

        return ClaudeCLIProvider(model, effort=effort, **kwargs)
    if kind == "anthropic":
        from pricedin.forecast.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model, effort=effort, **kwargs)
    if effort is not None:
        raise ValueError(f"{kind!r} takes no effort: {spec!r}")
    if kind == "constant":
        try:
            p = float(model)
        except ValueError:
            raise ValueError(f"constant spec needs a number: {spec!r}") from None
        return ConstantProvider(p)
    if kind == "echo":
        return EchoProvider(model) if model else EchoProvider()
    raise ValueError(f"unknown provider kind {kind!r} in {spec!r}")
