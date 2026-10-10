"""Deterministic fake providers for tests and offline dry runs (no network, no cost)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from pricedin.forecast.providers.base import Completion

Script = str | Sequence[str] | Callable[[str, str], str]

ECHO_REPLY = "Scripted reply: no model was called.\nPROBABILITY: 0.5"


class ScriptedProvider:
    """Returns canned text.

    ``script`` is one string (returned every call), a sequence (returned in order, the last
    entry repeating), or a callable ``(system, user) -> text``.
    """

    name = "scripted"

    def __init__(self, script: Script, model: str = "scripted"):
        self.script = script
        self.model = model
        label = script if isinstance(script, str) else type(script).__name__
        self.options: dict[str, Any] = {"script": label}
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        with self._lock:
            i = len(self.calls)
            self.calls.append((system, user))
        if callable(self.script):
            text = self.script(system, user)
        elif isinstance(self.script, str):
            text = self.script
        else:
            text = self.script[min(i, len(self.script) - 1)]
        return Completion(
            text=text,
            usage={"input_tokens": len(system + user) // 4, "output_tokens": len(text) // 4},
            cost_usd=0.0,
            model=self.model,
        )


class ConstantProvider(ScriptedProvider):
    """Always answers ``p`` (spec ``constant:<p>``): a no-skill baseline and a test double."""

    def __init__(self, p: float):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"constant probability must be in [0, 1], got {p}")
        super().__init__(f"Constant forecaster.\nPROBABILITY: {p}", model=f"constant-{p}")
        self.name = "constant"
        self.options = {"p": p}


class EchoProvider(ScriptedProvider):
    """Returns one fixed reply text (spec ``echo`` or ``echo:<text>``)."""

    def __init__(self, text: str = ECHO_REPLY):
        super().__init__(text, model="echo")
        self.name = "echo"
