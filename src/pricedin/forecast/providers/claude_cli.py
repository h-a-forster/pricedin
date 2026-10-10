"""Claude Code in headless print mode (``claude -p``), using the local Claude Code login.

No API key is needed. Each call runs, in a fresh empty temporary directory::

    claude -p --output-format json --no-session-persistence --model MODEL
        --system-prompt-file SYSTEM.md --tools "" --strict-mcp-config --setting-sources ""
        --safe-mode --disable-slash-commands --max-turns 1 [--effort LEVEL]

with the user prompt on stdin. The system prompt replaces Claude Code's own. No tools (so no web
search), MCP servers, settings files, hooks, plugins, skills or CLAUDE.md files are loaded.
``max_tokens`` is not supported by the CLI and is ignored. Cost is Claude Code's
``total_cost_usd`` (the API list price of the call, whether or not a subscription pays for it).

Leakage caveat (checked with Claude Code 2.1.287): even with a replaced system prompt, Claude
Code attaches context of its own to the first user message: an environment block (working
directory, OS, shell, model id), the account email, and "Today's date is <real date>". The
model therefore sees the real run date next to the forecast date stated in the prompt. Neither
an ISO datetime nor a plain date in ``CLAUDE_CODE_OVERRIDE_DATE`` changed it. Prefer the
``anthropic`` provider, which sends only the prompt, for results where this matters.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from typing import Any

from pricedin.forecast.providers.base import (
    Completion,
    ProviderError,
    TransientError,
    with_retries,
)

EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Account or login problems: retrying will not help.
_AUTH = re.compile(
    r"authentication|invalid api key|not logged in|/login|oauth token|unauthori[sz]ed"
    r"|credit balance|usage limit|quota",
    re.IGNORECASE,
)
_TAIL = 400


def _tail(text: str, n: int = _TAIL) -> str:
    return text.strip()[-n:]


def _result_object(stdout: str) -> dict[str, Any] | None:
    """The ``{"type": "result", ...}`` object in stdout (possibly after log lines)."""
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    found = None
    for line in stdout.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(line, start)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            found = obj
    return found


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def parse_cli_output(stdout: str, stderr: str, exit_code: int | None) -> Completion:
    """Turn ``claude -p --output-format json`` output into a Completion.

    Raises a fatal ProviderError for login/account failures and TransientError for rate limits,
    overloads, network errors and unexplained failures.
    """
    obj = _result_object(stdout)
    if obj is None:
        detail = _tail(stderr) or _tail(stdout) or "no output"
        if _AUTH.search(detail):
            raise ProviderError(f"claude-cli: {detail}")
        raise TransientError(f"claude-cli: exit code {exit_code} without a JSON result: {detail}")

    text = obj.get("result") if isinstance(obj.get("result"), str) else ""
    if obj.get("is_error") is True or obj.get("subtype") not in (None, "success"):
        status = obj.get("api_error_status")
        detail = f"{obj.get('subtype')}: {_tail(text) or _tail(stderr)}"
        if status in (401, 403) or _AUTH.search(detail):
            raise ProviderError(f"claude-cli: {detail}")
        if status == 404:
            raise ProviderError(f"claude-cli: unknown model or endpoint (404): {detail}")
        raise TransientError(f"claude-cli: API error {status}: {detail}")

    tokens = obj.get("usage") if isinstance(obj.get("usage"), Mapping) else {}
    usage = {
        "input_tokens": _int(tokens.get("input_tokens")),
        "output_tokens": _int(tokens.get("output_tokens")),
        "cache_read_tokens": _int(tokens.get("cache_read_input_tokens")),
        "cache_write_tokens": _int(tokens.get("cache_creation_input_tokens")),
    }
    details = tokens.get("output_tokens_details")
    if isinstance(details, Mapping) and _int(details.get("thinking_tokens")):
        usage["reasoning_tokens"] = _int(details.get("thinking_tokens"))
    cost = obj.get("total_cost_usd")
    model_usage = obj.get("modelUsage")
    model = next(iter(model_usage), "") if isinstance(model_usage, Mapping) and model_usage else ""
    return Completion(
        text=text,
        usage=usage,
        cost_usd=float(cost) if isinstance(cost, (int, float)) and cost >= 0 else None,
        model=model,
        stop_reason=obj.get("stop_reason"),
    )


class ClaudeCLIProvider:
    name = "claude-cli"

    def __init__(
        self,
        model: str,
        *,
        effort: str | None = None,
        executable: str = "claude",
        timeout: float = 600.0,
        retries: int = 4,
        extra_args: tuple[str, ...] = (),
    ):
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}, got {effort!r}")
        self.model = model
        self.effort = effort
        self.executable = executable
        self.timeout = timeout
        self.retries = retries
        self.extra_args = tuple(extra_args)
        self.options: dict[str, Any] = {"effort": effort, "extra_args": list(self.extra_args)}

    def argv(self, exe: str, system_file: str) -> list[str]:
        argv = [
            exe,
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--model",
            self.model,
            "--system-prompt-file",
            system_file,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--safe-mode",
            "--disable-slash-commands",
            "--max-turns",
            "1",
        ]
        if self.effort:
            argv += ["--effort", self.effort]
        return argv + list(self.extra_args)

    def _run_once(self, exe: str, system: str, user: str) -> Completion:
        with tempfile.TemporaryDirectory(prefix="pricedin-cli-", ignore_cleanup_errors=True) as tmp:
            # The system prompt goes through a file: multi-line arguments are fragile on Windows.
            system_file = os.path.join(tmp, "system.md")
            with open(system_file, "w", encoding="utf-8", newline="") as fh:
                fh.write(system)
            work = os.path.join(tmp, "work")
            os.mkdir(work)
            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    self.argv(exe, system_file),
                    input=user,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=work,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise TransientError(f"claude-cli: timed out after {self.timeout}s") from exc
            except OSError as exc:
                raise ProviderError(f"claude-cli: could not start {exe}: {exc}") from exc
            latency = time.perf_counter() - started
        comp = parse_cli_output(proc.stdout or "", proc.stderr or "", proc.returncode)
        comp.latency_s = round(latency, 3)
        comp.model = comp.model or self.model
        return comp

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        exe = shutil.which(self.executable)
        if exe is None:
            raise ProviderError(f"claude-cli: executable {self.executable!r} not found on PATH")
        return with_retries(
            lambda: self._run_once(exe, system, user),
            attempts=self.retries,
            what="claude-cli",
        )
