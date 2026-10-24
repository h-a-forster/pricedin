from __future__ import annotations

import json
import subprocess

import pytest

from pricedin.forecast.providers import (
    ConstantProvider,
    EchoProvider,
    ProviderError,
    TransientError,
    claude_cli,
    make_provider,
    parse_spec,
    with_retries,
)
from pricedin.forecast.providers.anthropic import AnthropicProvider, estimate_cost
from pricedin.forecast.providers.base import Completion, classify_sdk_error
from pricedin.forecast.providers.claude_cli import (
    ClaudeCLIProvider,
    parse_cli_output,
    parse_version,
)


def test_parse_spec() -> None:
    assert parse_spec("claude-cli:claude-opus-5-5@low") == ("claude-cli", "claude-opus-5-5", "low")
    assert parse_spec("anthropic:claude-haiku-5-5") == ("anthropic", "claude-haiku-5-5", None)
    assert parse_spec("constant:0.5") == ("constant", "0.5", None)
    assert parse_spec("echo") == ("echo", "", None)
    for bad in ("claude-cli", "claude-cli:", "claude-cli:@low"):
        with pytest.raises(ValueError):
            parse_spec(bad)


def test_make_provider() -> None:
    cli = make_provider("claude-cli:claude-haiku-5-5@low", timeout=30)
    assert isinstance(cli, ClaudeCLIProvider)
    assert (cli.model, cli.effort, cli.timeout) == ("claude-haiku-5-5", "low", 30)
    api = make_provider("anthropic:claude-opus-5-5@high")
    assert isinstance(api, AnthropicProvider) and api.options == {"effort": "high"}
    assert isinstance(make_provider("constant:0.3"), ConstantProvider)
    assert isinstance(make_provider("echo"), EchoProvider)
    assert make_provider("echo:hello").complete("s", "u", max_tokens=1).text == "hello"
    for bad in ("claude-cli:m@huge", "constant:abc", "constant:1.5", "constant:0.5@low", "x:y"):
        with pytest.raises(ValueError):
            make_provider(bad)


def test_constant_provider_reply_parses() -> None:
    from pricedin.forecast.parse import parse_probability

    c = make_provider("constant:0.25").complete("s", "u", max_tokens=1)
    assert parse_probability(c.text) == 0.25 and c.cost_usd == 0.0


def test_cli_argv_locks_down_claude_code() -> None:
    p = ClaudeCLIProvider("claude-haiku-5-5", effort="low")
    argv = p.argv("claude", "sys.md")
    joined = " ".join(argv)
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--system-prompt-file") + 1] == "sys.md"
    assert argv[argv.index("--max-turns") + 1] == "1"
    assert argv[argv.index("--effort") + 1] == "low"
    for flag in (
        "--strict-mcp-config",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-session-persistence",
    ):
        assert flag in argv
    assert "--append-system-prompt" not in joined
    assert "--effort" not in ClaudeCLIProvider("m").argv("claude", "s")


def _result(**kw) -> str:
    obj = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Reasoning.\nPROBABILITY: 0.4",
        "total_cost_usd": 0.0012,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 0,
        },
        "modelUsage": {"claude-haiku-5-5": {}},
    }
    obj.update(kw)
    return json.dumps(obj)


def test_parse_cli_output_success() -> None:
    c = parse_cli_output("some log line\n" + _result(), "", 0)
    assert c.text.endswith("PROBABILITY: 0.4")
    assert c.cost_usd == pytest.approx(0.0012)
    assert c.usage["input_tokens"] == 100 and c.usage["cache_read_tokens"] == 10
    assert c.model == "claude-haiku-5-5"


def test_parse_cli_output_errors() -> None:
    with pytest.raises(ProviderError) as exc:
        parse_cli_output("", "Error: Not logged in. Please run /login", 1)
    assert exc.value.fatal
    with pytest.raises(TransientError):
        parse_cli_output("", "connection reset", 1)
    with pytest.raises(TransientError):
        parse_cli_output(_result(is_error=True, subtype="error", api_error_status=529), "", 1)
    with pytest.raises(ProviderError):
        parse_cli_output(_result(is_error=True, subtype="error", api_error_status=404), "", 1)


def test_with_retries() -> None:
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("busy")
        return "ok"

    assert with_retries(flaky, attempts=4, sleep=lambda s: None) == "ok"

    def always() -> str:
        raise TransientError("busy")

    with pytest.raises(ProviderError) as exc:
        with_retries(always, attempts=2, sleep=lambda s: None)
    assert not exc.value.fatal


def test_classify_sdk_error() -> None:
    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, status: int):
            super().__init__(f"status {status}")
            self.status_code = status

    assert isinstance(classify_sdk_error(APIStatusError(429), "a"), TransientError)
    assert isinstance(classify_sdk_error(APIStatusError(529), "a"), TransientError)
    auth = classify_sdk_error(APIStatusError(401), "a")
    assert isinstance(auth, ProviderError) and auth.fatal
    bad = classify_sdk_error(APIStatusError(400), "a")
    assert isinstance(bad, ProviderError) and not bad.fatal


def test_anthropic_with_fake_client() -> None:
    class Block:
        type = "text"
        text = "Short reasoning.\nPROBABILITY: 0.7"

    class Usage:
        input_tokens = 1000
        output_tokens = 200
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0

    class Resp:
        content = [Block()]
        usage = Usage()
        model = "claude-haiku-5-5"
        stop_reason = "end_turn"

    sent = {}

    class Messages:
        def create(self, **kw):
            sent.update(kw)
            return Resp()

    class Client:
        messages = Messages()

    p = AnthropicProvider("claude-haiku-5-5", effort="low", client=Client())
    c = p.complete("sys", "user", max_tokens=500)
    assert c.text.endswith("0.7")
    assert sent["output_config"] == {"effort": "low"}
    assert "tools" not in sent and sent["system"] == "sys"
    assert c.cost_usd == pytest.approx(estimate_cost("claude-haiku-5-5", c.usage))
    assert estimate_cost("unknown-model", c.usage) is None
    assert c.model == "claude-haiku-5-5"
    assert c.harness.startswith("anthropic-sdk ") and c.harness == p.harness


def test_anthropic_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICEDIN_TEST_NO_KEY", raising=False)
    p = AnthropicProvider("claude-haiku-5-5", api_key_env="PRICEDIN_TEST_NO_KEY")
    with pytest.raises(ProviderError):
        p.complete("s", "u", max_tokens=10)


def test_parse_version() -> None:
    assert parse_version("2.1.287 (Claude Code)\n") == "2.1.287"
    assert parse_version("claude 3.0.0-beta.2") == "3.0.0-beta.2"
    assert parse_version("") == ""
    assert parse_version("no version here") == ""


def _fake_run(calls: list[list[str]], version_out: str = "2.1.287 (Claude Code)", rc: int = 0):
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Because.\nPROBABILITY: 0.4",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "total_cost_usd": 0.001,
        "modelUsage": {"claude-opus-5-5-20260601": {}},
    }

    def run(argv, **kw):
        calls.append(list(argv))
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, rc, stdout=version_out, stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(result), stderr="")

    return run


def test_cli_records_harness_version_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(calls))
    monkeypatch.setattr(claude_cli.shutil, "which", lambda exe: "/usr/bin/claude")
    p = ClaudeCLIProvider("claude-opus-5-5", effort="low")
    c1 = p.complete("sys", "user", max_tokens=100)
    c2 = p.complete("sys", "user", max_tokens=100)
    assert c1.harness == c2.harness == "claude-cli 2.1.287"
    assert c1.model == "claude-opus-5-5-20260601"
    assert sum(1 for a in calls if a[1:] == ["--version"]) == 1


def test_cli_harness_unknown_when_version_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(calls, "boom", rc=1))
    assert ClaudeCLIProvider("m").harness("/usr/bin/claude") == "claude-cli unknown"

    def broken(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 60)

    monkeypatch.setattr(claude_cli.subprocess, "run", broken)
    assert ClaudeCLIProvider("m").harness("/usr/bin/claude") == "claude-cli unknown"


def test_completion_round_trip_keeps_harness() -> None:
    c = Completion(text="x", model="m", harness="claude-cli 2.1.287")
    assert Completion.from_dict(c.to_dict()).harness == "claude-cli 2.1.287"
    old = {"text": "x", "usage": {}, "model": "m"}  # cache entry written before harness existed
    assert Completion.from_dict(old).harness == ""
