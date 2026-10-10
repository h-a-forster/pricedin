from __future__ import annotations

import httpx
import pytest

from pricedin.http import USER_AGENT, HttpClient, HttpError, cache_key, parse_retry_after


class FakeTime:
    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def make(handler, tmp_path=None, **kw) -> tuple[HttpClient, FakeTime]:
    ft = FakeTime()
    client = HttpClient(
        transport=httpx.MockTransport(handler),
        sleep=ft.sleep,
        clock=ft.clock,
        cache_dir=tmp_path,
        seed=0,
        **kw,
    )
    return client, ft


def test_user_agent_and_params():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"ok": True})

    client, _ = make(handler)
    assert client.get_json("https://x.test/api", {"a": 1, "b": "z"}) == {"ok": True}
    assert seen[0].headers["User-Agent"] == USER_AGENT
    assert "pricedin/" in USER_AGENT and "github.com/h-a-forster/pricedin" in USER_AGENT
    assert dict(seen[0].url.params) == {"a": "1", "b": "z"}


def test_retries_on_5xx_with_backoff():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json=[1])

    client, ft = make(handler, min_interval=0.0, backoff_base=1.0)
    assert client.get_json("https://x.test/") == [1]
    assert len(calls) == 3
    assert client.stats["retries"] == 2
    # Backoff 1 * 2**attempt with jitter in [0.5, 1.5).
    assert 0.5 <= ft.sleeps[0] < 1.5
    assert 1.0 <= ft.sleeps[1] < 3.0


def test_retry_after_is_honoured():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={})

    client, ft = make(handler, min_interval=0.0, backoff_base=0.1)
    client.get_json("https://x.test/")
    assert ft.sleeps and ft.sleeps[0] >= 7.0


def test_timeout_is_retried():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, json={"x": 1})

    client, _ = make(handler, min_interval=0.0)
    assert client.get_json("https://x.test/") == {"x": 1}
    assert len(calls) == 2


def test_non_retryable_and_exhausted():
    client, _ = make(lambda req: httpx.Response(404, text="nope"), min_interval=0.0)
    with pytest.raises(HttpError) as ei:
        client.get_json("https://x.test/")
    assert ei.value.status == 404
    assert client.stats["requests"] == 1

    client, _ = make(lambda req: httpx.Response(500), min_interval=0.0, max_retries=2)
    with pytest.raises(HttpError, match="3 attempts"):
        client.get_json("https://x.test/")
    assert client.stats["requests"] == 3


def test_non_json_body_raises():
    client, _ = make(lambda req: httpx.Response(200, text="Please limit requests"))
    with pytest.raises(HttpError, match="non-JSON"):
        client.get_json("https://x.test/")


def test_cache_avoids_second_request(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json={"n": len(calls)})

    client, _ = make(handler, tmp_path=tmp_path)
    a = client.get_json("https://x.test/a", {"q": "1"})
    b = client.get_json("https://x.test/a", {"q": "1"})
    assert a == b == {"n": 1}
    assert len(calls) == 1 and client.stats["cache_hits"] == 1
    files = list((tmp_path / "http").rglob("*.json"))
    assert len(files) == 1 and files[0].stem == cache_key("GET", "https://x.test/a", {"q": "1"})
    assert not list((tmp_path / "http").rglob(".tmp-*"))
    # A fresh client reads the same cache; use_cache=False bypasses it.
    client2, _ = make(handler, tmp_path=tmp_path)
    assert client2.get_json("https://x.test/a", {"q": "1"}) == {"n": 1}
    assert client2.get_json("https://x.test/a", {"q": "1"}, use_cache=False) == {"n": 2}
    # Different params -> different key.
    assert client2.get_json("https://x.test/a", {"q": "2"}) == {"n": 3}


def test_errors_are_not_cached(tmp_path):
    client, _ = make(lambda req: httpx.Response(404), tmp_path=tmp_path)
    with pytest.raises(HttpError):
        client.get_json("https://x.test/")
    assert not list(tmp_path.rglob("*.json"))


def test_per_host_interval():
    client, ft = make(
        lambda req: httpx.Response(200, json={}),
        min_interval=2.0,
        host_intervals={"slow.test": 5.0},
    )
    client.get_json("https://a.test/")
    client.get_json("https://a.test/")
    assert ft.sleeps == [2.0]
    client.get_json("https://b.test/")  # other host: no wait
    assert ft.sleeps == [2.0]
    client.get_json("https://slow.test/")
    client.get_json("https://slow.test/")
    assert ft.sleeps == [2.0, 5.0]
    assert client.stats["requests:a.test"] == 2


def test_parse_retry_after():
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after("Thu, 01 Jan 1970 00:01:40 GMT", now=40.0) == pytest.approx(60.0)
