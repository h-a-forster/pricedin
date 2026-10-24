"""Polite JSON-over-HTTP client shared by every network collector.

One place for the etiquette public APIs ask for: a descriptive User-Agent, a minimum interval
between requests to the same host (thread-safe), retries with exponential backoff and jitter on
429/5xx/timeouts that honour ``Retry-After``, and an optional on-disk response cache so re-running
a collection never re-hits an API. Cache entries are keyed by sha256 of (method, url, params) and
written atomically; only successful JSON responses are cached.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import random
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from pricedin import __version__

USER_AGENT = f"pricedin/{__version__} (+https://github.com/h-a-forster/pricedin)"

# Seconds between requests per host, unless overridden. GDELT allows one request per 5 s per IP;
# MediaWiki asks for serial requests, ~1/s is fine.
DEFAULT_HOST_INTERVALS: dict[str, float] = {
    "api.gdeltproject.org": 5.5,
    "en.wikipedia.org": 1.0,
}
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 520, 521, 522, 524})


class HttpError(RuntimeError):
    """A request failed for good (non-retryable status, retries exhausted, or bad JSON)."""

    def __init__(self, message: str, *, status: int | None = None, url: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url


def cache_key(method: str, url: str, params: Mapping[str, Any] | None) -> str:
    items = sorted((str(k), str(v)) for k, v in (params or {}).items())
    payload = json.dumps([method.upper(), url, items], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """``Retry-After`` as seconds (delta-seconds or an HTTP date); None if absent or unparsable."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    ref = datetime.now(UTC).timestamp() if now is None else now
    return max(0.0, when.timestamp() - ref)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # Windows: target briefly open by a reader
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class HttpClient:
    """GET JSON politely. Safe to share between threads.

    ``min_interval`` is the default per-host spacing in seconds; ``host_intervals`` overrides it
    per host (merged over ``DEFAULT_HOST_INTERVALS``). ``cache_dir=None`` disables the cache.
    ``transport``, ``sleep`` and ``clock`` exist for tests.
    """

    def __init__(
        self,
        *,
        user_agent: str = USER_AGENT,
        min_interval: float = 1.0,
        host_intervals: Mapping[str, float] | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 120.0,
        timeout: float = 30.0,
        cache_dir: str | Path | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        seed: int | None = None,
    ):
        self.min_interval = min_interval
        self.host_intervals = {**DEFAULT_HOST_INTERVALS, **(host_intervals or {})}
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._sleep = sleep
        self._clock = clock
        self._rng = random.Random(seed)
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._lock = threading.Lock()
        self._next_slot: dict[str, float] = {}
        self.stats: Counter[str] = Counter()  # requests, cache_hits, retries, per-host requests

    # -- lifecycle -------------------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals -------------------------------------------------------------------------
    def interval_for(self, host: str) -> float:
        return self.host_intervals.get(host, self.min_interval)

    def _wait_turn(self, host: str) -> None:
        # Reserve the next slot under the lock, then sleep outside it so other hosts aren't blocked.
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot.get(host, now))
            self._next_slot[host] = slot + self.interval_for(host)
        if slot > now:
            self._sleep(slot - now)

    def _push_back(self, host: str, delay: float) -> None:
        # After a 429/Retry-After, nobody else may hit this host before the delay passes either.
        with self._lock:
            until = self._clock() + delay
            self._next_slot[host] = max(self._next_slot.get(host, until), until)

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_max, self.backoff_base * (2**attempt))
        return base * (0.5 + self._rng.random())  # jitter in [0.5, 1.5) x

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "http" / key[:2] / f"{key}.json"

    # -- public ----------------------------------------------------------------------------
    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        use_cache: bool = True,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """GET ``url`` with ``params`` and return the decoded JSON body.

        Served from the cache when present. Raises ``HttpError`` on a non-retryable status, on
        exhausted retries, or when the body is not JSON.
        """
        key = cache_key("GET", url, params)
        path = self._cache_path(key) if use_cache else None
        if path is not None:
            try:
                with path.open(encoding="utf-8") as fh:
                    entry = json.load(fh)
                self.stats["cache_hits"] += 1
                return entry["body"]
            except (OSError, ValueError, KeyError):
                pass

        host = urlsplit(url).hostname or ""
        last_err = ""
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_turn(host)
            self.stats["requests"] += 1
            self.stats[f"requests:{host}"] += 1
            retry_after: float | None = None
            try:
                resp = self._client.get(url, params=dict(params or {}), headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 400:
                    try:
                        body = resp.json()
                    except ValueError as exc:
                        snippet = resp.text[:200].replace("\n", " ")
                        raise HttpError(
                            f"non-JSON response from {url}: {snippet!r}",
                            status=resp.status_code,
                            url=url,
                        ) from exc
                    if path is not None:
                        atomic_write_json(
                            path,
                            {
                                "url": url,
                                "params": {str(k): str(v) for k, v in (params or {}).items()},
                                "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "body": body,
                            },
                        )
                    return body
                last_status = resp.status_code
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]!r}"
                if resp.status_code not in RETRY_STATUSES:
                    raise HttpError(last_err, status=resp.status_code, url=url)
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))
            if attempt == self.max_retries:
                break
            delay = self._backoff(attempt)
            if retry_after is not None:
                delay = max(delay, min(retry_after, self.backoff_max))
            self.stats["retries"] += 1
            self._push_back(host, delay)  # the next _wait_turn sleeps it off
        raise HttpError(
            f"giving up on {url} after {self.max_retries + 1} attempts: {last_err}",
            status=last_status,
            url=url,
        )
