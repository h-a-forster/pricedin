"""On-disk cache of model replies, so re-running a study is free and reproducible.

Entries are keyed by sha256 of (provider, model, options, max_tokens, system, user, salt). The
runner passes ``salt = "<question_id>|<horizon>|<condition>|<sample>"``, so every task maps to
its own entry regardless of thread order, and two tasks that happen to share a prompt still get
independent samples.

Files are written atomically (temp file + rename); the cache is safe to share between threads.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pricedin.forecast.providers.base import Completion, Provider

DEFAULT_DIR = ".pricedin-cache"


def cache_key(
    provider: str,
    model: str,
    options: dict[str, Any],
    max_tokens: int,
    system: str,
    user: str,
    salt: str,
) -> str:
    payload = json.dumps(
        [provider, model, options, max_tokens, system, user, salt],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, directory: str | Path = DEFAULT_DIR):
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            with self._path(key).open(encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
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


class CachedProvider:
    """Wraps a provider; replies come from ``cache`` when present and are stored otherwise.

    Only successful completions are stored; errors propagate and are retried on the next run.
    """

    def __init__(self, inner: Provider, cache: ResponseCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.model = inner.model
        self.options = inner.options

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        salt: str = "",
    ) -> Completion:
        key = cache_key(
            self.inner.name,
            self.inner.model,
            self.inner.options,
            max_tokens,
            system,
            user,
            salt,
        )
        hit = self.cache.get(key)
        if hit is not None:
            comp = Completion.from_dict(hit)
            comp.cached = True
            return comp
        comp = self.inner.complete(system, user, max_tokens=max_tokens)
        self.cache.put(key, comp.to_dict())
        return comp
