from __future__ import annotations

import threading
from pathlib import Path

from pricedin.forecast.cache import CachedProvider, ResponseCache, cache_key
from pricedin.forecast.providers import ScriptedProvider


def test_cache_hit_and_salt(tmp_path: Path) -> None:
    inner = ScriptedProvider(["one", "two", "three"])
    p = CachedProvider(inner, ResponseCache(tmp_path))
    a = p.complete("sys", "user", max_tokens=10, salt="q|7d|base|0")
    b = p.complete("sys", "user", max_tokens=10, salt="q|7d|base|0")
    c = p.complete("sys", "user", max_tokens=10, salt="q|1d|base|0")
    assert (a.text, a.cached) == ("one", False)
    assert (b.text, b.cached) == ("one", True)
    assert c.text == "two"
    assert len(inner.calls) == 2

    # Same provider identity (name, model, options), new process: replies come from disk.
    again = CachedProvider(ScriptedProvider(["other"]), ResponseCache(tmp_path))
    assert again.complete("sys", "user", max_tokens=10, salt="q|1d|base|0").text == "two"


def test_key_depends_on_every_part() -> None:
    parts = ["p", "m", {"effort": "low"}, 10, "s", "u", "salt"]
    base = cache_key(*parts)
    for i, alt in enumerate(["p2", "m2", {"effort": "high"}, 11, "s2", "u2", "salt2"]):
        changed = list(parts)
        changed[i] = alt
        assert cache_key(*changed) != base
    assert cache_key(*parts) == base


def test_concurrent_puts(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)

    def worker(i: int) -> None:
        for _ in range(20):
            cache.put("ab" + "0" * 62, {"text": str(i)})
            cache.get("ab" + "0" * 62)  # may miss mid-replace on Windows, must not raise

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.get("ab" + "0" * 62)["text"] in {"0", "1", "2", "3"}
    assert not list(tmp_path.rglob(".tmp-*"))
    assert cache.get("cd" + "0" * 62) is None
