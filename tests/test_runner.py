from __future__ import annotations

from pathlib import Path

import pytest

from pricedin.forecast.prompt import PROMPT_VERSION, build_prompt, prompt_hash
from pricedin.forecast.providers import ProviderError, ScriptedProvider
from pricedin.forecast.providers.base import Completion
from pricedin.forecast.runner import load_forecasts, run_forecasts, shard_of
from pricedin.schema import Forecast, NewsContext, NewsItem, Question, Snapshot, read_jsonl

SPEC = "constant:0.3"


def make_question(i: int) -> Question:
    return Question(
        id=f"kalshi:Q{i}",
        venue="kalshi",
        event_id=f"E{i}",
        title=f"Question {i}?",
        rules="Resolves YES if it happens.",
        category="other",
        created_at="2026-08-01T00:00:00Z",
        open_at="2026-08-01T00:00:00Z",
        close_at="2026-08-22T12:00:00Z",
        scheduled_close_at="2026-08-22T12:00:00Z",
        outcome=i % 2,
        volume=100.0,
    )


AS_OF = {"7d": "2026-08-15T12:00:00Z", "1d": "2026-08-21T12:00:00Z", "1h": "2026-08-22T11:00:00Z"}


def make_data(n: int = 3, horizons=("7d", "1d", "1h")):
    qs = [make_question(i) for i in range(n)]
    snaps = [
        Snapshot(question_id=q.id, horizon=h, as_of=AS_OF[h], mid=0.6) for q in qs for h in horizons
    ]
    return qs, snaps


def rows(path: Path) -> list[Forecast]:
    return list(read_jsonl(path, Forecast))


def test_basic_run_and_resume(tmp_path: Path) -> None:
    qs, snaps = make_data()
    out = tmp_path / "f.jsonl"
    seen: list[int] = []
    s1 = run_forecasts(
        qs,
        snaps,
        provider_spec=SPEC,
        out_path=out,
        cache_dir=tmp_path / "cache",
        progress=lambda d, t, f: seen.append(d),
    )
    assert s1["done"] == 9 and s1["failed"] == 0 and s1["cached"] == 0
    assert sorted(seen) == list(range(1, 10))
    got = rows(out)
    assert len(got) == 9
    f = got[0]
    assert f.prob == 0.3 and f.forecaster == SPEC and f.condition == "base"
    assert f.prompt_version == PROMPT_VERSION and f.error is None
    assert f.model == "constant-0.3" and f.harness == "constant"
    q = next(q for q in qs if q.id == f.question_id)
    system, user, _ = build_prompt(q, f.as_of, "base")
    assert f.prompt_hash == prompt_hash(system, user)
    assert len({r.prompt_hash for r in got}) == 9  # every task saw a different prompt
    assert {(r.question_id, r.horizon, r.as_of) for r in got} == {
        (s.question_id, s.horizon, s.as_of) for s in snaps
    }

    s2 = run_forecasts(qs, snaps, provider_spec=SPEC, out_path=out, cache_dir=tmp_path / "cache")
    assert s2["tasks"] == 0 and s2["skipped_existing"] == 9
    assert len(rows(out)) == 9


def test_cache_makes_rerun_free(tmp_path: Path) -> None:
    qs, snaps = make_data(2, ("7d",))
    prov = ScriptedProvider("PROBABILITY: 0.8")
    kw = dict(provider_spec="echo", provider=prov, cache_dir=tmp_path / "c")
    run_forecasts(qs, snaps, out_path=tmp_path / "a.jsonl", **kw)
    s = run_forecasts(qs, snaps, out_path=tmp_path / "b.jsonl", **kw)
    assert len(prov.calls) == 2 and s["cached"] == 2 and s["cost_usd"] == 0.0
    a, b = rows(tmp_path / "a.jsonl"), rows(tmp_path / "b.jsonl")
    assert [(r.harness, r.model, r.prompt_hash) for r in a] == [
        (r.harness, r.model, r.prompt_hash) for r in b
    ]
    assert all(r.harness == "scripted" for r in b)


def test_failures_recorded_then_retried(tmp_path: Path) -> None:
    qs, snaps = make_data(2, ("7d",))
    out = tmp_path / "f.jsonl"
    bad = ScriptedProvider("I cannot say.")
    s1 = run_forecasts(
        qs, snaps, provider_spec="echo", provider=bad, out_path=out, cache_dir=None, retries=1
    )
    assert s1["failed"] == 2 and s1["done"] == 0
    assert len(bad.calls) == 4  # one retry each
    assert all(r.prob is None and "unparseable" in r.error for r in rows(out))
    assert rows(out)[0].rationale == "I cannot say."

    good = ScriptedProvider("ok\nPROBABILITY: 0.55")
    s2 = run_forecasts(qs, snaps, provider_spec="echo", provider=good, out_path=out, cache_dir=None)
    assert s2["done"] == 2
    assert len(rows(out)) == 4  # failed rows kept, new rows appended
    latest = load_forecasts(out)
    assert {f.prob for f in latest.values()} == {0.55}


def test_retry_salt_differs(tmp_path: Path) -> None:
    qs, snaps = make_data(1, ("7d",))
    prov = ScriptedProvider(["no number", "PROBABILITY: 0.2"])
    s = run_forecasts(
        qs, snaps, provider_spec="echo", provider=prov, out_path=tmp_path / "f.jsonl",
        cache_dir=tmp_path / "c", retries=1,
    )  # fmt: skip
    assert s["done"] == 1 and rows(tmp_path / "f.jsonl")[0].prob == 0.2


def test_fatal_error_aborts(tmp_path: Path) -> None:
    class Broken:
        name, model, options = "broken", "m", {}

        def __init__(self) -> None:
            self.n = 0

        def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
            self.n += 1
            raise ProviderError("not logged in")

    qs, snaps = make_data(3)
    prov = Broken()
    out = tmp_path / "f.jsonl"
    s = run_forecasts(
        qs, snaps, provider_spec="echo", provider=prov, out_path=out, cache_dir=None, workers=1
    )
    assert s["aborted"] == "not logged in"
    assert prov.n == 1 and s["failed"] == 1 and s["not_run"] == 8
    assert rows(out)[0].error == "not logged in"


def test_sharding_partitions_questions(tmp_path: Path) -> None:
    qs, snaps = make_data(12)
    seen: dict[str, int] = {}
    total = 0
    for i in range(3):
        out = tmp_path / f"s{i}.jsonl"
        s = run_forecasts(qs, snaps, provider_spec=SPEC, out_path=out, cache_dir=None, shard=(i, 3))
        total += s["done"]
        for r in rows(out):
            assert seen.setdefault(r.question_id, i) == i  # all horizons in one shard
            assert shard_of(r.question_id, 3) == i
    assert total == 36 and len(seen) == 12
    with pytest.raises(ValueError):
        run_forecasts(qs, snaps, provider_spec=SPEC, out_path=tmp_path / "x", shard=(3, 3))


def test_limit_and_snapshot_filters(tmp_path: Path) -> None:
    qs, snaps = make_data(3)
    snaps.append(Snapshot(question_id="kalshi:UNKNOWN", horizon="7d", as_of=AS_OF["7d"], mid=0.5))
    snaps.append(
        Snapshot(question_id=qs[0].id, horizon="late", as_of="2026-08-23T00:00:00Z", mid=0.5)
    )
    out = tmp_path / "f.jsonl"
    s = run_forecasts(qs, snaps, provider_spec=SPEC, out_path=out, cache_dir=None, limit=4)
    assert s["done"] == 4 and s["invalid_as_of"] == 1 and s["eligible"] == 9
    # The limit takes tasks in a deterministic order (question id, then 7d, 1d, 1h); workers
    # may finish, and so write, in any order.
    assert {(r.question_id, r.horizon) for r in rows(out)} == {
        ("kalshi:Q0", "7d"),
        ("kalshi:Q0", "1d"),
        ("kalshi:Q0", "1h"),
        ("kalshi:Q1", "7d"),
    }


def test_mixed_forecaster_file_rejected(tmp_path: Path) -> None:
    qs, snaps = make_data(1, ("7d",))
    out = tmp_path / "f.jsonl"
    run_forecasts(qs, snaps, provider_spec=SPEC, out_path=out, cache_dir=None)
    with pytest.raises(ValueError):
        run_forecasts(qs, snaps, provider_spec="constant:0.9", out_path=out, cache_dir=None)
    with pytest.raises(ValueError):
        run_forecasts(qs, snaps, provider_spec=SPEC, condition="news", out_path=out)


def test_news_condition(tmp_path: Path) -> None:
    qs, snaps = make_data(2, ("7d",))
    news = [
        NewsContext(
            question_id=qs[0].id,
            horizon="7d",
            as_of=AS_OF["7d"],
            query="q",
            items=[
                NewsItem(title="Old news", seen_at="2026-08-10T00:00:00Z", domain="a.com"),
                NewsItem(title="Leak", seen_at="2026-08-16T00:00:00Z", domain="b.com"),
            ],
        )
    ]
    prov = ScriptedProvider("PROBABILITY: 0.5")
    out = tmp_path / "n.jsonl"
    s = run_forecasts(
        qs, snaps, news, provider_spec="echo", provider=prov, condition="news",
        out_path=out, cache_dir=None, workers=1,
    )  # fmt: skip
    assert s["done"] == 2 and s["news_missing"] == 1
    users = {u.splitlines()[0]: u for _, u in prov.calls}
    assert (
        "Old news" in users["Question: Question 0?"]
        and "Leak" not in users["Question: Question 0?"]
    )
    assert "found no headlines" in users["Question: Question 1?"]
    by_q = {r.question_id: r for r in rows(out)}
    assert by_q[qs[0].id].usage["news_items"] == 1
    assert by_q[qs[1].id].usage["news_items"] == 0
