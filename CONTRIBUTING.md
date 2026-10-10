# Contributing

Bug reports, new venues, providers and forecasts for new models are welcome.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
uv sync --all-extras --dev
uv run pricedin --help
```

## Checks

```sh
uv run pytest -m "not slow and not network"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Test rules:

- No network and no API keys in the default tests. Use fixtures in `tests/fixtures` and the
  scripted provider.
- Use `tmp_path` for every file.
- Must pass on Windows: `pathlib`, forward slashes in code, no shell assumptions.
- Mark anything over a few seconds with `@pytest.mark.slow`, and anything that needs the
  internet with `@pytest.mark.network`.

## Adding a venue

A venue is a module in `src/pricedin/venues/` (see `kalshi.py` and `polymarket.py`) with three
functions:

- `list_candidates(...)` returns resolved `Question` objects that pass the eligibility rules.
- `snapshot(question, horizon, ...)` returns the market price as of `close_at - horizon`, or
  `None` if the market had stopped trading by then. It may only use data at or before that time.
- `live_snapshot(...)` returns the current price and order book for open questions.

Field semantics matter for leakage:

- `close_at` is the scheduled end as listed while the market was open. It is never the actual
  close, so horizons cannot depend on how the market ended. Early closes are what leak.
- Keep the actual close in `extra` (for example `extra["actual_close"]`), along with anything
  else needed for audit.
- `outcome` is the resolution (1 = YES, 0 = NO). It must not reach prompts or snapshots.
- `event_id` groups markets that share news and outcomes. It is the bootstrap cluster.

Register the venue in the CLI, add fee terms in `eval/fees.py`, and test with recorded
responses. Describe the close-time rules in the module docstring, as the existing venues do.

## Adding a forecaster or provider

Providers live in `src/pricedin/forecast/providers/` and implement
`complete(system, user, *, max_tokens) -> Completion` (see `base.py`). Register the spec prefix
in `providers/__init__.py`. Spec strings look like `claude-cli:<model>@effort` or
`anthropic:<model>`.

1. Read API keys from environment variables only. Never log or store them.
2. Return raw text, token usage and cost in the `Completion`.
3. Raise the existing error types so retries and fatal failures behave the same way.
4. Put the SDK behind an optional extra in `pyproject.toml`.
5. Test with a fake client. No network in tests.

## Guidelines

- Changing any prompt wording in `forecast/prompt.py` bumps `PROMPT_VERSION`. The version is
  stored with each forecast.
- Published study numbers never change silently. A change to the study design, filters,
  horizons or scoring is a new study version (`studies/v2.toml`), not an edit to `v1`.
- Do not change scoring or pooling (`eval/`) without an issue first.
- Match the surrounding style; ruff enforces the rest (line length 100).

## Submitting results for a new model

Open an issue with the "Result submission" template. Include:

- The forecast JSONL file for the study, not just a summary.
- Study name and version, for example `v1`.
- Exact model id, provider spec string, and every setting that affects output: reasoning
  effort, max tokens, prompt version.
- The pricedin version or commit, and the commands you ran.

Forecast files contain model rationales. Remove API keys and private paths first. Results from
a modified study or prompt are labelled as such.

## Pull requests

One change per pull request, with tests and docs. Add a line under "Unreleased" in
[CHANGELOG.md](CHANGELOG.md) for user-visible changes.

Security issues: see [SECURITY.md](SECURITY.md).
