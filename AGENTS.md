# AGENTS.md

Guide for coding agents (and humans) working on pricedin. `README.md` states the question and
the v1 result; `docs/preregistration.md` is the analysis plan; `CONTRIBUTING.md` has the venue,
provider and result-submission rules this file points to.

## What this is

A leakage-safe harness and study: does an LLM's own knowledge, with no web research, add
information to a prediction-market price? Study v1 (712 Polymarket questions, Claude Opus,
Sonnet and Haiku 5.5) finds no measurable gain. Data and results are frozen in `data/v1/` and
`results/v1/`.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11, typed code, `ruff check` + `ruff format`
  (line length 100), pytest. Must pass on Windows, macOS and Linux.
- Leakage rules are the point of the project: `close_at` is the listing-time scheduled end,
  never the actual close; snapshots and news use only data at or before the snapshot time;
  `outcome` never reaches a prompt or snapshot. See CONTRIBUTING "Adding a venue".
- Published numbers never change silently. A change to design, filters, horizons or scoring
  is a new study (`studies/v2.toml`), not an edit to `v1`. Any prompt wording change bumps
  `PROMPT_VERSION`.
- Every headline number in `README.md` must trace to `results/v1/report.md` or
  `results/v1/summary.json`.
- Model calls cost money: forecasting goes through the on-disk response cache and the
  resumable runner; never re-run a study's forecasts to "check" something. Re-scoring
  (`pricedin evaluate`) is free and offline.
- API keys come from environment variables only and are never printed, logged or stored.
- Scratch files go in `tmp/` (gitignored). No `.bak` copies; use git.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The repository
  is public: change it through pull requests with CI. No personal names, local user paths or
  machine details in committed files or logs.

## How to test

```sh
uv sync --all-extras --dev
uv run pytest -m "not slow and not network"
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pricedin evaluate studies/v1.toml --out runs/ci --bootstrap 200   # CI re-score, ~25 s
```

A full re-score (`uv run pricedin evaluate studies/v1.toml --out tmp/eval`, about 2 min)
reproduces `results/v1/evaluation.json` exactly; run it after any change under `src/pricedin/eval/`.

## Status and next steps (2026-10-30)

Released: v0.1.0 (tagged). v0.1.1 is prepared on a branch (CHANGELOG, release notes in
`docs/release-notes-0.1.1.md`); the tag is not created yet.

Open items:

- **Fable arm.** Claude Fable 5.1 was dropped from v1 for cost (about $200 via the API; see
  the preregistration amendments). Adding it means forecasting the v1 questions with a new
  forecaster spec and re-scoring, reported as an addition to v1, not a change to the
  published table.
- **Prospective arm.** `studies/live.toml` and `.github/workflows/live.yml` work but run only
  on manual dispatch; no live data has been collected (`docs/live.md`).
- **Kalshi.** Excluded from v1 (no deadline fixed at listing). The client remains; a Kalshi
  study needs a look-ahead-free anchor first.
- **CI hygiene.** GitHub Actions are pinned by tag, not commit SHA; Dependabot has open
  branches for `actions/checkout`, `astral-sh/setup-uv`, `anthropic` and `ruff`.
