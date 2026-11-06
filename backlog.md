# Backlog

## In progress

## Waiting on me

- v0.1.1 is released: open a PR from `polish-2026-10-30`, let CI pass, merge, tag, and set the release date in `CHANGELOG.md` and `CITATION.cff` to the tag date.
- Dependabot PRs #8 to #11 are reviewed and closed out; `anthropic` 2.3.0 (#10) is a major bump, so check its changes before merging.
- Stale remote branches are gone: `claude/bump-actions`, `upgrades-2026-10-23`, and the Dependabot branches for checkout 7, setup-uv 7 and 10.3.1, anthropic 1.17.0 and ruff 0.17.1.

## Decided

- 2026-10-10: Claude Fable 5.1 stays out of v1 for cost; adding it later is an addition to v1, not a change to the published table.
- 2026-10-10: Kalshi stays out of v1 because no listing-time field gives a reliable deadline; the client stays in the code.
- 2026-10-23: the `live` workflow runs on manual dispatch only until the prospective arm is enabled.

## Proposed

- Fable arm forecasted on the v1 questions and scored as an addition (about $120 at the amendments' $0.06 per call for 1,995 calls; the amendments quote $200).
- Prospective arm collecting live data (`docs/live.md`), so the study has a research-agent comparison free of the date caveat.
- Sports and crypto subgroups tested as a pre-registered v2 question, with per-category recalibration as the benchmark.
- Polymarket live rows priced when the book is one-sided, from `/prices-history` as in v1, instead of being lost as no usable book.
- CI fails when a re-score moves the committed v1 point estimates.
- Kalshi study built on a look-ahead-free deadline anchor.
