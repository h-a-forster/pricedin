# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-11

### Added

- Venue collectors for Kalshi and Polymarket with leakage-safe snapshot rules: horizons are
  measured from the listing-time scheduled end, prices come only from data at or before the
  snapshot time, and early closes are recorded for audit but never used.
- Point-in-time news from the Wikipedia Current events portal, restricted to items dated
  before each snapshot.
- Forecast providers `claude-cli` and `anthropic`, with an on-disk response cache and a
  versioned prompt (`PROMPT_VERSION`).
- Evaluation: logistic pooling of the model forecast with the market price, a
  forecast-encompassing test with cluster-robust errors, an event-clustered bootstrap and Holm
  correction, all scored out of sample.
- P&L of trading on the model's disagreement, after venue fees.
- Power analysis for planning sample size.
- Report and figures.
- Study v1 (`studies/v1.toml`): 712 Polymarket questions, Claude Opus, Sonnet and Haiku 5.5,
  frozen data and results. No measurable gain over the market price.
- CLI: `collect`, `snapshots`, `news`, `forecast`, `merge`, `evaluate`, `report`, `power`,
  `live`, `status`. `collect --resume` continues an interrupted Kalshi scan.
