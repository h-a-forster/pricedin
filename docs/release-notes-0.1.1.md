# pricedin 0.1.1

Maintenance release. Study v1 data and results are unchanged: `pricedin evaluate
studies/v1.toml` reproduces the committed `results/v1/evaluation.json` exactly apart from its
`version` field, and `pricedin report` reproduces the committed `summary.json` and figures
exactly and `report.md` apart from its version line.

## Added

- Forecast provenance on every new row: `model`, `harness` and `prompt_hash`.
- [docs/live.md](live.md): running the prospective arm locally or on GitHub Actions.

## Changed

- The `live` workflow runs only on manual dispatch until a schedule is added.

## Fixed

- Polymarket and Kalshi collectors skip malformed prices, book levels and candles instead of
  using them or failing.
- HTTP 408 is retried; the final error names the last status and URL.
- Response cache: concurrent writes of the same key no longer raise on Windows.
- README: two rounded Brier scores and the size of the Holm family (36 tests).
- `pricedin live` on Polymarket: `as_of` is now when the book was read, not when it last changed.
  The old value could be close to an hour stale, which pushed snapshots out of the forecast
  window so they were logged as missed. No live data has been collected, so nothing is affected.
- Polymarket live books: a one-sided book takes its price from the last trade only if the trade
  agrees with the side that exists, and otherwise is logged as no usable book. `/book` can
  report the complement token's trade, such as 0.99 against a 0.29 ask.
- `pricedin evaluate` refuses to run while forecast files are waiting for `pricedin merge`. It
  used to skip them and write results without those rows. The README and the data card now
  include the merge step.
- README: the news bullet says what the numbers show (the gap reaches 0.105 Brier, not
  "slightly"), the sports and crypto subgroups are disclosed, `pip install "pricedin[anthropic]"` is gone
  (the package is not on PyPI), and the data licences are separated from the code's.
- CI and `live`: actions pinned to commit SHAs; `uv sync --locked`, so a stale lockfile fails.

The earlier entries are in [CHANGELOG.md](../CHANGELOG.md).
