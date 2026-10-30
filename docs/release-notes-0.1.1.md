# pricedin 0.1.1

Maintenance release. Study v1 data and results are unchanged: `pricedin evaluate
studies/v1.toml` reproduces the committed `results/v1/evaluation.json` exactly.

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

Full list in [CHANGELOG.md](../CHANGELOG.md).
