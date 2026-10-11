# pricedin

[![CI](https://github.com/h-a-forster/pricedin/actions/workflows/ci.yml/badge.svg)](https://github.com/h-a-forster/pricedin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

What does an LLM add to a prediction-market price from its own knowledge, before it does any
research? A leakage-safe baseline on Polymarket.

## Result

**No. Without research, Claude adds nothing measurable to the Polymarket price.**

712 Polymarket questions, 1,743 snapshots. Three Claude models forecast each one without
seeing the price. Fitted on questions that resolved before September 2026, scored on later
ones (828 test rows per model):

| | 7 days before | 1 day before | 1 hour before |
| --- | ---: | ---: | ---: |
| Market price | 0.141 | 0.088 | 0.070 |
| Market, recalibrated | 0.142 | 0.085 | 0.067 |
| Claude Opus 5.5 alone | 0.223 | 0.209 | 0.204 |
| Market + Opus (pool) | 0.140 | 0.085 | 0.067 |

Brier score on the test period, lower is better. 0.25 is a coin flip.

![Brier by horizon](results/v1/figures/brier_by_horizon.png)

- **The model alone is far worse than the market.** Its Brier is 0.20–0.23 at every
  horizon. The market sharpens from 0.14 a week out to 0.07 an hour out; the model barely
  moves.
- **Pooling adds nothing.** Market + LLM beats the recalibrated market by at most 0.0013
  Brier, inside the noise. None of the 18 pre-registered tests rejects after Holm adjustment
  (smallest adjusted p = 0.94). The weight on the LLM is indistinguishable from zero for
  Opus, Sonnet and Haiku alike.
- **Light news does not help.** 252 snapshots had relevant dated headlines. Adding them
  made the pool slightly *worse* at every horizon (small samples, wide intervals). The
  market had already priced the news.
- **No money in it.** Trading the gap between the pool and the market, after fees, earns no
  more than trading the recalibrated market alone. The LLM on its own loses money.

![Out-of-sample gain](results/v1/figures/forest.png)

The full report, with calibration, ablations, clustering checks, per-category results and
P&L, is in [results/v1/report.md](results/v1/report.md).

## The question

A market price is already a probability. "Fed cuts in September" at 80¢ means 80%. Ask an LLM
the same question at the same moment, without showing it the price. Does its answer tell you
anything the price does not?

This is a *forecast-encompassing* test. It is not "is the LLM as accurate as the market?". A
less accurate forecast can still add information, and an equally accurate one can add none.

## What this is, and what it is not

In practice nobody trades on an LLM's unaided guess. A useful forecaster would research the
question online first. In a backtest it cannot: a web search today returns pages written after
the question resolved. So this study measures three narrower things:

- **A lower bound.** What the model adds from its training knowledge and reasoning alone
  (`base`), or with light, time-stamped news headlines (`news`).
- **Where value comes from.** If `base` adds nothing and `news` does, the value is in
  retrieval, not in the model.
- **A leak-proof harness.** Snapshots, prompts, fees and statistics that a stronger
  forecaster, such as a research agent run prospectively, can be plugged into.

A null result here does not show that LLM forecasters cannot beat markets. It shows that an
unresearched one does not.

## Design

1. **Questions.** 712 binary Polymarket markets listed from 1 July 2026, after the June 2026
   training cutoff of every model used, and resolved by 9 October 2026. At most one per event,
   at most 100 per category, at least $10,000 traded.
2. **Snapshots.** The market mid at 7 days, 1 day and 1 hour before the deadline known at
   listing.
3. **Forecasts.** Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5 (low effort) give a probability.
   They see the question, its rules and the date of the snapshot. They never see the price.
4. **Pooling.** `logit(y) = a + b·logit(market) + c·logit(LLM)` is fitted on questions that
   resolved early and scored on those that resolved later.
5. **Tests.**
   - H1: `c ≠ 0` (cluster-robust Wald test).
   - H2: the pool beats the recalibrated market on Brier score out of sample (one-sided,
     event-clustered bootstrap).
   - Both are Holm-adjusted together.
6. **Money.** A simple strategy trades the gap between the pool and the market, after
   Polymarket's fee formula.

The full plan, fixed before any forecast was scored, is in
[docs/preregistration.md](docs/preregistration.md).

### Leakage controls

A backtest is only as good as its guard against look-ahead.

- **Cutoffs.** Every question was created after every model's training cutoff (June 2026,
  per Anthropic's model overview). Outcomes come after creation.
- **No tools.** The model runs with no tools, no web access, no MCP servers and one turn.
- **Listing-time anchors.** Horizons hang on the deadline known at listing, never on the
  actual close. An early close, such as a "by <date>" market resolving YES, depends on the
  outcome.
- **Closed markets.** There is no snapshot if the market had already closed. A real-time
  observer would not have one either.
- **Dated news.** Headlines are read from Wikipedia revisions dated before each snapshot.
  The collector refuses to write anything later.
- **Empirical check.** Where the market sat between 20% and 80% and the model was ≥90% sure either way,
  the model is right about half the time (134 of 262 calls, 51%; see
  `scripts/leakage_check.py`). A model that knew outcomes would be right almost
  always.

## Quickstart

Install from GitHub:

```bash
uv tool install git+https://github.com/h-a-forster/pricedin
```

Re-score the frozen study with no network access:

```bash
git clone https://github.com/h-a-forster/pricedin && cd pricedin
uv run pricedin evaluate studies/v1.toml
uv run --extra plot pricedin report studies/v1.toml
```

Rebuild it from scratch:

```bash
uv run pricedin collect studies/v1.toml      # list and sample questions
uv run pricedin snapshots studies/v1.toml    # market state at each horizon
uv run pricedin news studies/v1.toml         # dated headlines
uv run pricedin forecast studies/v1.toml -f claude-cli:claude-opus-5-5@low -c base
uv run pricedin evaluate studies/v1.toml && uv run --extra plot pricedin report studies/v1.toml
```

`claude-cli:` runs the forecasts through a local Claude Code install. `anthropic:` uses the
API instead (`pip install "pricedin[anthropic]"` and an `ANTHROPIC_API_KEY`).

## Limitations

- **No research.** This is the point of the design, and the main reason the result is a
  lower bound.
- **Claude only.** Three Claude models at low effort. Other model families may differ.
- **One window, one venue.** Polymarket questions resolved July to October 2026.
- **Kalshi left out.** Kalshi has no reliable deadline field fixed at listing, so its snapshot
  times could not be anchored without look-ahead. The Kalshi client stays in the code; see the
  amendments in the [analysis plan](docs/preregistration.md).
- **Today's date.** Claude Code tells the model the real current date, and this cannot be
  switched off. The prompt tells the model to ignore it, and the date says nothing about any
  outcome. The `anthropic:` provider and the live collector avoid it.
- **Volume filter.** Lifetime volume selects which markets are sampled, and it includes
  trading after the snapshot. It shapes the sample, not the comparison.
- **Clipping.** Probabilities are clipped to [0.01, 0.99] before any logit. Markets priced
  below 1¢ or above 99¢ are treated as 1¢ and 99¢.
- **Polymarket P&L.** Polymarket has no historical order book, so its P&L uses the mid and
  is approximate.

## Repository

| path | content |
| --- | --- |
| `src/pricedin/venues/` | Polymarket and Kalshi clients: listing, snapshots, fees |
| `src/pricedin/news/` | dated Wikipedia headlines and relevance filter |
| `src/pricedin/forecast/` | prompt, providers, resumable runner |
| `src/pricedin/eval/` | pooling, scoring, inference, P&L |
| `studies/v1.toml` | the study config |
| `data/v1/` | frozen questions, snapshots, news and forecasts ([data card](docs/data.md)) |
| `results/v1/` | `evaluation.json`, `report.md`, figures |
| `docs/` | [analysis plan](docs/preregistration.md), [data card](docs/data.md), [related work](docs/related-work.md) |

## Citation

See [CITATION.cff](CITATION.cff). MIT licence.
