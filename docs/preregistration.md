# Analysis plan (study v1)

Fixed and committed before any forecast was scored. The config is
[`studies/v1.toml`](../studies/v1.toml); the git history shows this file predates every file
under `data/v1/forecasts/`.

## Question

Does an LLM's probability carry information about the outcome that the market price at the
same moment does not?

This is a *forecast-encompassing* question (Chong and Hendry 1986; Fair and Shiller 1990). It
differs from "is the LLM as accurate as the market?". An LLM can be less accurate and still
add information, and an equally accurate LLM can add none.

## Data

- **Questions.** Binary Kalshi and Polymarket markets listed on or after 2026-07-01 and
  closed before 2026-10-09. All four models have a June 2026 training-data cutoff. No model
  can have seen these outcomes.
- **Eligibility.** The market resolved cleanly YES or NO. It is not a combo. It traded for at
  least 8 days. Lifetime volume is at least 1,000 contracts (Kalshi) or $10,000 (Polymarket).
- **Sample.** At most one question per event. At most 100 per (venue, category). Selection is
  by a stable hash of the question id, which makes it a deterministic function of the
  candidate list.
- **Reference close.** The earlier of the actual trading close and the scheduled end. A
  snapshot is taken at 7 days, 1 day and 1 hour before it.
- **Market price.** The last observation at or before the snapshot time, and at most 6 hours
  old.
  - Kalshi: the mid of the 1-minute candle's best bid and ask.
  - Polymarket: the CLOB price history.
- **News.** Items from Wikipedia's daily "Current events" pages for the 14 days before the
  snapshot, read at the last revision at or before the snapshot time. They are ranked for
  relevance by BM25, top 25.

## Forecasts

- **Models.** Claude Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.1, run through Claude Code
  headless mode (`claude -p`). Each call is one turn with no tools, so no web search. Claude
  Code's system prompt is replaced by ours. Prompt `v1` is in
  [`src/pricedin/forecast/prompt.py`](../src/pricedin/forecast/prompt.py).
- **What the model never sees.** The market price, the volume, the actual close time, or
  anything dated after the snapshot.
- **Conditions.**
  - `base`: the question and its rules only.
  - `news`: also the dated headlines.

Coverage depends on budget. Opus 5.5 runs both conditions at every horizon. The other models
run a subset, reported as such.

## Models compared

All are fitted on the training period and scored on the test period:

| name | definition |
| --- | --- |
| `market` | market mid, unchanged |
| `market_recal` | logistic recalibration of the market mid: σ(a + b·logit m) |
| `llm`, `llm_recal`, `llm_extremized` | the LLM alone: raw, recalibrated, extremized σ(k·logit p) |
| `average` | σ((logit m + logit p) / 2) |
| `pool` | σ(a + b·logit m + c·logit p) |

`market_recal` is the benchmark. Prediction markets show a favourite-longshot bias, and a
pool that only fixed that bias would look like LLM skill.

- **Clipping.** Probabilities are clipped to [0.01, 0.99] before any logit.
- **Split.** The training period has questions closing before the median close time. The
  test period has snapshots taken at or after it. Rows whose snapshot is before the split
  but whose close is after it are dropped as an embargo.

## Hypotheses

The primary family is Holm-adjusted across horizons × forecasters × conditions.

- **H1, encompassing.** In logit(y) = a + b·logit(m) + c·logit(p), test c = 0. This uses a
  Wald test with cluster-robust (CR1) errors clustered by event, on the full sample.
- **H2, out-of-sample gain.** Δ = Brier(`market_recal`) − Brier(`pool`) on the test period.
  The 95% CI and p-value come from an event-clustered bootstrap with 2,000 resamples.

## Secondary analyses

- Δ against the raw `market`, and log-score versions of H2.
- Breakdowns by horizon, venue and category. Cells with fewer than 30 rows are flagged.
- The **contested** subset, where the market mid is in [0.05, 0.95] at the snapshot.
- Questions with no early close.
- `news` vs `base`, which is the retrieval ablation. Extremizing and recalibration are also
  ablations.
- A walk-forward version with monthly folds.
- **P&L after fees.**
  - Kalshi uses its quadratic taker fee per order on candle bid/ask.
  - Polymarket uses the market's own fee schedule and an assumed spread, labelled as
    approximate.
  - The rule is to trade 100 contracts when the expected edge after fees exceeds 2¢.
  - The P&L is compared with the same strategy on `market_recal`.

## Known limitations

- **Retrospective design.** Market prices come from history, and the LLM is asked "as of" a
  past date. The model cannot know the outcomes. Claude Code still adds the real current
  date to its context, and this cannot be turned off. The prompt tells the model to ignore
  it. A prospective collector (`pricedin live`) removes this caveat for future studies.
- **Volume filter.** Volume is measured over the market's whole life, so it uses
  information from after the snapshot. This affects which questions are sampled, not how
  the market and the LLM compare on them.
- **Polymarket quotes.** Polymarket has no historical bid/ask, so its P&L is approximate.
