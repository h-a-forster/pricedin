# Related work

Most LLM forecasting benchmarks score a model against the outcome, or against a crowd or base
rate. Few ask whether the model adds anything to the *market price at the same moment*, and
fewer still control for the market's own known biases. The closest work is listed first.

## LLMs against or with market prices

- **AIA Forecaster** (arXiv [2511.07678](https://arxiv.org/abs/2511.07678)). On liquid
  markets the LLM alone loses to the market (Brier 0.108 vs 0.098), but a weighted mix of the
  two beats either. This is the same pooling question pricedin asks, with a closed system.
- **Prophet Arena** (arXiv [2510.17638](https://arxiv.org/abs/2510.17638)). 1,367 events.
  The best model's Brier is 0.184 against the market's 0.187, with overlapping intervals.
  The models are better calibrated than the market but lose money: return per unit bet is
  0.943 at best. Markets absorb news faster close to resolution.
- **KalshiBench** (arXiv [2512.16030](https://arxiv.org/abs/2512.16030)). 300 Kalshi
  questions resolving after the models' cutoffs. Every model is overconfident (ECE 0.12 to
  0.40). The baseline is the base rate, not the market.
- **Halawi et al. 2024**, "Approaching human-level forecasting with language models"
  (arXiv [2402.18563](https://arxiv.org/abs/2402.18563)). A retrieval system scores Brier
  0.179 against the crowd's 0.149. Averaged with the crowd it improves on the crowd slightly
  (0.146).
- **Schoenegger et al. 2024**, "Wisdom of the silicon crowd"
  (arXiv [2402.19379](https://arxiv.org/abs/2402.19379)). An average of 12 LLMs matches 925
  humans on 31 questions. Simply averaging human and model forecasts beats showing the model
  the human median.
- **ForecastBench** (arXiv [2409.19839](https://arxiv.org/abs/2409.19839)). Superforecasters
  still lead the best LLMs, and the gap is narrowing. On market questions, models that are
  shown the market price mostly do best by copying it.

## Leakage

- **Paleka et al., "Pitfalls in evaluating language model forecasters"** (arXiv
  [2506.00723](https://arxiv.org/abs/2506.00723)). Lists the ways temporal leakage inflates
  backtested forecasting scores. pricedin's design responds to it:
  - questions are listed after every model's training cutoff;
  - the model gets no tools;
  - news is read as of the snapshot time;
  - horizons hang on a deadline known at listing, never on the actual close.

## Market efficiency and biases

- **Bürgi, Deng and Whelan 2025**, "Makers and Takers: The Economics of the Kalshi Prediction
  Market" ([PDF](https://www.karlwhelan.com/Papers/Kalshi.pdf)). Over 300,000 Kalshi
  contracts.
  - Prices sharpen towards close but show a clear favourite-longshot bias.
  - Contracts under 10¢ lose over 60%.
  - This bias is why pricedin's benchmark is the *recalibrated* market, not the raw price.

## Methods

- **Forecast encompassing.**
  - Chong and Hendry 1986, "Econometric evaluation of linear macro-economic models".
  - Fair and Shiller 1990, "Comparing information in forecasts from econometric models".
  - Both test whether one forecast holds all the information in another.
- **Logit pooling and extremizing.**
  - Satopää et al. 2014, "Combining multiple probability predictions using a simple logit
    model" (*IJF*).
  - Baron et al. 2014, "Two reasons to make aggregated probability forecasts more extreme"
    (*Decision Analysis*).
- **Murphy 1973.** "A new vector partition of the probability score". Splits the Brier score
  into reliability, resolution and uncertainty.
- **Diebold and Mariano 1995.** "Comparing predictive accuracy". Questions in one event are
  correlated, so pricedin clusters by event.
- **Cameron, Gelbach and Miller 2008.** "Bootstrap-based improvements for inference with
  clustered errors". Supports the event-clustered bootstrap and the CR1 standard errors.
