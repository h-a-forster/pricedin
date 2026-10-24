# Running the prospective arm

`studies/live.toml` collects forecasts on markets that are still open: each pass takes a live
order book at 7 d / 1 d / 1 h before a market's scheduled close, forecasts it at that moment and
records the outcome once the market settles. Nothing runs unless you start it.

A pass is idempotent and safe to repeat; run one roughly every hour (the horizon windows are
+-35 min, `tolerance_minutes` in the study file). A horizon that gets no pass inside its window
is logged as missed and never backfilled.

## Locally

```sh
uv run pricedin live studies/live.toml --once --dry-run   # snapshot only, no model calls
uv run pricedin live studies/live.toml --once             # forecasters from studies/live.toml
```

Schedule the second command hourly with cron or Windows Task Scheduler. The forecasters listed
in `studies/live.toml` use the `claude-cli` provider (local Claude Code login).

## On GitHub Actions

`.github/workflows/live.yml` runs one pass with the `anthropic:` provider and commits
`data/live` back to the repository. It has no schedule by default, so it never queues runs
that do nothing. To enable it:

1. Add an `ANTHROPIC_API_KEY` repository secret.
2. Set the repository variable `PRICEDIN_LIVE` to `true` (Settings > Secrets and variables >
   Actions). Without it every run of the job is skipped.
3. Try one manual run first: Actions > live > Run workflow, with `dry_run` ticked.
4. Add the hourly trigger under `on:` in the workflow:

   ```yaml
   on:
     schedule:
       - cron: "7 * * * *" # hourly, off the top of the hour
     workflow_dispatch:
       # ...unchanged
   ```

GitHub delays scheduled runs (often 5-30 min) and may drop them under load, so expect some
missed horizons. Scheduled workflows also stop after 60 days without repository activity on
public repositories; GitHub warns by email first, and they can be re-enabled from the Actions tab.

Each pass makes at most `max_forecasts_per_pass` model calls (60 by default), so budget for up
to 24 times that per day.

## Provenance

Every forecast row records the provider spec (`forecaster`), the model id the endpoint
reported (`model`), the client version (`harness`, e.g. `claude-cli 2.1.287` or
`anthropic-sdk 0.71.0`), the prompt template version (`prompt_version`) and a hash of the exact
prompt sent (`prompt_hash`), so a later model or client upgrade shows up in the data.
