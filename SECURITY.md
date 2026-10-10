# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Report privately through the repository's **Security** tab ("Report a vulnerability").
Do not open a public issue.

Include the pricedin version, your OS, what goes wrong, and how to reproduce it. Expect an
acknowledgement within a week.

## Threat model

pricedin is a local command-line tool that fetches public market data and sends prompts to a
model provider you choose.

- **API keys come from the environment only.** pricedin reads them from environment variables,
  never from config or data files, and never logs or stores them. Do not paste keys into issues
  or result submissions.
- **The response cache holds prompts and raw replies.** `.pricedin-cache/` stores the full
  prompt (question, rules, news) and the model's raw output. Treat it like any other data from
  your provider account, and check it before sharing.
- **The HTTP cache holds public market data only.** It stores responses from the venue,
  Wikipedia and GDELT public APIs, with no credentials.
- **Forecast files hold model rationales.** Forecast JSONL includes the model's reasoning text.
  Check it before attaching it to an issue.
- **Network calls go to a fixed set of hosts:** the Kalshi and Polymarket public APIs,
  Wikipedia, GDELT, and the model provider you select.
- **Provider SDKs are optional extras** and are not audited by this project.
