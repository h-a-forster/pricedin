## What and why

<!-- What this changes and why. Link the issue if there is one. -->

## Checklist

- [ ] `uv run pytest -m "not slow and not network"` passes
- [ ] `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` pass
- [ ] Tests added or updated (offline, no API keys, Windows-safe)
- [ ] Documentation updated if behaviour or options changed
- [ ] Entry added under "Unreleased" in `CHANGELOG.md` for user-visible changes
- [ ] `PROMPT_VERSION` bumped if prompt wording changed
- [ ] Published study numbers unchanged, or a new study version is added
- [ ] No leakage: nothing after the snapshot time reaches prompts or prices
