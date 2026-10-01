# Contributing

## Workflow

1. Branch from `main`: `feat/…`, `fix/…`, `chore/…`, `docs/…`, `exp/…` (experiments).
2. Keep pull requests small and focused on one change.
3. CI must pass: lint, format, type check, tests (and evals, from M5).
4. Significant technical decisions get an ADR in `docs/adr/`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): `feat: add fusion of damage and part masks`.

## Local checks

```bash
uv run pre-commit run --all-files
uv run mypy
uv run pytest
```

## Rules

- Never commit data, model weights, or secrets (`.env`). Use DVC and the model registry.
- Raw data is immutable. Write derived data to `data/interim/` or `data/processed/`.
- New behaviour needs tests; new model or agent behaviour needs an eval.
