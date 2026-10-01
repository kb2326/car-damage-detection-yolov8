# ADR 0002: Repository structure and tooling

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

The original course project kept scripts, data, weights, notebooks and reports in a flat root with
`requirements.txt` and no tests or CI. Dataset images and weights were git-ignored, so the
repository could not be reproduced from a clone.

## Decision

- **Keep the repository and its history**; move the original project into `legacy/`, frozen.
- **`src/` layout** (`src/claimlens`), so tests run against the installed package, not stray files.
- **uv + `pyproject.toml` + `uv.lock`**, Python 3.12, for fast reproducible environments.
- **ruff** (lint + format), **mypy --strict**, **pytest + coverage**, **pre-commit**.
- **GitHub Actions CI** running the same checks on every push and pull request.
- **Data and weights out of git**: `data/` (raw / interim / processed) versioned with DVC from M2;
  weights in `models/` and an MLflow registry from M3.

## Consequences

- The original code no longer runs from the root; it stays available in `legacy/` for reference.
- Contributors need `uv` installed.
- Until DVC is set up in M2, the dataset must be re-downloaded from Roboflow on a new machine.
