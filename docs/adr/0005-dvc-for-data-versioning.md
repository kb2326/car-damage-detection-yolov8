# ADR 0005: DVC for data versioning

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

Datasets are hundreds of megabytes, cannot go into git, and must be reproducible: a model's
results only mean something if the exact data behind them can be rebuilt.

## Decision

- Track immutable raw snapshots with `dvc add` (`data/raw/*.dvc`).
- Define the build as DVC stages in `dvc.yaml` (`build-damage-v1`, `build-parts-v1`); `dvc.lock`
  records the hash of every input and output.
- Publish dataset statistics as DVC metrics (`reports/data/*-stats.json`), committed to git.
- Default remote: a local folder beside the repository (`../claimlens-dvc-remote`).
- Off-site option: DagsHub (free 10 GB, also hosts MLflow for M3):
  `uv run dvc remote add -d dagshub https://dagshub.com/<user>/claimlens.dvc`.
- Not chosen: the Google Drive remote, currently blocked by Google's app verification policy.

## Consequences

- `uv run dvc repro` rebuilds only what changed; `uv run dvc pull` restores data on a new machine
  once a shared remote is configured.
- The local remote is not a real backup; moving to DagsHub before M3 training is recommended.
