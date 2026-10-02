# ADR 0008: Training on Kaggle with a local experiment logbook

- **Status:** Accepted
- **Date:** 2026-10-02

## Context

M3 trains our own segmentation models: YOLO11-seg for damage (M3a) and for car parts (M3b). This
laptop has no NVIDIA GPU, and long CPU model runs overheat it (M2b). Results must be
reproducible: anyone should be able to say which code, data and settings produced a given set of
weights. The weights are trained on CarDD, whose terms allow only non-commercial use without
redistribution (ADR 0004), so they cannot be published.

## Decision

- **Train on a free Kaggle GPU (T4) in private kernels.**
  - Each job (`training/kaggle/train/`, `training/kaggle/train-parts/`) checks out a **pinned commit**.
  - It installs the locked `vision` group with uv.
  - It runs `claimlens train run <run>`.
  - It writes each run straight to `/kaggle/working/<run>/`, so finished work survives a time-out.
  - Laptop GPUs and hosted training services were not chosen.
- **Ship the data as a private Kaggle dataset with a fingerprint.**
  - `scripts/make_train_bundle.py <dataset>` zips the DVC-built export together with
    `dataset.json`, which holds the md5 DVC recorded for it.
  - `train run` refuses a bundle whose md5 differs from `dvc.lock` at the pinned commit, so a
    stale bundle cannot be trained against newer code.
- **Define runs as configuration.** `config/training.toml` holds a `[defaults]` table plus one table
  per run (model, dataset, task, epochs, early-stopping patience, image size, batch, seed). Every
  finished run leaves:
  - a `manifest.json` (commit, dataset md5, config, trainer version, status);
  - a `metrics.json` (validation and test mask and box mAP).
- **Keep a local experiment logbook.** `claimlens train import` logs a downloaded run into MLflow,
  with a SQLite backend in `var/mlflow/` (git-ignored). It records parameters, per-epoch metrics,
  final metrics, plots and weights, and registers a version of `claimlens-<task>`. Re-importing is a
  no-op. A hosted tracker (DagsHub) was considered and declined, to keep CarDD-derived weights off
  another service.
- **Choose the champion on validation only.**
  - `claimlens train select` picks the run with the highest **validation** mask mAP50.
  - It writes `config/models.toml` and sets the MLflow alias `champion`.
  - Test metrics are reported for every run but never used to choose, so the test split stays an
    honest final check.
- **Keep weights private.**
  - Weights live in `models/<task>/<run>/` (tracked with `dvc add`, local remote), in the MLflow
    artifact store and in the private Kaggle output.
  - They never go to git, Hugging Face or a public Kaggle item.
- **Measure damage size from the mask, not the box.** `YoloSegDetector` computes
  `image_area_fraction` from the predicted polygon (shoelace formula), because a box around a
  diagonal scratch is mostly undamaged panel.

## Consequences

- **What it costs:**
  - A run appears in the logbook only after it is downloaded and imported; there is no live view
    while Kaggle trains.
  - A training job sees only the pinned commit. Code changed later needs a new pin and a new
    kernel version.
  - Weights have one local copy plus the Kaggle output until an off-site DVC remote exists; the
    backup item on the roadmap still stands.
- **What it gives:**
  - Two jobs (damage and parts) can train at the same time on Kaggle while code is written locally.
  - Every model in the registry can be traced to a commit, a dataset fingerprint and a config.
- **What changes in M3b:** calibration and the R6 threshold are handled in M3b (ADR 0009). In M3a
  the decision policy is unchanged; the golden report shows a what-if table instead.
