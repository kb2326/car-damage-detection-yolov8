# M3a design: our own damage segmentation model

- **Status:** Draft for review
- **Date:** 2026-10-02
- **Parent spec:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) §12, §13, §20
- **Follows:** M2 (data engine). **Followed by:** M3b (part model, fusion, calibration, ONNX,
  model card).

## 1. Why

The claim pipeline still uses the legacy course model. On golden v1 it gets 4 of 30 fast-tracks
right. Most misses come from two rules: R5 (the estimate is too high because the model
over-sizes damage) and R6 (the model is unsure, with confidence below 0.40). M3a replaces it with a
damage segmentation model trained on `damage-v1`, the deduplicated CarDD set built in M2.

## 2. Goals and success criteria

| Goal | Measure | Target |
|---|---|---|
| A good damage model | Test **mask mAP50** of the chosen run (6 classes) | ≥ 0.50 (spec §20) |
| Safety holds | Escalation recall on golden v1 | **= 1.00** (must) |
| The system gets more useful | Correct fast-tracks on golden v1 | > 4 of 30 |
| Runs fast enough on a laptop | Median CPU time per image | Recorded, informs n vs s |
| Every run is reproducible | A run can be traced to its commit, dataset hash and config | Every imported run |

A model below 0.50 does not block M3a. The shortfall is reported and the retro decides the next
step. Escalation recall below 1.00 does block it.

**Out of scope (M3b):** part model, fusion, severity from damage-to-part ratio, confidence
calibration, ONNX export, model card, RT-DETR comparison. **Out of scope (M3a):** changing the
decision policy or the rate card.

## 3. Decisions already made

- **Training on Kaggle** (free T4 GPU) with the pattern proven in M2b: private dataset, private
  script kernel, pinned commit, outputs pulled with `kaggle kernels output`.
- **Local experiment logbook:** MLflow with a SQLite backend in `var/mlflow/` (git-ignored). Runs
  are imported after training; no hosted tracking service.
- **Two candidates:** `yolo11n-seg` and `yolo11s-seg`, same data and settings.
- **Selection on validation only:** the chosen run is the one with the highest validation mask
  mAP50. Test numbers are reported for both but never used to choose.
- **Weights stay private:** `models/damage/` (DVC, local remote) and the private Kaggle output.
  Never committed and never published (CarDD terms, ADR 0004).
- **Policy unchanged:** `min_finding_confidence` stays 0.40 in M3a; the report adds a what-if
  table for other thresholds without changing the policy.

## 4. Architecture

```
damage-v1 (DVC) --zip--> private Kaggle dataset "claimlens-damage-v1" (+ dataset.json with hash)
                                  |
config/training.toml --> Kaggle kernel (pinned commit) --> claimlens train run <run> (per run)
                                  |  train, validate each epoch, test once at the end
                                  v
             var/runs/<run>/: manifest.json, metrics.json, results.csv, plots, weights/best.pt
                                  |  kaggle kernels output
                                  v
claimlens train import <run> --> MLflow (var/mlflow) + registered model "claimlens-damage"
                                  |  weights -> models/damage/<run>/best.pt (dvc add)
claimlens train select   ------> config/models.toml: damage = "<run>" ; MLflow alias "champion"
                                  v
YoloSegDetector (claim pipeline) --> eval-triage on golden v1
```

## 5. Components

### 5.1 `config/training.toml`

```toml
[defaults]
dataset = "damage-v1"
epochs = 100
patience = 20        # early stop after 20 epochs without validation improvement
imgsz = 640
batch = 16
seed = 20261002
workers = 2

[runs.damage-yolo11n-v1]
model = "yolo11n-seg.pt"

[runs.damage-yolo11s-v1]
model = "yolo11s-seg.pt"
```

Loaded into a frozen Pydantic `TrainingRun` (defaults merged per run). Unknown keys or run names
are errors.

### 5.2 `claimlens.training`

- `config.py`: `TrainingRun`, `load_training_runs(path)`.
- `manifest.py`: `RunManifest` (run name, commit, dataset name and DVC md5, config, start and end
  time, `status: complete | incomplete`, Ultralytics version), and `RunMetrics` (validation and
  test box and mask mAP50 and mAP50-95, per-class mask mAP50, best epoch, epochs run).
- `run.py`: `run_training(run, *, dataset_dir, expected_md5, out_dir, trainer)`. Checks that the
  md5 recorded in the bundle's `dataset.json` equals the md5 for `data/processed/damage-v1` in
  `dvc.lock` at the pinned commit (so a stale bundle cannot be trained against newer code), calls the trainer, evaluates the test split once, and writes the manifest, metrics and
  artifacts. `trainer` is a protocol; the real one wraps Ultralytics (`ultralytics_trainer.py`,
  excluded from coverage like the other model adapters), and tests use a fake.
- `import_run.py`: `import_run(run_dir, *, tracking_uri, models_dir, allow_incomplete)`. Validates
  the manifest, logs params, per-epoch metrics from `results.csv`, final metrics, plots and
  weights to MLflow, registers a version of `claimlens-damage`, and copies the weights to
  `models/damage/<run>/best.pt`. Re-importing the same run is a no-op (idempotent by run name and
  commit).
- `select.py`: `select_champion(runs)` picks the highest validation mask mAP50, writes
  `config/models.toml`, and sets the MLflow alias `champion`.
- `benchmark.py`: median CPU milliseconds per image over 20 fixed test images.

Requires a new `training` dependency group locally (`mlflow`); the Kaggle job uses `vision`.

### 5.3 Kaggle job: `training/kaggle/train/`

`run_train.py` plus `kernel-metadata.json` (private, GPU, internet). The script pins `COMMIT`,
unpacks the dataset bundle, installs the locked `vision` group, runs `claimlens train run` for each
run in `RUNS` in turn, and copies `var/runs/<run>/` to `/kaggle/working`. If time runs out, the
last saved best checkpoint is kept and the manifest says `incomplete`. `scripts/make_train_bundle.py`
zips `data/processed/damage-v1` with a `dataset.json` holding its DVC md5.

### 5.4 `YoloSegDetector` (`claimlens.vision.yolo_seg`)

Implements `Detector`. For each predicted instance it builds a `DamageFinding` with:
- the box, confidence and class (via `normalize_class_name`, so unknown classes raise);
- `image_area_fraction` from the **mask polygon area** (shoelace formula on normalised
  coordinates), not the box. The pure function `polygon_area_fraction` is unit-tested.

`model_version` is `yolo11-seg:<run>`. Missing weights raise at construction. Requires `vision`.

### 5.5 CLI

- `claimlens train run <run> --dataset-dir … --out …` (used inside the Kaggle job)
- `claimlens train import <run> --from var/kaggle/output/<run> [--allow-incomplete]`
- `claimlens train select`
- `claimlens train benchmark <run>`
- `run`, `resume` and `eval-triage` gain `--detector legacy|yolo-seg` (default `legacy` until
  `select` has run, then the champion in `config/models.toml`). `eval-triage` gains
  `--what-if 0.25,0.40,0.55` (thresholds for the what-if table only).

### 5.6 Data and DVC

- `models/damage/<run>/` tracked with `dvc add` and pushed to the local remote.
- No DVC stage for training itself (it runs on Kaggle). Reproducibility comes from the manifest:
  commit, dataset md5 and config.

## 6. Evaluation outputs

- `evals/reports/<date>-damage-model-v1.md`: n vs s validation and test mask and box mAP50 and
  mAP50-95, per-class test mask mAP50, best epoch, training time, CPU ms per image, and the chosen
  run with the reason.
- `evals/reports/<date>-triage-baseline-v1-yolo11.md`: golden v1 with the champion detector,
  compared with the legacy baseline, plus a what-if table at confidence thresholds 0.25, 0.40
  and 0.55.

## 7. Error handling

| Situation | Behaviour |
|---|---|
| Bundle `dataset.json` md5 differs from `dvc.lock` at the pinned commit | `train run` refuses to start |
| Unknown run name or bad config | Clear error, exit code 1 |
| Manifest missing, or run name or commit mismatch | `train import` refuses |
| Incomplete run | `train import` refuses unless `--allow-incomplete`; tagged `incomplete` in MLflow |
| Missing weights or unknown class at inference | `YoloSegDetector` raises (same as legacy) |
| `mlflow` or `ultralytics` not installed | Clear message naming the `uv sync --group` to run |

## 8. Testing

- Unit tests:
  - config loading and merging;
  - manifest and metrics validation;
  - `run_training` with a fake trainer (hash check, outputs written, incomplete status);
  - `import_run` into a temporary SQLite MLflow (params, metrics, registration, idempotency);
  - `select_champion`;
  - `polygon_area_fraction`;
  - CLI wiring.
- One integration test, skipped without weights, runs `YoloSegDetector` on one photo.
- CI never needs a GPU, data or weights. Coverage stays at 95% or above (model adapters excluded).

## 9. Risks

| Risk | Mitigation |
|---|---|
| Kaggle session limit (12 h) or weekly GPU quota (about 30 h) | n is about 1 h and s about 2 h on a T4; early stopping; one kernel trains both in sequence |
| Mask area makes most damage "minor", so prices drop and fast-tracks rise wrongly | Escalation recall must stay 1.00; the rate card is unchanged and the report shows severity counts |
| Confidence scores of the new model differ from the legacy one | Policy unchanged in M3a; the what-if table shows the effect; calibration in M3b |
| Ultralytics downloads base weights at train time | Kaggle job has internet; the base weight name is recorded in the manifest |
| Weights leak publicly | Private Kaggle kernel and outputs; `models/` git-ignored; reviewed in the PR |
