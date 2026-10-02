# M3b design: part model, fusion, calibration, ONNX and model card

- **Status:** Draft for review
- **Date:** 2026-10-02
- **Parent spec:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) §12, §13, §20
- **Follows:** [M3a](2026-10-02-m3a-damage-model-design.md) (damage model, training, MLflow).

## 1. Why

M3a gives the pipeline its own damage model, which says *what* the damage is. Two problems remain.
First, severity still comes from how much of the **photo** the damage fills, so a close-up of a
small scratch looks "severe". Second, the R6 threshold (0.40) is a guess, made for a different
model. M3b adds a part model, so damage can be measured against the **part** it sits on, and
calibration, so confidence numbers mean what they say.

## 2. Goals and success criteria

| Goal | Measure | Target |
|---|---|---|
| A usable part model | Test mask mAP50 of `parts-yolo11n-v1` (22 classes) | Reported; ≥ 0.50 expected |
| Fusion works | Share of approved `fusion-eval-v1` parts whose group the part model predicts at the same place (mask IoU ≥ 0.5) | Reported |
| Calibrated confidence | ECE of the damage champion on validation, before and after temperature scaling | After < before |
| Evidence-based threshold | Recommended R6 threshold, with its golden-set what-if row | Reported (recommendation only) |
| Fast portable inference | ONNX export of both champions; CPU ms per image, `.pt` vs `.onnx` | Recorded |
| Safety holds | Escalation recall on golden v1 with damage + parts + fusion + calibration | **= 1.00** (must) |
| Honest documentation | `docs/model-card.md` for both models | Written |

**Out of scope:**
- Changing `min_finding_confidence` in `config/decision_policy.toml`. Calibration only recommends a
  value (decided with the owner); a change is a separate PR.
- RT-DETR (stretch, dropped).
- Training more damage models.

## 3. Decisions already made

- **Part model:** one run `parts-yolo11n-v1` on `parts-v1` with all **22 classes**, trained on
  Kaggle exactly like M3a. Classes map to the 9 part groups (`config/taxonomy.toml`) only at fusion
  time.
- **Task-aware registry:** `claimlens-damage` and `claimlens-parts` are separate registered
  models. `config/models.toml` gains a `[parts]` champion.
- **Fusion:**
  - Each damage instance goes to the part instance with the largest
    intersection ÷ damage area, measured on rasterised masks.
  - `part_area_ratio` = damage area ÷ part area.
  - A damage instance with no overlapping part keeps `part = None` and the image-fraction size.
- **Severity:** the new rate card version `rate-card-2026.10-v1` adds part-ratio bands (minor below
  0.10, moderate below 0.30, otherwise severe). The image-fraction bands remain as the fallback.
  Prices stay fictional and unchanged.
- **Calibration:**
  - One temperature `T` for the damage champion, fitted on **validation** predictions matched to
    ground truth (same class, mask IoU ≥ 0.5) by minimising negative log-likelihood over a fixed
    grid.
  - `T` is stored in `config/models.toml`, and the detector reports `sigmoid(logit(p) / T)`.
  - ECE uses 10 equal-width bins.
- **Threshold recommendation:** the lowest threshold on a 0.05 grid where at least 80% of the kept
  calibrated validation predictions are correct.
- **ONNX:** Ultralytics export (opset default, static 640) with `onnx` and `onnxruntime` added to
  the `vision` group. The detectors accept `.pt` or `.onnx`.

## 4. Architecture

```
photo ─► damage model (.pt/.onnx) ─► instances (class, conf, box, polygon) ─► calibrate conf (T)
     └─► part model   (.pt/.onnx) ─► instances (class, conf, box, polygon) ─► part group
                                   fuse: largest intersection / damage area
                                          ▼
                 DamageFinding(type, conf, box, image_area_fraction, part, part_area_ratio)
                                          ▼
                 pricing: severity from part_area_ratio (fallback: image fraction)
```

`FusedDetector` implements the existing `Detector` protocol, so the workflow, the event log and the
agent do not change.

## 5. Components

### 5.1 Segmentation instances (`claimlens.vision.instances`)

- `SegInstance(label: str, confidence: float, box_xyxy, polygon_xyn: tuple[float, ...])`.
- The `Segmenter` protocol: `model_version` and `segment(image_path) -> tuple[list[SegInstance], width, height]`.
- `UltralyticsSegmenter(weights)`: `.pt` or `.onnx`. It is the only Ultralytics-touching code here
  and is coverage-excluded. `YoloSegDetector` from M3a becomes a thin adapter over it.

### 5.2 Mask geometry (`claimlens.vision.masks`)

`rasterize(polygon_xyn, size=256) -> numpy bool array` (PIL `ImageDraw`), plus `mask_iou`,
`intersection_over(a, b)` and `area`. Masks are compared on a fixed 256 × 256 grid, which is
accurate to about 0.4% of the image side. That is enough for assigning parts and setting severity.

### 5.3 Fusion (`claimlens.fusion`)

`fuse(damage: Sequence[SegInstance], parts: Sequence[SegInstance], groups: PartGroups, *, min_overlap=0.10) -> list[FusedDamage]`.
It picks the part with the highest `intersection / damage_area` at or above `min_overlap`, and
computes `part_area_ratio`. `FusedDetector(damage, parts, groups, calibrator)` implements
`Detector`.

`DamageFinding` gains `part_area_ratio: float | None = None` (the field `part` already exists).
Old events stay valid because the new field has a default.

### 5.4 Pricing (`claimlens.pricing`)

`RateCard` gains an optional `part_bands` (`minor_max_ratio`, `moderate_max_ratio`). The function
`severity_for(finding, card)` uses `part_area_ratio` when the finding has a part and the card has
part bands; otherwise it uses the image fraction. `config/rate_card.toml` becomes version
`rate-card-2026.10-v1` with the same prices.

### 5.5 Calibration (`claimlens.training.calibration`)

- `match_predictions(preds, truths, iou=0.5) -> list[tuple[float, bool]]` (confidence, correct),
  matched greedily by confidence, same class.
- `fit_temperature(pairs) -> float` (grid 0.25–4.0, step 0.05, minimum NLL).
- `ece(pairs, bins=10) -> float`.
- `recommend_threshold(pairs, T, precision=0.80) -> float | None`.
- `claimlens train calibrate <run>` segments the validation images on the CPU (814 photos, a few
  minutes for the n model), reads the ground-truth polygons from the YOLO label files, and writes
  the temperature and threshold to `config/models.toml` and `reports/models/<run>-calibration.json`.

### 5.6 ONNX (`claimlens train export <run>`)

This exports `models/<task>/<run>/best.onnx`. `claimlens train benchmark <run> --format onnx`
records `cpu_ms_per_image_onnx`. ONNX files are covered by the same `dvc add` as the weights.

### 5.7 Training and registry changes

- `TrainingRun` gains `task: Literal["damage", "parts"]` (default `damage`).
- `config/training.toml` gains `[runs.parts-yolo11n-v1]` with `task = "parts"` and
  `dataset = "parts-v1"`.
- `import_run` registers `claimlens-<task>` and writes to `models/<task>/`.
- `select` writes the champion for the given task. `--task` defaults to `damage`.
- `scripts/make_train_bundle.py` takes the dataset name.
- The Kaggle job folder `training/kaggle/train-parts/` mirrors `train/` with its own `RUNS` and
  dataset.

### 5.8 CLI

`--detector fused` becomes the default when both champions exist: damage + parts + fusion +
calibration. `legacy` and `yolo-seg` remain available.

## 6. Evaluation outputs

- `evals/reports/<date>-parts-model-v1.md`: test metrics per class, plus the `fusion-eval-v1`
  agreement per part group.
- `reports/models/<run>-calibration.json` and a calibration section in the damage model report:
  T, ECE before and after, the recommended threshold, and precision and recall at it.
- `evals/reports/<date>-triage-baseline-v1-fused.md`: golden v1 with the fused detector, compared
  with legacy and M3a, and a what-if table including the recommended threshold.
- `docs/model-card.md`.

## 7. Error handling

| Situation | Behaviour |
|---|---|
| No part found under a damage instance | `part = None`; severity falls back to the image fraction |
| Several parts overlap the damage | The one with the largest intersection ÷ damage area wins; ties go to the smaller part |
| Polygon with fewer than 3 points | Zero area; such a damage instance cannot be fused and keeps image-fraction size |
| Temperature fit on no matched predictions | `calibrate` refuses with a clear error |
| `.onnx` file given but `onnxruntime` missing | Clear message naming `uv sync --group vision` |
| `fused` requested without a parts champion | Clear error naming `claimlens train select --task parts` |

## 8. Testing

- Unit tests, no models:
  - rasterising and IoU on known shapes;
  - fusion on synthetic squares (inside, straddling, none, ties);
  - severity from the part ratio and the fallback;
  - matching, temperature fit (recovers a known T on synthetic data), ECE, threshold;
  - `FusedDetector` with fake segmenters;
  - task-aware import, select and models config;
  - CLI wiring.
- An integration test, skipped without weights, runs the fused detector on one photo.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Part model is weak on close-ups (as the M2b zero-shot model was) | Fallback to image fraction; fusion agreement is reported per group |
| Changing the severity rule raises fast-tracks wrongly | Escalation recall gate = 1.00; the report shows severity counts before and after |
| Calibration on validation over-fits a single number | One parameter only; the reliability table on validation is reported, and the test split is not used for fitting |
| ONNX numbers differ slightly from `.pt` | Report both; the pipeline default stays `.pt` unless the ONNX test metrics match within 0.01 |
| Kaggle GPU quota (two trainings in one day) | The n model only; about 1–1.5 h |
