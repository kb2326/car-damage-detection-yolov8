# M3 retrospective: vision models (M3a + M3b)

- **Dates:** 2026-10-02
- **Plans:** [`m3a-damage-model`](../plans/2026-10-02-m3a-damage-model.md),
  [`m3b-parts-fusion`](../plans/2026-10-02-m3b-parts-fusion.md)

## What we shipped

- **M3a: our own damage model.**
  - Kaggle GPU training pinned to a commit, with a dataset fingerprint check.
  - A local MLflow logbook and registry; champion chosen on validation.
  - `YoloSegDetector` in the pipeline (ADR 0008).
- **M3b: the rest of the vision stack.**
  - A part model, and fusion that says which part each damage is on.
  - Part-ratio severity and temperature calibration with a recommended threshold.
  - ONNX export, plus a model card (ADR 0009).
- Both trainings (damage n and s, then parts) ran **at the same time** on Kaggle while the code
  for the next steps was written.
- 305 tests, 97% line coverage, strict typing; the integration tests run the real models locally.

## Numbers

| | Val mask mAP50 | Test mask mAP50 | CPU ms (`.pt`) |
|---|---|---|---|
| damage-yolo11n-v1 | 0.719 | 0.735 | 132 |
| **damage-yolo11s-v1 (champion)** | **0.738** | **0.733** | 357 |
| **parts-yolo11n-v1 (champion)** | 0.753 | **0.746** | 193 |

The course model had val mAP50 0.137.

| Golden v1 (97 claims) | Route accuracy | Escalation recall | Correct fast-tracks |
|---|---|---|---|
| Legacy | 0.73 | 1.00 | 4 of 30 |
| Damage model (M3a) | 0.80 | 1.00 | 11 of 30 |
| Fused (M3b) | 0.80 | 1.00 | 11 of 30 |

- **Calibration:** T = 0.80, ECE 0.059 → 0.040, recommended threshold 0.65 (not applied).
- **Fusion:** 88% of golden findings get a part; 112 of 400 severities change (77 up, 35 down).
- **Part agreement** with reviewed parts on CarDD close-ups: 61 of 133 (46%).

## What went well

- **The safety gate earned its keep twice.** Sizing damage by its outline looked like an
  improvement but fast-tracked claims that needed a person (escalation recall 0.97, then 0.99).
  Both times the gate stopped the change, the cause was found in minutes, and the fix was to
  follow the rate card's own definition (box area).
- **Running two GPU jobs in parallel** saved more than an hour.
- **A one-epoch CPU smoke test before the GPU run** caught nothing, but would have caught an
  Ultralytics API mistake for the cost of seconds instead of a wasted Kaggle session.
- **Selecting on validation mattered:** `n` scored marginally higher on test, but `s` won on
  validation, and the rule held.

## What was harder than expected

- **The new models did not reduce R5 or R6 misses as much as hoped.** 19 fast-track cases still
  go to review. With damage-only it was 10 for price, 8 for low confidence and 1 for agent
  uncertainty; with fusion it is 12, 6 and 1, because part-relative pricing moved two cases from
  R6 to R5.
- **ONNX was slower than PyTorch on this CPU,** the opposite of the usual claim, so it stays an
  export.
- **The part model struggles on close-ups** (46% agreement), so part-relative pricing is only
  partly trustworthy on CarDD-style photos.
- **Merging M3a into M3b** needed hand-resolved conflicts in the CLI and the tracking code, because
  both branches changed the same functions.
- **Syncing dependency groups breaks `cv2` again** (`opencv-python` and `opencv-python-headless`
  share a folder). A reinstall fixes it each time.

## What we will change next

- **Decide on the R6 threshold:** 0.65 costs 2 fast-tracks on golden v1.
- **Look at the remaining R5 misses with real prices:** the fictional rate card may be too strict
  for minor damage.
- **Get a human review of the golden claims and a spot-check of `fusion-eval-v1`** before quoting
  these numbers outside the project.
- **M4:** wrap the detectors as MCP tools for the agent.
