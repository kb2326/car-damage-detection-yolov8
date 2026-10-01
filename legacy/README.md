# Legacy: original course project (frozen)

This folder preserves the original STAT 5350 project, a YOLOv8n car-damage detector with a
Streamlit UI and FastAPI service. It is kept **unchanged** for comparison and is excluded from
linting, type checks and CI. Nothing in `src/claimlens` imports from here.

| Path | Contents |
|---|---|
| `app/` | detector class, Streamlit UI, FastAPI app, config, utils, requirements |
| `notebooks/01_train_yolov8n_colab.ipynb` | Colab training run (YOLOv8n, 50 epochs, T4) |
| `deploy/` | original Dockerfile and docker-compose |
| `docs/course-report.docx` | course write-up |
| `docs/yolo-training-notes.txt` | study notes (describes anchors/objectness, which YOLOv8 does not use) |

Model weights moved to `models/legacy/yolov8n-cardamage-v6.pt`; the dataset moved to
`data/raw/roboflow-car-damage-v6/`. Neither is tracked in git.

## Audit (2026-10-01)

Findings that motivated the rebuild:

1. **Label mismatch (critical).** The trained weights contain 7 classes
   (`crack, dent, glass shatter, lamp broken, scratch, tire flat, smash`), but the app maps
   predictions onto a hard-coded list of 17 different classes. Every displayed label is wrong:
   a predicted *dent* is shown as "Front-Windscreen-Damage" and rated **High** severity.
2. **Weak model.** Validation mAP50 = 0.137, mAP50-95 = 0.062, precision 0.15, recall 0.25.
   No test-split evaluation was run.
3. **Small, imbalanced data.** 180 training images; *tire flat* has 2 instances; *crack*,
   *lamp broken* and *tire flat* are absent from validation and test. 12 validation images lack
   labels and 12 test labels lack images. Polygon labels were trained as boxes only.
4. **RGB/BGR swap.** The UI and API pass RGB arrays to Ultralytics, which expects BGR.
5. **Engineering issues.** The API converts its own 400 errors into 500s; it shares a mutable
   threshold across concurrent requests and blocks the event loop with synchronous inference;
   the UI reloads the model on every interaction; confidence values below 0.25 have no effect;
   the Dockerfile lacks `libgl1`; weights are not in the repo, so a fresh clone cannot run.
6. **Unsupported claims.** "17 damage types", "production ready" and "80% cost reduction" are not
   backed by evidence. The report cites YOLOv3's paper as "YOLOv8".
