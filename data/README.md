# Data

Datasets live here on disk but are **never committed to git**. They are versioned with
[DVC](https://dvc.org): git holds only the `.dvc` pointer files, `dvc.yaml` and `dvc.lock`.
Full documentation of contents, terms and cleaning: [`docs/data-card.md`](../docs/data-card.md).

## Layout

| Folder | Rule |
|---|---|
| `raw/` | Exactly as downloaded, with a `MANIFEST.json` of checksums. **Immutable**: never edit in place. |
| `interim/` | Records, near-duplicate clusters and split assignments. Rebuilt by `dvc repro`. |
| `processed/` | YOLO segmentation datasets ready for training. Rebuilt by `dvc repro`. |

## Workflow

```bash
# First time on a machine (needs ROBOFLOW_API_KEY in .env for the CarDD copy)
uv run claimlens data fetch cardd-roboflow-v6
uv run claimlens data fetch carparts-seg

# Or, with a shared DVC remote configured
uv run dvc pull

# Rebuild every dataset whose inputs changed, then inspect the metrics
uv run dvc repro
uv run dvc metrics show
```

## Sources

| Folder | What it is | Used for | Terms |
|---|---|---|---|
| `raw/cardd-roboflow-v6/` | Roboflow copy of **CarDD**: 4,000 images, 6 damage classes, COCO polygons | `damage-v1` | CarDD terms: non-commercial research and education; do not redistribute |
| `raw/carparts-seg/` | Roboflow `car-seg` via Ultralytics: 3,833 images, 23 part classes | `parts-v1` | CC BY 4.0 |
| `raw/legacy-course-subset/` | The 339-image course-project folder (7 classes incl. `smash`, mixed sources) | golden claims only | CC BY 4.0, mixed |

Why the CarDD terms apply to the Roboflow copy: [ADR 0004](../docs/adr/0004-data-sources-and-licensing.md).
