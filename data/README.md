# Data

Datasets live here on disk but are **never committed to git**. From M2 they are versioned with
[DVC](https://dvc.org); until then, re-download sources from the links below.

## Layout (convention)

| Folder | Rule |
|---|---|
| `raw/` | Exactly as downloaded. **Immutable**: never edit files in place. |
| `interim/` | Intermediate outputs (deduplicated, relabelled, converted). Reproducible from `raw/`. |
| `processed/` | Final training-ready splits. Reproducible from `raw/` via `training/` pipelines. |

## Sources

### `raw/legacy-course-subset/`

- **What it is:** the 339-image folder used by the original course project. It is **not** Roboflow
  v6 (which has 4,000 images and 6 classes); it has 7 classes including `smash` and mixes CarDD
  images with other Roboflow Universe sources.
- **Use:** golden claims only (`evals/golden/v0`), never training.
- **Terms:** Roboflow Universe exports, CC BY 4.0, mixed sources.
- **Known issues:** see the audit in [`legacy/README.md`](../legacy/README.md)

### CarDD (planned, M2)

Wang et al., *CarDD: A New Dataset for Vision-based Car Damage Detection*, IEEE T-ITS 2023.
License terms must be confirmed before use or redistribution of derived models.
