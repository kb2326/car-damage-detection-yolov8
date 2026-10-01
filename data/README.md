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

### `raw/roboflow-car-damage-v6/`

- **Source:** [Roboflow Universe: car-damage-detection-vyhvw, v6](https://universe.roboflow.com/auto-industry/car-damage-detection-vyhvw/dataset/6)
- **License:** CC BY 4.0 (attribution required)
- **Contents:** 339 images (train 180 / valid 111 / test 48), 7 classes, polygon labels in YOLO format
- **Known issues:** see the audit in [`legacy/README.md`](../legacy/README.md)

### CarDD (planned, M2)

Wang et al., *CarDD: A New Dataset for Vision-based Car Damage Detection*, IEEE T-ITS 2023.
License terms must be confirmed before use or redistribution of derived models.
