# Models

Model weights are stored here locally and are **not committed to git**. Trained weights are
tracked with DVC (`models/damage.dvc`, local remote) and registered in a local MLflow logbook
(`var/mlflow`). Because they are trained on CarDD (non-commercial terms, ADR 0004), they are
**private**: never published to Hugging Face, a public Kaggle item, or git.

| File | Description |
|---|---|
| `legacy/yolov8n-cardamage-v6.pt` | Original course model. YOLOv8n, 7 classes, trained 2024-12-11. Val mAP50 0.137. Baseline only. |
| `damage/<run>/best.pt` | M3a YOLO11-seg damage models; the champion is named in `config/models.toml`. |
