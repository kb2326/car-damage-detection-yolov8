"""Ultralytics YOLO11-seg training and evaluation. Requires `uv sync --group vision` and a GPU.

Excluded from coverage: it is exercised by the Kaggle job, not by CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claimlens.training.config import TrainingRun
from claimlens.training.manifest import SplitMetrics


class UltralyticsTrainer:
    def __init__(self, device: str = "0") -> None:
        import ultralytics

        self._device = device
        self.version = f"ultralytics-{ultralytics.__version__}"

    def train(self, run: TrainingRun, *, data_yaml: Path, project_dir: Path) -> None:
        from ultralytics import YOLO

        model: Any = YOLO(run.model)
        model.train(
            data=str(data_yaml),
            epochs=run.epochs,
            patience=run.patience,
            imgsz=run.imgsz,
            batch=run.batch,
            seed=run.seed,
            deterministic=True,
            workers=run.workers,
            device=self._device,
            project=str(project_dir.parent),
            name=project_dir.name,
            exist_ok=True,
            plots=True,
            verbose=False,
        )

    def evaluate(self, weights: Path, *, data_yaml: Path, split: str, imgsz: int) -> SplitMetrics:
        from ultralytics import YOLO

        model: Any = YOLO(str(weights))
        result: Any = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=16,
            device=self._device,
            plots=False,
            verbose=False,
            project=str(weights.parents[2] / "eval"),
            name=split,
            exist_ok=True,
        )
        names: dict[int, str] = model.names
        per_class = {
            names[int(index)]: float(ap)
            for index, ap in zip(result.seg.ap_class_index, result.seg.ap50, strict=True)
        }
        return SplitMetrics(
            box_map50=float(result.box.map50),
            box_map50_95=float(result.box.map),
            mask_map50=float(result.seg.map50),
            mask_map50_95=float(result.seg.map),
            per_class_mask_map50=per_class,
        )
