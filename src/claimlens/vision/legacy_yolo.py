"""Adapter for the original course model (YOLOv8n, 7 classes). Baseline only.

Requires the optional dependency group: `uv sync --group vision`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claimlens.domain import DamageFinding
from claimlens.vision.base import findings_from_predictions


class LegacyYoloDetector:
    def __init__(self, weights: Path, conf_floor: float = 0.10) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        from ultralytics import YOLO

        self._model: Any = YOLO(str(weights))
        self._conf_floor = conf_floor
        self.model_version = f"legacy-yolov8n:{weights.name}"

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        predictions = self._model.predict(
            source=str(image_path), conf=self._conf_floor, verbose=False
        )
        result = predictions[0]
        height, width = result.orig_shape
        boxes = result.boxes
        return findings_from_predictions(
            photo_id=photo_id,
            boxes_xyxy=boxes.xyxy.tolist(),
            confidences=boxes.conf.tolist(),
            class_ids=[int(c) for c in boxes.cls.tolist()],
            class_names=result.names,
            image_width=int(width),
            image_height=int(height),
        )
