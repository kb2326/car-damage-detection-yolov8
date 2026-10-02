"""Our YOLO11-seg damage model.

Size (`image_area_fraction`) comes from the **box**, because the rate card's severity bands and the
golden oracle are both defined on box area. Sizing by the mask outline made damage look smaller
than the bands assume and fast-tracked two claims that needed a person (M3a golden gate). Mask
outlines are used for fusion with parts in M3b instead.

Requires `uv sync --group vision`. Excluded from coverage; see tests/integration/test_yolo_seg.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claimlens.domain import DamageFinding
from claimlens.vision.base import findings_from_predictions


class YoloSegDetector:
    def __init__(self, weights: Path, run: str, conf_floor: float = 0.10) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        from ultralytics import YOLO

        self._model: Any = YOLO(str(weights))
        self._conf_floor = conf_floor
        self.model_version = f"yolo11-seg:{run}"

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        result = self._model.predict(source=str(image_path), conf=self._conf_floor, verbose=False)[
            0
        ]
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
