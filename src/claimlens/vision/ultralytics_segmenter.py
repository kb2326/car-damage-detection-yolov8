"""Ultralytics YOLO-seg as a `Segmenter`, from `.pt` or `.onnx`; plus ONNX export.

Requires `uv sync --group vision`. Excluded from coverage; see tests/integration/test_fused.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from claimlens.vision.instances import SegInstance, Segmentation


class UltralyticsSegmenter:
    def __init__(self, weights: Path, *, name: str, conf_floor: float = 0.10) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        onnx = weights.suffix == ".onnx"
        if onnx and importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError("onnxruntime is not installed: run `uv sync --group vision`")
        from ultralytics import YOLO

        self._model: Any = YOLO(str(weights), task="segment")
        self._conf_floor = conf_floor
        self.model_version = f"{name}{'-onnx' if onnx else ''}"

    def segment(self, image_path: Path) -> Segmentation:
        result = self._model.predict(
            source=str(image_path), conf=self._conf_floor, imgsz=640, verbose=False
        )[0]
        height, width = result.orig_shape
        if result.masks is None:
            return Segmentation(instances=(), width=int(width), height=int(height))
        names: dict[int, str] = result.names
        instances = tuple(
            SegInstance(
                label=names[int(cls)],
                confidence=float(conf),
                box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                polygon_xyn=tuple(float(v) for v in polygon.reshape(-1).tolist()),
            )
            for box, conf, cls, polygon in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                result.masks.xyn,
                strict=True,
            )
        )
        return Segmentation(instances=instances, width=int(width), height=int(height))


def export_onnx(weights: Path, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    model: Any = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False)
    return Path(str(exported))
