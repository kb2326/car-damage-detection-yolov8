"""Model-agnostic segmentation output: one labelled outline per detected object."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import Field

from claimlens.domain import Frozen


class SegInstance(Frozen):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box_xyxy: tuple[float, float, float, float]
    polygon_xyn: tuple[float, ...]


class Segmentation(Frozen):
    instances: tuple[SegInstance, ...]
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Segmenter(Protocol):
    model_version: str

    def segment(self, image_path: Path) -> Segmentation: ...
