"""Detector interface and conversion from raw model predictions to findings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from claimlens.domain import BoundingBox, DamageFinding, DamageType


class Detector(Protocol):
    @property
    def model_version(self) -> str: ...

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]: ...


def normalize_class_name(name: str) -> DamageType:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return DamageType(key)
    except ValueError:
        raise ValueError(f"model class {name!r} is not a known damage type") from None


def _clamp(value: float, upper: float) -> float:
    return min(max(float(value), 0.0), upper)


def polygon_area_fraction(points_xyn: Sequence[float]) -> float:
    """Area of a polygon in normalised [0, 1] coordinates (shoelace formula), clamped to [0, 1]."""
    if len(points_xyn) % 2:
        raise ValueError("polygon coordinates must come in x, y pairs")
    xs, ys = points_xyn[0::2], points_xyn[1::2]
    if len(xs) < 3:
        return 0.0
    twice_area = sum(
        xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs))
    )
    return min(abs(twice_area) / 2.0, 1.0)


def findings_from_predictions(
    *,
    photo_id: str,
    boxes_xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    class_names: Mapping[int, str],
    image_width: int,
    image_height: int,
    area_fractions: Sequence[float] | None = None,
) -> list[DamageFinding]:
    """`area_fractions` (mask area / image area) replaces the box area when given."""
    if not len(boxes_xyxy) == len(confidences) == len(class_ids):
        raise ValueError("boxes, confidences and class ids must have the same length")
    if area_fractions is not None and len(area_fractions) != len(boxes_xyxy):
        raise ValueError("area fractions must have the same length as the boxes")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive")
    width, height = float(image_width), float(image_height)
    findings: list[DamageFinding] = []
    predictions = zip(boxes_xyxy, confidences, class_ids, strict=True)
    for index, (box, confidence, class_id) in enumerate(predictions):
        if class_id not in class_names:
            raise ValueError(f"class id {class_id} is not in the model's class names")
        x1, x2 = _clamp(box[0], width), _clamp(box[2], width)
        y1, y2 = _clamp(box[1], height), _clamp(box[3], height)
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        box_fraction = min(area / (width * height), 1.0)
        fraction = box_fraction if area_fractions is None else area_fractions[index]
        findings.append(
            DamageFinding(
                photo_id=photo_id,
                damage_type=normalize_class_name(class_names[class_id]),
                confidence=float(confidence),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                image_area_fraction=min(max(fraction, 0.0), 1.0),
            )
        )
    return findings
