"""Turn open-vocabulary detections and masks into part-group annotations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    phrase: str
    score: float
    box: Box


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_detections(
    detections: Sequence[Detection],
    prompts: Mapping[str, str],
    *,
    min_score: float,
    iou_threshold: float = 0.5,
    max_per_group: int = 2,
) -> tuple[list[tuple[str, Detection]], Counter[str]]:
    """Exact phrase -> group, best score first, at most `max_per_group` non-overlapping boxes."""
    kept: list[tuple[str, Detection]] = []
    dropped: Counter[str] = Counter()
    for detection in sorted(detections, key=lambda d: (-d.score, d.phrase, d.box)):
        group = prompts.get(detection.phrase.strip().lower())
        if group is None:
            dropped["ambiguous_phrase"] += 1
            continue
        if detection.score < min_score:
            dropped["low_score"] += 1
            continue
        same_group = [d for g, d in kept if g == group]
        overlaps = any(box_iou(d.box, detection.box) > iou_threshold for d in same_group)
        if overlaps or len(same_group) >= max_per_group:
            dropped["duplicate"] += 1
            continue
        kept.append((group, detection))
    return kept, dropped


def mask_to_polygon(
    mask: npt.NDArray[np.bool_], *, epsilon_fraction: float = 0.005
) -> tuple[float, ...] | None:
    """Largest outer contour of a binary mask, simplified, as normalised (x, y) pairs."""
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_fraction * cv2.arcLength(largest, True)
    points = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2).tolist()
    if len(points) < 3:
        return None
    return tuple(round(v, 6) for x, y in points for v in (x / width, y / height))
