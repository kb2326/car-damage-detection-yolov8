"""Polygon masks on a fixed grid: good enough for part assignment and severity, no extra deps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

GRID = 256
Mask = npt.NDArray[np.bool_]


def rasterize(polygon_xyn: Sequence[float], size: int = GRID) -> Mask:
    """Fill a polygon given in normalised [0, 1] x, y pairs onto a size x size boolean grid."""
    if len(polygon_xyn) % 2:
        raise ValueError("polygon coordinates must come in x, y pairs")
    canvas = Image.new("1", (size, size), 0)
    if len(polygon_xyn) >= 6:
        points = [
            (float(x) * size, float(y) * size)
            for x, y in zip(polygon_xyn[0::2], polygon_xyn[1::2], strict=True)
        ]
        ImageDraw.Draw(canvas).polygon(points, fill=1)
    return np.array(canvas, dtype=bool)


def area(mask: Mask) -> int:
    return int(mask.sum())


def intersection(a: Mask, b: Mask) -> int:
    return int(np.logical_and(a, b).sum())


def mask_iou(a: Mask, b: Mask) -> float:
    union = int(np.logical_or(a, b).sum())
    return 0.0 if union == 0 else intersection(a, b) / union
