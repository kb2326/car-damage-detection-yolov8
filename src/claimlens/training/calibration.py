"""Confidence calibration: match predictions to truth, fit one temperature, measure ECE."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from claimlens.data.taxonomy import PartGroups
from claimlens.domain import Frozen
from claimlens.fusion import calibrate_confidence
from claimlens.vision.instances import SegInstance
from claimlens.vision.masks import mask_iou, rasterize

Truth = tuple[str, tuple[float, ...]]
Pair = tuple[float, bool]
_GRID = [round(0.25 + 0.05 * i, 2) for i in range(76)]  # 0.25 .. 4.00


class CalibrationResult(Frozen):
    run: str
    images: int
    predictions: int
    correct: int
    temperature: float
    ece_before: float
    ece_after: float
    recommended_threshold: float | None
    precision_at_threshold: float | None
    kept_at_threshold: int


def read_yolo_labels(path: Path, names: Mapping[int, str]) -> list[Truth]:
    truths: list[Truth] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        truths.append((names[int(parts[0])], tuple(float(v) for v in parts[1:])))
    return truths


def match_predictions(
    preds: Sequence[SegInstance], truths: Sequence[Truth], *, iou: float = 0.5
) -> list[Pair]:
    """Greedy, highest confidence first; a truth can be matched once; classes must agree."""
    truth_masks = [(label, rasterize(polygon)) for label, polygon in truths]
    used: set[int] = set()
    pairs: list[Pair] = []
    for pred in sorted(preds, key=lambda p: p.confidence, reverse=True):
        pred_mask = rasterize(pred.polygon_xyn)
        best_index, best_iou = None, iou
        for index, (label, truth_mask) in enumerate(truth_masks):
            if index in used or label != pred.label:
                continue
            value = mask_iou(pred_mask, truth_mask)
            if value >= best_iou:
                best_index, best_iou = index, value
        if best_index is not None:
            used.add(best_index)
        pairs.append((pred.confidence, best_index is not None))
    return pairs


def _nll(pairs: Sequence[Pair], temperature: float) -> float:
    total = 0.0
    for p, correct in pairs:
        q = min(max(calibrate_confidence(p, temperature), 1e-9), 1.0 - 1e-9)
        total -= math.log(q) if correct else math.log(1.0 - q)
    return total


def fit_temperature(pairs: Sequence[Pair]) -> float:
    if not pairs:
        raise ValueError("no matched predictions to calibrate; check the split and label folders")
    return min(_GRID, key=lambda t: (_nll(pairs, t), abs(t - 1.0)))


def ece(pairs: Sequence[Pair], *, temperature: float = 1.0, bins: int = 10) -> float:
    if not pairs:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, correct in pairs:
        q = calibrate_confidence(p, temperature)
        buckets[min(int(q * bins), bins - 1)].append((q, correct))
    total = len(pairs)
    error = 0.0
    for bucket in buckets:
        if bucket:
            confidence = sum(q for q, _ in bucket) / len(bucket)
            accuracy = sum(c for _, c in bucket) / len(bucket)
            error += len(bucket) / total * abs(confidence - accuracy)
    return error


def recommend_threshold(
    pairs: Sequence[Pair], temperature: float, *, precision: float = 0.80
) -> float | None:
    """Lowest threshold on a 0.05 grid whose kept calibrated predictions reach the precision."""
    for step in range(1, 20):
        threshold = round(step * 0.05, 2)
        kept = [c for p, c in pairs if calibrate_confidence(p, temperature) >= threshold]
        if kept and sum(kept) / len(kept) >= precision:
            return threshold
    return None


def part_agreement(
    truths: Mapping[str, Sequence[Truth]],
    preds: Mapping[str, Sequence[SegInstance]],
    groups: PartGroups,
    *,
    iou: float = 0.5,
) -> dict[str, tuple[int, int]]:
    """Per part group: reviewed parts the part model also finds (same group, mask IoU >= iou)."""
    agreed: dict[str, int] = {}
    total: dict[str, int] = {}
    for image_id, image_truths in truths.items():
        predicted = [
            (groups.group_of(p.label), rasterize(p.polygon_xyn)) for p in preds.get(image_id, ())
        ]
        for group, polygon in image_truths:
            total[group] = total.get(group, 0) + 1
            mask = rasterize(polygon)
            if any(g == group and mask_iou(mask, m) >= iou for g, m in predicted):
                agreed[group] = agreed.get(group, 0) + 1
    return {group: (agreed.get(group, 0), count) for group, count in sorted(total.items())}
