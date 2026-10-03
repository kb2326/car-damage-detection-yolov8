"""Damage-to-part fusion: which part is each damage on, and how much of that part does it cover."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from claimlens.data.taxonomy import PartGroups
from claimlens.domain import BoundingBox, DamageFinding, Frozen
from claimlens.vision.base import normalize_class_name
from claimlens.vision.instances import SegInstance, Segmenter
from claimlens.vision.masks import area, intersection, rasterize


class FusedDamage(Frozen):
    damage: SegInstance
    part_group: str | None
    part_area_ratio: float | None


def fuse(
    damage: Sequence[SegInstance],
    parts: Sequence[SegInstance],
    groups: PartGroups,
    *,
    min_overlap: float = 0.10,
) -> list[FusedDamage]:
    """Each damage goes to the plausible part covering most of it (ties: the smaller part).

    `part_area_ratio` is the share of that part the damage covers: overlap / part area, so it
    is never above 1.
    """
    part_masks = [(part, rasterize(part.polygon_xyn)) for part in parts]
    fused: list[FusedDamage] = []
    for instance in damage:
        damage_mask = rasterize(instance.polygon_xyn)
        damage_area = area(damage_mask)
        damage_type = normalize_class_name(instance.label).value
        best: tuple[tuple[float, int], str, int, int] | None = None
        if damage_area > 0:
            for part, part_mask in part_masks:
                part_area = area(part_mask)
                if part_area == 0:
                    continue
                group = groups.group_of(part.label)
                if not groups.plausible(damage_type, group):
                    continue
                shared = intersection(damage_mask, part_mask)
                overlap = shared / damage_area
                if overlap < min_overlap:
                    continue
                key = (overlap, -part_area)
                if best is None or key > best[0]:
                    best = (key, group, shared, part_area)
        if best is None:
            fused.append(FusedDamage(damage=instance, part_group=None, part_area_ratio=None))
        else:
            _, group, shared, part_area = best
            fused.append(
                FusedDamage(damage=instance, part_group=group, part_area_ratio=shared / part_area)
            )
    return fused


def calibrate_confidence(p: float, temperature: float) -> float:
    """Temperature scaling on the logit: T > 1 softens, T < 1 sharpens, T = 1 is unchanged."""
    clipped = min(max(p, 1e-6), 1.0 - 1e-6)
    logit = math.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


class FusedDetector:
    def __init__(
        self,
        damage: Segmenter,
        parts: Segmenter,
        groups: PartGroups,
        *,
        temperature: float | None = None,
        min_overlap: float = 0.10,
    ) -> None:
        self._damage = damage
        self._parts = parts
        self._groups = groups
        self._temperature = temperature
        self._min_overlap = min_overlap
        suffix = "" if temperature is None else f"+T{temperature:.2f}"
        self.model_version = f"fused:{damage.model_version}+{parts.model_version}{suffix}"

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        damage = self._damage.segment(image_path)
        parts = self._parts.segment(image_path)
        width, height = float(damage.width), float(damage.height)
        findings: list[DamageFinding] = []
        for item in fuse(
            damage.instances, parts.instances, self._groups, min_overlap=self._min_overlap
        ):
            x1, y1, x2, y2 = (
                min(max(item.damage.box_xyxy[0], 0.0), width),
                min(max(item.damage.box_xyxy[1], 0.0), height),
                min(max(item.damage.box_xyxy[2], 0.0), width),
                min(max(item.damage.box_xyxy[3], 0.0), height),
            )
            confidence = item.damage.confidence
            if self._temperature is not None:
                confidence = calibrate_confidence(confidence, self._temperature)
            findings.append(
                DamageFinding(
                    photo_id=photo_id,
                    damage_type=normalize_class_name(item.damage.label),
                    confidence=confidence,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    # Box area, as the rate card defines (M3a gate); the mask is used for parts.
                    image_area_fraction=min(
                        max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / (width * height), 1.0
                    ),
                    part=item.part_group,
                    part_area_ratio=item.part_area_ratio,
                )
            )
        return findings
