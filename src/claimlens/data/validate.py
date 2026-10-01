"""Data contract: every record must be checked before it can be split or exported."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from PIL import Image

from claimlens.data.records import ImageRecord
from claimlens.data.taxonomy import Taxonomy
from claimlens.domain import Frozen

_HARD_LIMIT = 0.01


class Issue(Frozen):
    image_id: str
    severity: Literal["error", "warning"]
    code: str
    message: str


class ValidationReport(Frozen):
    total_images: int
    errors: tuple[Issue, ...]
    warning_counts: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.errors


def polygon_area(polygon: Sequence[float]) -> float:
    xs, ys = polygon[0::2], polygon[1::2]
    twice = sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))
    return abs(twice) / 2


def validate_record(
    record: ImageRecord, repo_root: Path, taxonomy: Taxonomy, *, min_area: float = 1e-4
) -> list[Issue]:
    issues: list[Issue] = []

    def add(severity: Literal["error", "warning"], code: str, message: str) -> None:
        issues.append(Issue(image_id=record.image_id, severity=severity, code=code, message=message))

    path = repo_root / record.path
    if not path.is_file():
        add("error", "missing_file", f"{record.path} does not exist")
    else:
        try:
            with Image.open(path) as image:
                size = image.size
        except Exception:
            add("error", "unreadable_image", f"{record.path} cannot be opened")
        else:
            if size != (record.width, record.height):
                add(
                    "error",
                    "size_mismatch",
                    f"file is {size}, record says {record.width}x{record.height}",
                )

    if not record.annotations:
        add("warning", "no_annotations", "image has no instances")
    for number, annotation in enumerate(record.annotations, start=1):
        if annotation.label not in taxonomy.classes:
            add("error", "unknown_label", f"instance {number}: label {annotation.label!r}")
        polygon = annotation.polygon
        if any(v < -_HARD_LIMIT or v > 1 + _HARD_LIMIT for v in polygon):
            add("error", "out_of_bounds", f"instance {number}: coordinates outside [0, 1]")
            continue
        if len(polygon) < 6 or len(polygon) % 2 or polygon_area(polygon) <= 0:
            add("warning", "degenerate_polygon", f"instance {number}: not a valid polygon")
            continue
        if any(v < 0 or v > 1 for v in polygon):
            add("warning", "clamped", f"instance {number}: coordinates slightly outside [0, 1]")
        if polygon_area(polygon) < min_area:
            add("warning", "tiny_instance", f"instance {number}: area below {min_area}")
    return issues


def validate_records(
    records: Sequence[ImageRecord], repo_root: Path, taxonomy: Taxonomy
) -> ValidationReport:
    errors: list[Issue] = []
    warnings: Counter[str] = Counter()
    for record in records:
        for issue in validate_record(record, repo_root, taxonomy):
            if issue.severity == "error":
                errors.append(issue)
            else:
                warnings[issue.code] += 1
    return ValidationReport(
        total_images=len(records), errors=tuple(errors), warning_counts=dict(warnings)
    )
