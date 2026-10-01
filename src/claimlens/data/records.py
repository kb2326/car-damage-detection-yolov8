"""One canonical record per image, shared by every source and stage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from claimlens.domain import Frozen


class Annotation(Frozen):
    """One instance mask as a polygon of normalised (x, y) pairs in [0, 1]."""

    label: str
    polygon: tuple[float, ...]
    score: float | None = None


class ImageRecord(Frozen):
    image_id: str
    source: str
    source_split: str
    path: str
    width: int
    height: int
    annotations: tuple[Annotation, ...] = ()
    origin_id: str | None = None


@dataclass
class ConversionResult:
    records: list[ImageRecord]
    ignored_labels: Counter[str] = field(default_factory=Counter)


def relative_posix(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def write_records(path: Path, records: Sequence[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF line endings on every OS so DVC hashes match across machines.
    text = "".join(r.model_dump_json() + "\n" for r in records)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_records(path: Path) -> list[ImageRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [ImageRecord.model_validate_json(line) for line in lines if line.strip()]
    seen: set[str] = set()
    for record in records:
        if record.image_id in seen:
            raise ValueError(f"duplicate image id {record.image_id}")
        seen.add(record.image_id)
    return records
