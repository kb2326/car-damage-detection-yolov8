"""Write a dataset split as an Ultralytics YOLO segmentation folder."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from claimlens.data.records import ImageRecord
from claimlens.data.taxonomy import Taxonomy
from claimlens.data.validate import polygon_area

YOLO_SPLIT = {"train": "train", "valid": "val", "test": "test"}


def _usable(polygon: Sequence[float]) -> bool:
    return len(polygon) >= 6 and len(polygon) % 2 == 0 and polygon_area(polygon) > 0


def _label_line(class_index: int, polygon: Sequence[float]) -> str:
    coords = " ".join(f"{min(max(v, 0.0), 1.0):.6f}" for v in polygon)
    return f"{class_index} {coords}"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _data_yaml(out_dir: Path, taxonomy: Taxonomy) -> str:
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *(f"  {i}: {name}" for i, name in enumerate(taxonomy.classes)),
    ]
    return "\n".join(lines) + "\n"


def export_yolo_seg(
    records: Sequence[ImageRecord],
    splits: Mapping[str, str],
    taxonomy: Taxonomy,
    *,
    repo_root: Path,
    out_dir: Path,
) -> dict[str, int]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts: Counter[str] = Counter()
    for record in records:
        split = splits[record.image_id]
        if split == "excluded":
            counts["excluded"] += 1
            continue
        yolo_split = YOLO_SPLIT[split]
        name = record.image_id.replace(":", "__")
        source = repo_root / record.path
        _link_or_copy(source, out_dir / "images" / yolo_split / f"{name}{source.suffix.lower()}")
        lines = [
            _label_line(taxonomy.index(a.label), a.polygon)
            for a in record.annotations
            if _usable(a.polygon)
        ]
        label_path = out_dir / "labels" / yolo_split / f"{name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
        counts[yolo_split] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.yaml").write_text(_data_yaml(out_dir, taxonomy), encoding="utf-8", newline="\n")
    return dict(counts)
