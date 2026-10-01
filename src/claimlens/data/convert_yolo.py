"""Convert a YOLO segmentation split (images/ + labels/) into image records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml
from PIL import Image

from claimlens.data.records import Annotation, ConversionResult, ImageRecord, relative_posix
from claimlens.data.taxonomy import Taxonomy

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_yolo_seg_line(line: str) -> tuple[int, tuple[float, ...]]:
    parts = line.split()
    class_index = int(parts[0])
    coords = [float(value) for value in parts[1:]]
    if len(coords) == 4:
        cx, cy, w, h = coords
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        box = (x1, y1, x2, y1, x2, y2, x1, y2)
        return class_index, tuple(round(v, 6) for v in box)
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError("expected a box (4 values) or a polygon (6 or more, even)")
    return class_index, tuple(round(v, 6) for v in coords)


def read_class_names(yaml_path: Path) -> list[str]:
    names = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=int)]
    return [str(name) for name in names]


def convert_yolo_split(
    images_dir: Path,
    labels_dir: Path,
    *,
    source: str,
    split: str,
    class_names: Sequence[str],
    taxonomy: Taxonomy,
    repo_root: Path,
) -> ConversionResult:
    ignored: Counter[str] = Counter()
    records: list[ImageRecord] = []
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        annotations: list[Annotation] = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            lines = label_path.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    class_index, polygon = parse_yolo_seg_line(line)
                except ValueError as exc:
                    raise ValueError(f"{label_path.name}:{line_no}: {exc}") from None
                if not 0 <= class_index < len(class_names):
                    raise ValueError(
                        f"{label_path.name}:{line_no}: class index {class_index} is out of range"
                    )
                name = class_names[class_index]
                label = taxonomy.canonical(name)
                if label is None:
                    ignored[name] += 1
                    continue
                annotations.append(Annotation(label=label, polygon=polygon))
        records.append(
            ImageRecord(
                image_id=f"{source}:{image_path.stem}",
                source=source,
                source_split=split,
                path=relative_posix(image_path, repo_root),
                width=width,
                height=height,
                annotations=tuple(annotations),
            )
        )
    return ConversionResult(records=records, ignored_labels=ignored)
