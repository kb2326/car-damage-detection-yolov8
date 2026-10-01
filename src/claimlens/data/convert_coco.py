"""Convert a COCO instance-segmentation split (Roboflow export) into image records."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from claimlens.data.records import Annotation, ConversionResult, ImageRecord, relative_posix
from claimlens.data.taxonomy import Taxonomy

_CARDD_ID = re.compile(r"^(\d{6})_jpg\.rf\.")


def _normalise(coords: Sequence[float], width: int, height: int) -> tuple[float, ...]:
    return tuple(
        round(float(v) / (width if i % 2 == 0 else height), 6) for i, v in enumerate(coords)
    )


def convert_coco_split(
    annotations_file: Path,
    *,
    source: str,
    split: str,
    taxonomy: Taxonomy,
    repo_root: Path,
) -> ConversionResult:
    data: dict[str, Any] = json.loads(annotations_file.read_text(encoding="utf-8"))
    names = {int(c["id"]): str(c["name"]) for c in data["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)

    ignored: Counter[str] = Counter()
    records: list[ImageRecord] = []
    for image in sorted(data["images"], key=lambda item: str(item["file_name"])):
        file_name = str(image["file_name"])
        width, height = int(image["width"]), int(image["height"])
        annotations: list[Annotation] = []
        for annotation in by_image.get(int(image["id"]), []):
            name = names[int(annotation["category_id"])]
            label = taxonomy.canonical(name)
            if label is None:
                ignored[name] += 1
                continue
            segmentation = annotation["segmentation"]
            if not isinstance(segmentation, list):
                raise ValueError(
                    f"{annotations_file}: annotation {annotation['id']} uses RLE masks, "
                    "which are not supported"
                )
            if not segmentation:
                raise ValueError(
                    f"{annotations_file}: annotation {annotation['id']} has no polygon "
                    "(box-only annotations are not supported)"
                )
            for part in segmentation:
                annotations.append(Annotation(label=label, polygon=_normalise(part, width, height)))
        match = _CARDD_ID.match(file_name)
        records.append(
            ImageRecord(
                image_id=f"{source}:{Path(file_name).stem}",
                source=source,
                source_split=split,
                path=relative_posix(annotations_file.parent / file_name, repo_root),
                width=width,
                height=height,
                annotations=tuple(annotations),
                origin_id=match.group(1) if match else None,
            )
        )
    return ConversionResult(records=records, ignored_labels=ignored)
