# M2a Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `claimlens data fetch <source>` and `claimlens data build <dataset>` turn raw public datasets into validated, deduplicated, leakage-free YOLO-seg training sets (`damage-v1`, `parts-v1`) with statistics, versioned by DVC.

**Architecture:** Converters map each source (COCO-seg or YOLO-seg) onto one `ImageRecord` JSONL format using a versioned taxonomy that maps labels **by name**. A data contract validates every record. Perceptual hashes cluster near-duplicates across all sources and the golden-claims images; whole clusters are assigned to one split and golden-adjacent clusters are excluded. An exporter writes YOLO-seg folders; a stats module writes DVC metrics.

**Tech Stack:** Python 3.12, Pydantic v2, Pillow, ImageHash, NumPy 2 (`bitwise_count`), PyYAML, urllib (stdlib), DVC 3.

**Spec:** [`docs/specs/2026-10-01-m2-data-engine-design.md`](../specs/2026-10-01-m2-data-engine-design.md) (parent: `2026-10-01-claimlens-design.md` section 12).

## Global Constraints

- Everything from the M1 plan's Global Constraints still applies (uv, mypy `--strict`, ruff format before ruff check, ASCII-only Python, `match=` on `pytest.raises(ValueError)`, Conventional Commits with the Co-Authored-By line, CI without data/weights/keys).
- New runtime dependencies are limited to `imagehash>=4.3`, `numpy>=2.0`, `pyyaml>=6.0`. `dvc>=3.50` is dev-only.
- Labels are mapped by **name** through `config/taxonomy.toml`, never by index position.
- Raw data under `data/raw/<source>/` is immutable: fetch refuses to overwrite an existing folder.
- The Roboflow API key is read from the `ROBOFLOW_API_KEY` environment variable or `.env`, and never printed or logged.
- Zip extraction must refuse paths that escape the destination folder.
- `data/`, `models/` and `var/` content is never committed; only `.dvc` pointer files and `data/README.md` are.

## Review Focus

1. **An image that appears in both a source's train and test split** (exact or resized copy) must end up in exactly one split (Task 7, `test_cluster_spanning_train_and_test_goes_to_test`).
2. **A training image that is a near-duplicate of a golden-claims photo** must be excluded from all exported splits (Task 11, `test_build_excludes_images_near_golden_photos`).
3. **A label name that is not in the taxonomy** must stop the build with a clear error, not be dropped or remapped silently (Task 2, `test_unknown_label_raises`; Task 4, `test_coco_unknown_category_raises`).
4. **A malicious zip entry such as `../../evil.txt`** must not be written outside the destination (Task 10, `test_extract_refuses_path_traversal`).
5. **Degenerate polygons** (fewer than 3 points, zero area, slightly outside [0, 1]) must not crash export or produce invalid YOLO lines (Task 8, `test_export_skips_degenerate_and_clamps_polygons`).

---

## File structure

```
config/taxonomy.toml           canonical damage and part classes (Task 2)
config/datasets.toml           sources and datasets (Task 10)
src/claimlens/data/
  __init__.py
  taxonomy.py                  Taxonomy, normalize_label, UnknownLabelError (Task 2)
  records.py                   Annotation, ImageRecord, ConversionResult, JSONL IO (Task 3)
  convert_coco.py              COCO-seg -> records (Task 4)
  convert_yolo.py              YOLO-seg -> records (Task 5)
  validate.py                  data contract (Task 6)
  dedupe.py                    sha256 + pHash clusters (Task 7)
  split.py                     cluster-aware split assignment (Task 7)
  export.py                    YOLO-seg export (Task 8)
  stats.py                     statistics + Markdown (Task 9)
  config.py                    SourceConfig, DatasetConfig, loaders, read_secret (Task 10)
  fetch.py                     downloads, safe unzip, manifest (Task 10)
  pipeline.py                  build_dataset orchestration (Task 11)
dvc.yaml                       build stages (Task 12)
docs/data-card.md              datasheet (Task 12)
```

---

### Task 1: Rename the legacy subset and regenerate golden paths

**Files:**
- Move: `data/raw/roboflow-car-damage-v6/` → `data/raw/legacy-course-subset/` (filesystem move; not tracked by git)
- Modify: `scripts/build_golden_v0.py` (`DATASET` constant)
- Modify: `data/README.md`, `legacy/README.md` (paths and description)
- Regenerate: `evals/golden/v0/claims.jsonl`

**Interfaces:**
- Produces: golden v0 photo paths under `data/raw/legacy-course-subset/test/images/`.

- [ ] **Step 1: Move the folder and update the constant**

```bash
mv data/raw/roboflow-car-damage-v6 data/raw/legacy-course-subset
```

In `scripts/build_golden_v0.py` change `DATASET = Path("data/raw/roboflow-car-damage-v6")` to `DATASET = Path("data/raw/legacy-course-subset")`.

- [ ] **Step 2: Regenerate and prove only paths changed**

Run: `uv run python scripts/build_golden_v0.py`
Expected: `Wrote 50 golden claims ...`

Run: `git diff --stat evals/golden/v0/claims.jsonl && git show HEAD:evals/golden/v0/claims.jsonl | sed 's#data/raw/roboflow-car-damage-v6#data/raw/legacy-course-subset#g' | diff - evals/golden/v0/claims.jsonl && echo IDENTICAL`
Expected: `IDENTICAL`

- [ ] **Step 3: Update documentation**

In `data/README.md`, replace the `### raw/roboflow-car-damage-v6/` section with:

```markdown
### `raw/legacy-course-subset/`

- **What it is:** the 339-image folder used by the original course project. It is **not** Roboflow
  v6 (which has 4,000 images and 6 classes); it has 7 classes including `smash` and mixes CarDD
  images with other Roboflow Universe sources.
- **Use:** golden claims only (`evals/golden/v0`), never training.
- **Terms:** Roboflow Universe exports, CC BY 4.0, mixed sources.
- **Known issues:** see the audit in [`legacy/README.md`](../legacy/README.md)
```

In `legacy/README.md` replace `data/raw/roboflow-car-damage-v6/` with `data/raw/legacy-course-subset/`.

- [ ] **Step 4: Run the suite and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add scripts/build_golden_v0.py data/README.md legacy/README.md evals/golden/v0/claims.jsonl
git commit -m "chore: rename legacy dataset folder to legacy-course-subset"
```

---

### Task 2: Taxonomy

**Files:**
- Create: `config/taxonomy.toml`
- Create: `src/claimlens/data/__init__.py` (docstring only: `"""Data engine: ingest, validate, deduplicate, split and export datasets."""`)
- Create: `src/claimlens/data/taxonomy.py`
- Test: `tests/unit/test_taxonomy.py`

**Interfaces:**
- Produces: `normalize_label(label: str) -> str`; `UnknownLabelError(ValueError)` with `.label`; `Taxonomy(name, classes, ignore=())` with `canonical(label) -> str | None` (None for ignored labels) and `index(label) -> int`; `load_taxonomies(path) -> dict[str, Taxonomy]`.

- [ ] **Step 1: Write** `config/taxonomy.toml`

```toml
# Canonical label sets. Class order is the YOLO class index: only ever append new classes.
version = "taxonomy-v1"

[damage]
classes = ["crack", "dent", "glass_shatter", "lamp_broken", "scratch", "tire_flat"]
# Present in the legacy course subset only, with too few examples to train.
ignore = ["smash"]

[parts]
classes = [
  "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light", "back_light",
  "back_right_door", "back_right_light", "front_bumper", "front_door", "front_glass",
  "front_left_door", "front_left_light", "front_light", "front_right_door", "front_right_light",
  "hood", "left_mirror", "right_mirror", "tailgate", "trunk", "wheel",
]
# Catch-all class with no consistent meaning.
ignore = ["object"]
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_taxonomy.py`

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.taxonomy import Taxonomy, UnknownLabelError, load_taxonomies, normalize_label

ROOT = Path(__file__).resolve().parents[2]
DAMAGE = Taxonomy(name="damage", classes=("crack", "glass_shatter"), ignore=("smash",))


def test_normalize_label_handles_spaces_hyphens_and_case() -> None:
    assert normalize_label(" Glass-Shatter ") == "glass_shatter"
    assert normalize_label("tire flat") == "tire_flat"


def test_canonical_maps_by_name() -> None:
    assert DAMAGE.canonical("glass shatter") == "glass_shatter"
    assert DAMAGE.index("glass_shatter") == 1


def test_ignored_label_returns_none() -> None:
    assert DAMAGE.canonical("smash") is None


def test_unknown_label_raises() -> None:
    with pytest.raises(UnknownLabelError, match="Front-Windscreen-Damage"):
        DAMAGE.canonical("Front-Windscreen-Damage")


def test_duplicate_or_overlapping_classes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Taxonomy(name="x", classes=("dent", "dent"))
    with pytest.raises(ValidationError, match="both"):
        Taxonomy(name="x", classes=("dent",), ignore=("dent",))


def test_repo_taxonomies_load() -> None:
    taxonomies = load_taxonomies(ROOT / "config" / "taxonomy.toml")
    assert taxonomies["damage"].classes[0] == "crack"
    assert len(taxonomies["damage"].classes) == 6
    assert len(taxonomies["parts"].classes) == 22
    assert taxonomies["parts"].canonical("object") is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_taxonomy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data'`

- [ ] **Step 4: Write** `src/claimlens/data/taxonomy.py`

```python
"""Label taxonomies: map every source's label names onto one canonical class list."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import model_validator

from claimlens.domain import Frozen


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


class UnknownLabelError(ValueError):
    def __init__(self, taxonomy: str, label: str) -> None:
        super().__init__(f"label {label!r} is not in the {taxonomy} taxonomy")
        self.label = label


class Taxonomy(Frozen):
    name: str
    classes: tuple[str, ...]
    ignore: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("taxonomy classes must be unique")
        overlap = set(self.classes) & set(self.ignore)
        if overlap:
            raise ValueError(f"labels listed as both class and ignored: {sorted(overlap)}")
        return self

    def canonical(self, label: str) -> str | None:
        """Return the canonical class, None for an ignored label, or raise for an unknown one."""
        key = normalize_label(label)
        if key in self.classes:
            return key
        if key in self.ignore:
            return None
        raise UnknownLabelError(self.name, label)

    def index(self, label: str) -> int:
        return self.classes.index(label)


def load_taxonomies(path: Path) -> dict[str, Taxonomy]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        name: Taxonomy.model_validate({"name": name, **body})
        for name, body in data.items()
        if isinstance(body, dict)
    }
```

- [ ] **Step 5: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_taxonomy.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 6 passed; no issues.

```bash
git add config/taxonomy.toml src/claimlens/data tests/unit/test_taxonomy.py
git commit -m "feat: add versioned label taxonomy mapped by name"
```

---

### Task 3: Image records

**Files:**
- Create: `src/claimlens/data/records.py`
- Test: `tests/unit/test_records.py`

**Interfaces:**
- Produces: `Annotation(label, polygon: tuple[float, ...])` (normalised x, y pairs); `ImageRecord(image_id, source, source_split, path, width, height, annotations=(), origin_id=None)`; `ConversionResult(records: list[ImageRecord], ignored_labels: Counter[str])` (dataclass); `relative_posix(path, repo_root) -> str`; `write_records(path, records)`; `read_records(path) -> list[ImageRecord]`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_records.py`

```python
from pathlib import Path

import pytest

from claimlens.data.records import (
    Annotation,
    ImageRecord,
    read_records,
    relative_posix,
    write_records,
)


def _record(image_id: str = "src:a") -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source="src",
        source_split="train",
        path="data/raw/src/a.jpg",
        width=640,
        height=480,
        annotations=(Annotation(label="dent", polygon=(0.1, 0.1, 0.5, 0.1, 0.5, 0.5)),),
    )


def test_records_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "interim" / "records.jsonl"
    records = [_record("src:a"), _record("src:b")]
    write_records(path, records)
    assert read_records(path) == records


def test_duplicate_image_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_records(path, [_record(), _record()])
    with pytest.raises(ValueError, match="duplicate image id src:a"):
        read_records(path)


def test_relative_posix_uses_forward_slashes(tmp_path: Path) -> None:
    target = tmp_path / "data" / "raw" / "x.jpg"
    assert relative_posix(target, tmp_path) == "data/raw/x.jpg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_records.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.records'`

- [ ] **Step 3: Write** `src/claimlens/data/records.py`

```python
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
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def read_records(path: Path) -> list[ImageRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [ImageRecord.model_validate_json(line) for line in lines if line.strip()]
    seen: set[str] = set()
    for record in records:
        if record.image_id in seen:
            raise ValueError(f"duplicate image id {record.image_id}")
        seen.add(record.image_id)
    return records
```

- [ ] **Step 4: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_records.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 3 passed; no issues.

```bash
git add src/claimlens/data/records.py tests/unit/test_records.py
git commit -m "feat: add canonical image records with JSONL storage"
```

---

### Task 4: COCO-seg converter

**Files:**
- Create: `src/claimlens/data/convert_coco.py`
- Create: `tests/data_helpers.py`
- Test: `tests/unit/test_convert_coco.py`

**Interfaces:**
- Consumes: `Taxonomy` (Task 2), `Annotation`, `ImageRecord`, `ConversionResult`, `relative_posix` (Task 3).
- Produces: `convert_coco_split(annotations_file: Path, *, source: str, split: str, taxonomy: Taxonomy, repo_root: Path) -> ConversionResult`. Image ids are `f"{source}:{file stem}"`; `origin_id` is the 6-digit CarDD id when the file name starts with `NNNNNN_jpg.rf.`. Test helper `make_pattern_image(path, seed, size=(320, 240)) -> Path` draws seeded rectangles so perceptual hashes are meaningful.

- [ ] **Step 1: Write the test helper** `tests/data_helpers.py`

```python
"""Helpers for data-engine tests."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw


def make_pattern_image(path: Path, seed: int, size: tuple[int, int] = (320, 240)) -> Path:
    """An image of seeded coloured rectangles: same seed, same picture."""
    rng = random.Random(seed)
    image = Image.new("RGB", size, (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    draw = ImageDraw.Draw(image)
    width, height = size
    for _ in range(6):
        x1, y1 = rng.randrange(width // 2), rng.randrange(height // 2)
        x2, y2 = x1 + rng.randrange(20, width // 2), y1 + rng.randrange(20, height // 2)
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        draw.rectangle((x1, y1, x2, y2), fill=colour)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG" if path.suffix.lower() == ".png" else "JPEG", quality=95)
    return path
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_convert_coco.py`

```python
import json
from pathlib import Path

import pytest

from claimlens.data.convert_coco import convert_coco_split
from claimlens.data.taxonomy import Taxonomy, UnknownLabelError
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"), ignore=("smash",))


def _coco(
    tmp_path: Path, categories: list[dict[str, object]], annotations: list[dict[str, object]]
) -> Path:
    split_dir = tmp_path / "data" / "raw" / "cardd" / "train"
    make_pattern_image(split_dir / "000581_jpg.rf.abc.jpg", seed=1, size=(200, 100))
    make_pattern_image(split_dir / "other.jpg", seed=2, size=(200, 100))
    payload = {
        "categories": categories,
        "images": [
            {"id": 0, "file_name": "000581_jpg.rf.abc.jpg", "width": 200, "height": 100},
            {"id": 1, "file_name": "other.jpg", "width": 200, "height": 100},
        ],
        "annotations": annotations,
    }
    path = split_dir / "_annotations.coco.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


CATEGORIES: list[dict[str, object]] = [
    {"id": 0, "name": "dent", "supercategory": "none"},
    {"id": 1, "name": "crack", "supercategory": "dent"},
    {"id": 2, "name": "dent", "supercategory": "dent"},
    {"id": 3, "name": "smash", "supercategory": "dent"},
]


def test_coco_polygons_are_normalised_and_labels_mapped(tmp_path: Path) -> None:
    annotations: list[dict[str, object]] = [
        {"id": 0, "image_id": 0, "category_id": 2, "segmentation": [[0, 0, 100, 0, 100, 50]]},
        {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[0, 0, 200, 0, 200, 100]]},
        {"id": 2, "image_id": 1, "category_id": 3, "segmentation": [[0, 0, 10, 0, 10, 10]]},
    ]
    result = convert_coco_split(
        _coco(tmp_path, CATEGORIES, annotations),
        source="cardd",
        split="train",
        taxonomy=TAXONOMY,
        repo_root=tmp_path,
    )

    first, second = result.records
    assert first.image_id == "cardd:000581_jpg.rf.abc"
    assert first.origin_id == "000581"
    assert first.path == "data/raw/cardd/train/000581_jpg.rf.abc.jpg"
    assert [a.label for a in first.annotations] == ["dent", "crack"]
    assert first.annotations[0].polygon == (0.0, 0.0, 0.5, 0.0, 0.5, 0.5)
    assert second.annotations == ()
    assert second.origin_id is None
    assert result.ignored_labels == {"smash": 1}


def test_coco_unknown_category_raises(tmp_path: Path) -> None:
    categories: list[dict[str, object]] = [{"id": 5, "name": "Headlight-Damage"}]
    annotations: list[dict[str, object]] = [
        {"id": 0, "image_id": 0, "category_id": 5, "segmentation": [[0, 0, 1, 0, 1, 1]]}
    ]
    with pytest.raises(UnknownLabelError, match="Headlight-Damage"):
        convert_coco_split(
            _coco(tmp_path, categories, annotations),
            source="cardd",
            split="train",
            taxonomy=TAXONOMY,
            repo_root=tmp_path,
        )


def test_coco_rle_masks_are_rejected(tmp_path: Path) -> None:
    annotations: list[dict[str, object]] = [
        {
            "id": 7,
            "image_id": 0,
            "category_id": 2,
            "segmentation": {"counts": "abc", "size": [100, 200]},
        }
    ]
    with pytest.raises(ValueError, match="annotation 7 uses RLE"):
        convert_coco_split(
            _coco(tmp_path, CATEGORIES, annotations),
            source="cardd",
            split="train",
            taxonomy=TAXONOMY,
            repo_root=tmp_path,
        )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_convert_coco.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.convert_coco'`

- [ ] **Step 4: Write** `src/claimlens/data/convert_coco.py`

```python
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
```

- [ ] **Step 5: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_convert_coco.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 3 passed; no issues.

```bash
git add src/claimlens/data/convert_coco.py tests/data_helpers.py tests/unit/test_convert_coco.py
git commit -m "feat: convert COCO segmentation exports into image records"
```

---

### Task 5: YOLO-seg converter

**Files:**
- Create: `src/claimlens/data/convert_yolo.py`
- Test: `tests/unit/test_convert_yolo.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces: `parse_yolo_seg_line(line: str) -> tuple[int, tuple[float, ...]]` (a 4-value box becomes a 4-corner polygon); `read_class_names(yaml_path: Path) -> list[str]` (accepts list or index-keyed mapping); `convert_yolo_split(images_dir, labels_dir, *, source, split, class_names, taxonomy, repo_root) -> ConversionResult`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_convert_yolo.py`

```python
from pathlib import Path

import pytest

from claimlens.data.convert_yolo import convert_yolo_split, parse_yolo_seg_line, read_class_names
from claimlens.data.taxonomy import Taxonomy
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"), ignore=("smash",))
NAMES = ["crack", "dent", "smash"]


def test_box_line_becomes_rectangle_polygon() -> None:
    assert parse_yolo_seg_line("1 0.5 0.5 0.2 0.4") == (1, (0.4, 0.3, 0.6, 0.3, 0.6, 0.7, 0.4, 0.7))


def test_bad_line_raises() -> None:
    with pytest.raises(ValueError, match="expected a box"):
        parse_yolo_seg_line("1 0.5 0.5 0.2 0.4 0.1")


def test_read_class_names_accepts_list_and_mapping(tmp_path: Path) -> None:
    listed = tmp_path / "a.yaml"
    listed.write_text("names: ['crack', 'dent']\n", encoding="utf-8")
    mapped = tmp_path / "b.yaml"
    mapped.write_text("names:\n  1: dent\n  0: crack\n", encoding="utf-8")
    assert read_class_names(listed) == ["crack", "dent"]
    assert read_class_names(mapped) == ["crack", "dent"]


def test_convert_yolo_split_reads_labels_by_name(tmp_path: Path) -> None:
    images = tmp_path / "data" / "raw" / "legacy" / "train" / "images"
    labels = tmp_path / "data" / "raw" / "legacy" / "train" / "labels"
    make_pattern_image(images / "a.jpg", seed=1)
    make_pattern_image(images / "b.png", seed=2)
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("1 0.1 0.1 0.4 0.1 0.4 0.4\n2 0 0 1 0 1 1\n\n", encoding="utf-8")

    result = convert_yolo_split(
        images,
        labels,
        source="legacy",
        split="train",
        class_names=NAMES,
        taxonomy=TAXONOMY,
        repo_root=tmp_path,
    )

    first, second = result.records
    assert first.image_id == "legacy:a"
    assert (first.width, first.height) == (320, 240)
    assert [a.label for a in first.annotations] == ["dent"]
    assert second.annotations == ()
    assert result.ignored_labels == {"smash": 1}


def test_out_of_range_class_index_names_the_file(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    make_pattern_image(images / "a.jpg", seed=1)
    labels.mkdir()
    (labels / "a.txt").write_text("7 0.1 0.1 0.4 0.1 0.4 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a\.txt:1: class index 7"):
        convert_yolo_split(
            images,
            labels,
            source="s",
            split="train",
            class_names=NAMES,
            taxonomy=TAXONOMY,
            repo_root=tmp_path,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_convert_yolo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.convert_yolo'`

- [ ] **Step 3: Write** `src/claimlens/data/convert_yolo.py`

```python
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
```

- [ ] **Step 4: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_convert_yolo.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 5 passed; no issues.

```bash
git add src/claimlens/data/convert_yolo.py tests/unit/test_convert_yolo.py
git commit -m "feat: convert YOLO segmentation folders into image records"
```

---

### Task 6: Data contract validation

**Files:**
- Create: `src/claimlens/data/validate.py`
- Test: `tests/unit/test_validate.py`

**Interfaces:**
- Consumes: `ImageRecord`, `Annotation` (Task 3), `Taxonomy` (Task 2).
- Produces: `polygon_area(polygon) -> float`; `Issue(image_id, severity, code, message)`; `ValidationReport(total_images, errors, warning_counts)` with property `ok`; `validate_record(record, repo_root, taxonomy, *, min_area=1e-4) -> list[Issue]`; `validate_records(records, repo_root, taxonomy) -> ValidationReport`.

| Code | Severity | Rule |
|---|---|---|
| `missing_file` | error | image file does not exist |
| `unreadable_image` | error | Pillow cannot open it |
| `size_mismatch` | error | file size differs from the record's width/height |
| `unknown_label` | error | label is not a taxonomy class |
| `out_of_bounds` | error | any coordinate below -0.01 or above 1.01 |
| `no_annotations` | warning | image has no instances |
| `degenerate_polygon` | warning | fewer than 3 points or zero area (dropped at export) |
| `clamped` | warning | a coordinate slightly outside [0, 1] (clamped at export) |
| `tiny_instance` | warning | area below `min_area` |

- [ ] **Step 1: Write the failing test** `tests/unit/test_validate.py`

```python
from pathlib import Path

import pytest

from claimlens.data.records import Annotation, ImageRecord
from claimlens.data.taxonomy import Taxonomy
from claimlens.data.validate import Issue, polygon_area, validate_record, validate_records
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("dent",))
SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5)


def _record(tmp_path: Path, *annotations: Annotation, width: int = 320) -> ImageRecord:
    make_pattern_image(tmp_path / "a.jpg", seed=1)
    return ImageRecord(
        image_id="s:a",
        source="s",
        source_split="train",
        path="a.jpg",
        width=width,
        height=240,
        annotations=annotations,
    )


def _codes(issues: list[Issue]) -> list[str]:
    return sorted(i.code for i in issues)


def test_polygon_area_of_square() -> None:
    assert polygon_area(SQUARE) == pytest.approx(0.16)


def test_clean_record_has_no_issues(tmp_path: Path) -> None:
    assert (
        validate_record(
            _record(tmp_path, Annotation(label="dent", polygon=SQUARE)), tmp_path, TAXONOMY
        )
        == []
    )


def test_record_level_errors(tmp_path: Path) -> None:
    record = _record(tmp_path, Annotation(label="crack", polygon=SQUARE), width=999)
    assert _codes(validate_record(record, tmp_path, TAXONOMY)) == ["size_mismatch", "unknown_label"]
    missing = record.model_copy(update={"path": "nope.jpg"})
    assert "missing_file" in _codes(validate_record(missing, tmp_path, TAXONOMY))


def test_annotation_warnings(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2)),
        Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2, 0.3, 0.3)),
        Annotation(label="dent", polygon=(0.0, 0.0, 1.005, 0.0, 1.005, 1.0)),
        Annotation(label="dent", polygon=(0.1, 0.1, 0.101, 0.1, 0.101, 0.101)),
    )
    assert _codes(validate_record(record, tmp_path, TAXONOMY)) == [
        "clamped",
        "degenerate_polygon",
        "degenerate_polygon",
        "tiny_instance",
    ]


def test_far_out_of_bounds_is_an_error(tmp_path: Path) -> None:
    record = _record(tmp_path, Annotation(label="dent", polygon=(0, 0, 1.5, 0, 1.5, 1)))
    assert _codes(validate_record(record, tmp_path, TAXONOMY)) == ["out_of_bounds"]


def test_report_counts_warnings_and_collects_errors(tmp_path: Path) -> None:
    good = _record(tmp_path)
    bad = good.model_copy(update={"image_id": "s:b", "path": "missing.jpg"})
    report = validate_records([good, bad], tmp_path, TAXONOMY)
    assert not report.ok
    assert report.total_images == 2
    assert [e.code for e in report.errors] == ["missing_file"]
    assert report.warning_counts == {"no_annotations": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_validate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.validate'`

- [ ] **Step 3: Write** `src/claimlens/data/validate.py`

```python
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
        issues.append(
            Issue(image_id=record.image_id, severity=severity, code=code, message=message)
        )

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
```

- [ ] **Step 4: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_validate.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 6 passed; no issues.

```bash
git add src/claimlens/data/validate.py tests/unit/test_validate.py
git commit -m "feat: add data contract validation for image records"
```

---

### Task 7: Near-duplicate clusters and cluster-aware splits

**Files:**
- Modify: `pyproject.toml` (runtime deps `imagehash>=4.3`, `numpy>=2.0`, `pyyaml>=6.0`; mypy override for `imagehash`)
- Create: `src/claimlens/data/dedupe.py`
- Create: `src/claimlens/data/split.py`
- Test: `tests/unit/test_dedupe_split.py`

**Interfaces:**
- Consumes: `ImageRecord` (Task 3).
- Produces: `ImageHash(image_id, sha256, phash)` (dataclass); `hash_file(image_id: str, path: Path) -> ImageHash`; `find_clusters(hashes: Sequence[ImageHash], max_distance: int) -> dict[str, int]` (cluster ids are 0..k-1 in order of first appearance); `assign_splits(records, clusters, protected_clusters) -> dict[str, str]` with values `train | valid | test | excluded`.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` set:

```toml
dependencies = [
  "imagehash>=4.3",
  "numpy>=2.0",
  "pillow>=11.0",
  "pydantic>=2.8",
  "pyyaml>=6.0",
]
```

Remove `"pyyaml>=6.0",` from the `dev` group (keep `types-pyyaml`). Change the mypy override module list to `module = ["ultralytics", "ultralytics.*", "imagehash", "imagehash.*"]`.

Run: `uv sync`
Expected: installs ImageHash and its dependencies.

- [ ] **Step 2: Write the failing test** `tests/unit/test_dedupe_split.py`

```python
from pathlib import Path

from PIL import Image

from claimlens.data.dedupe import ImageHash, find_clusters, hash_file
from claimlens.data.records import ImageRecord
from claimlens.data.split import assign_splits
from tests.data_helpers import make_pattern_image


def _record(image_id: str, split: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id, source="s", source_split=split, path=f"{image_id}.jpg", width=1, height=1
    )


def test_resized_copy_is_a_near_duplicate_and_new_image_is_not(tmp_path: Path) -> None:
    original = make_pattern_image(tmp_path / "a.jpg", seed=1)
    resized = tmp_path / "a_small.jpg"
    with Image.open(original) as image:
        image.resize((256, 192)).save(resized, quality=80)
    other = make_pattern_image(tmp_path / "b.jpg", seed=2)

    hashes = [hash_file("a", original), hash_file("a_small", resized), hash_file("b", other)]
    clusters = find_clusters(hashes, max_distance=6)

    assert clusters["a"] == clusters["a_small"]
    assert clusters["b"] != clusters["a"]
    assert sorted(set(clusters.values())) == [0, 1]


def test_identical_bytes_cluster_even_with_distance_zero_threshold() -> None:
    hashes = [ImageHash("x", "same", 0b1111), ImageHash("y", "same", 0)]
    assert find_clusters(hashes, max_distance=0) == {"x": 0, "y": 0}


def test_cluster_spanning_train_and_test_goes_to_test() -> None:
    records = [_record("a", "train"), _record("b", "test"), _record("c", "train")]
    clusters = {"a": 0, "b": 0, "c": 1}
    assert assign_splits(records, clusters, protected_clusters=set()) == {
        "a": "test",
        "b": "test",
        "c": "train",
    }


def test_cluster_spanning_train_and_valid_goes_to_valid() -> None:
    records = [_record("a", "train"), _record("b", "valid")]
    assert assign_splits(records, {"a": 0, "b": 0}, set()) == {"a": "valid", "b": "valid"}


def test_protected_cluster_is_excluded() -> None:
    records = [_record("a", "train"), _record("b", "test")]
    assert assign_splits(records, {"a": 0, "b": 1}, protected_clusters={0}) == {
        "a": "excluded",
        "b": "test",
    }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_dedupe_split.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.dedupe'`

- [ ] **Step 4: Write** `src/claimlens/data/dedupe.py`

```python
"""Exact and perceptual duplicate detection, clustered with union-find."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageHash:
    image_id: str
    sha256: str
    phash: int


def hash_file(image_id: str, path: Path) -> ImageHash:
    data = path.read_bytes()
    with Image.open(path) as image:
        perceptual = imagehash.phash(image)
    return ImageHash(
        image_id=image_id,
        sha256=hashlib.sha256(data).hexdigest(),
        phash=int(str(perceptual), 16),
    )


def find_clusters(
    hashes: Sequence[ImageHash], max_distance: int, *, block: int = 512
) -> dict[str, int]:
    """Group images whose bytes match or whose pHash differs by at most `max_distance` bits."""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    first_with_sha: dict[str, int] = {}
    for i, item in enumerate(hashes):
        if item.sha256 in first_with_sha:
            union(first_with_sha[item.sha256], i)
        else:
            first_with_sha[item.sha256] = i

    values = np.array([item.phash for item in hashes], dtype=np.uint64)
    for start in range(0, len(hashes), block):
        chunk = values[start : start + block]
        distances = np.bitwise_count(chunk[:, None] ^ values[None, :])
        rows, cols = np.nonzero(distances <= max_distance)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            if start + row < col:
                union(start + row, col)

    cluster_of_root: dict[int, int] = {}
    return {
        item.image_id: cluster_of_root.setdefault(find(i), len(cluster_of_root))
        for i, item in enumerate(hashes)
    }
```

- [ ] **Step 5: Write** `src/claimlens/data/split.py`

```python
"""Assign whole near-duplicate clusters to one split so no image leaks across splits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence, Set

from claimlens.data.records import ImageRecord


def assign_splits(
    records: Sequence[ImageRecord],
    clusters: Mapping[str, int],
    protected_clusters: Set[int],
) -> dict[str, str]:
    """Test wins over valid, valid over train; protected clusters are excluded entirely."""
    source_splits: dict[int, set[str]] = defaultdict(set)
    for record in records:
        source_splits[clusters[record.image_id]].add(record.source_split)

    def target(cluster: int) -> str:
        if cluster in protected_clusters:
            return "excluded"
        splits = source_splits[cluster]
        if "test" in splits:
            return "test"
        if "valid" in splits:
            return "valid"
        return "train"

    return {record.image_id: target(clusters[record.image_id]) for record in records}
```

- [ ] **Step 6: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_dedupe_split.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 5 passed; no issues.

```bash
git add pyproject.toml uv.lock src/claimlens/data/dedupe.py src/claimlens/data/split.py tests/unit/test_dedupe_split.py
git commit -m "feat: cluster near-duplicate images and assign whole clusters to one split"
```

---

### Task 8: YOLO-seg export

**Files:**
- Create: `src/claimlens/data/export.py`
- Test: `tests/unit/test_export.py`

**Interfaces:**
- Consumes: `ImageRecord` (Task 3), `Taxonomy` (Task 2), `polygon_area` (Task 6).
- Produces: `YOLO_SPLIT = {"train": "train", "valid": "val", "test": "test"}`; `export_yolo_seg(records, splits, taxonomy, *, repo_root, out_dir) -> dict[str, int]` (counts per YOLO split plus `excluded`). Writes `images/<split>/<id>`, `labels/<split>/<id>.txt` and `data.yaml` (absolute `path`, class names from the taxonomy). The output folder is rebuilt from scratch.

- [ ] **Step 1: Write the failing test** `tests/unit/test_export.py`

```python
from pathlib import Path

from claimlens.data.export import export_yolo_seg
from claimlens.data.records import Annotation, ImageRecord
from claimlens.data.taxonomy import Taxonomy
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"))


def _record(tmp_path: Path, image_id: str, split: str, *annotations: Annotation) -> ImageRecord:
    name = image_id.split(":")[1]
    make_pattern_image(tmp_path / "raw" / f"{name}.jpg", seed=len(name))
    return ImageRecord(
        image_id=image_id,
        source="s",
        source_split=split,
        path=f"raw/{name}.jpg",
        width=320,
        height=240,
        annotations=annotations,
    )


def test_export_writes_yolo_folders_and_yaml(tmp_path: Path) -> None:
    square = Annotation(label="dent", polygon=(0.1, 0.1, 0.5, 0.1, 0.5, 0.5))
    records = [
        _record(tmp_path, "s:a", "train", square),
        _record(tmp_path, "s:bb", "valid"),
        _record(tmp_path, "s:ccc", "test"),
    ]
    splits = {"s:a": "train", "s:bb": "valid", "s:ccc": "excluded"}
    out = tmp_path / "processed" / "damage-v1"
    (out / "stale.txt").parent.mkdir(parents=True)
    (out / "stale.txt").write_text("old", encoding="utf-8")

    counts = export_yolo_seg(records, splits, TAXONOMY, repo_root=tmp_path, out_dir=out)

    assert counts == {"train": 1, "val": 1, "excluded": 1}
    assert not (out / "stale.txt").exists()
    assert (out / "images" / "train" / "s__a.jpg").is_file()
    label = (out / "labels" / "train" / "s__a.txt").read_text(encoding="utf-8")
    assert label == "1 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000\n"
    assert (out / "labels" / "val" / "s__bb.txt").read_text(encoding="utf-8") == ""
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "val: images/val" in yaml_text
    assert "  1: dent" in yaml_text


def test_export_skips_degenerate_and_clamps_polygons(tmp_path: Path) -> None:
    records = [
        _record(
            tmp_path,
            "s:a",
            "train",
            Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2)),
            Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2, 0.3, 0.3)),
            Annotation(label="crack", polygon=(0.0, -0.004, 1.005, 0.0, 1.0, 1.0)),
        )
    ]
    out = tmp_path / "out"
    export_yolo_seg(records, {"s:a": "train"}, TAXONOMY, repo_root=tmp_path, out_dir=out)
    label = (out / "labels" / "train" / "s__a.txt").read_text(encoding="utf-8")
    assert label == "0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_export.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.export'`

- [ ] **Step 3: Write** `src/claimlens/data/export.py`

```python
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
        label_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        counts[yolo_split] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.yaml").write_text(_data_yaml(out_dir, taxonomy), encoding="utf-8")
    return dict(counts)
```

- [ ] **Step 4: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_export.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 2 passed; no issues.

```bash
git add src/claimlens/data/export.py tests/unit/test_export.py
git commit -m "feat: export datasets as YOLO segmentation folders"
```

---

### Task 9: Dataset statistics

**Files:**
- Create: `src/claimlens/data/stats.py`
- Test: `tests/unit/test_stats.py`

**Interfaces:**
- Consumes: `ImageRecord` (Task 3), `Taxonomy` (Task 2).
- Produces: `dataset_stats(dataset_id, records, splits, taxonomy, *, clusters, ignored_labels, warning_counts) -> dict[str, Any]` with keys `dataset, taxonomy, images, instances, leaks_prevented, ignored_labels, validation_warnings`; `stats_markdown(stats) -> str`. `leaks_prevented` is the number of images whose final split differs from their source split.

- [ ] **Step 1: Write the failing test** `tests/unit/test_stats.py`

```python
from claimlens.data.records import Annotation, ImageRecord
from claimlens.data.stats import dataset_stats, stats_markdown
from claimlens.data.taxonomy import Taxonomy

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"))
SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5)


def _record(image_id: str, split: str, *labels: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source="s",
        source_split=split,
        path="x.jpg",
        width=1,
        height=1,
        annotations=tuple(Annotation(label=label, polygon=SQUARE) for label in labels),
    )


def test_stats_count_images_instances_and_leaks() -> None:
    records = [
        _record("a", "train", "dent", "dent"),
        _record("b", "train", "crack"),
        _record("c", "test", "dent"),
    ]
    splits = {"a": "train", "b": "test", "c": "test"}
    stats = dataset_stats(
        "damage-v1",
        records,
        splits,
        TAXONOMY,
        clusters={"a": 0, "b": 1, "c": 1},
        ignored_labels={"smash": 3},
        warning_counts={"tiny_instance": 2},
    )
    assert stats["images"] == {"train": 1, "test": 2}
    assert stats["instances"] == {"train": {"crack": 0, "dent": 2}, "test": {"crack": 1, "dent": 1}}
    assert stats["leaks_prevented"] == 1
    assert stats["clusters"] == 2
    assert stats["ignored_labels"] == {"smash": 3}

    markdown = stats_markdown(stats)
    assert "# Dataset damage-v1" in markdown
    assert "| test | 2 | 1 | 1 |" in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_stats.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.stats'`

- [ ] **Step 3: Write** `src/claimlens/data/stats.py`

```python
"""Dataset statistics, written as DVC metrics and a Markdown summary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from claimlens.data.records import ImageRecord
from claimlens.data.taxonomy import Taxonomy

_ORDER = ("train", "valid", "test", "excluded")


def dataset_stats(
    dataset_id: str,
    records: Sequence[ImageRecord],
    splits: Mapping[str, str],
    taxonomy: Taxonomy,
    *,
    clusters: Mapping[str, int],
    ignored_labels: Mapping[str, int],
    warning_counts: Mapping[str, int],
) -> dict[str, Any]:
    images: Counter[str] = Counter(splits[r.image_id] for r in records)
    instances: dict[str, dict[str, int]] = {}
    for split in _ORDER:
        if images[split]:
            counts = Counter(
                a.label for r in records if splits[r.image_id] == split for a in r.annotations
            )
            instances[split] = {label: counts[label] for label in taxonomy.classes}
    return {
        "dataset": dataset_id,
        "taxonomy": taxonomy.name,
        "images": {split: images[split] for split in _ORDER if images[split]},
        "instances": instances,
        "clusters": len(set(clusters[r.image_id] for r in records)),
        "leaks_prevented": sum(
            1 for r in records if splits[r.image_id] not in ("excluded", r.source_split)
        ),
        "ignored_labels": dict(ignored_labels),
        "validation_warnings": dict(warning_counts),
    }


def stats_markdown(stats: Mapping[str, Any]) -> str:
    classes = list(next(iter(stats["instances"].values()), {}).keys())
    lines = [
        f"# Dataset {stats['dataset']}",
        "",
        f"- Taxonomy: `{stats['taxonomy']}`",
        f"- Near-duplicate clusters: {stats['clusters']}",
        f"- Images moved to another split to prevent leakage: {stats['leaks_prevented']}",
        f"- Ignored labels: {stats['ignored_labels'] or 'none'}",
        f"- Validation warnings: {stats['validation_warnings'] or 'none'}",
        "",
        "| Split | Images | " + " | ".join(classes) + " |",
        "|---" * (len(classes) + 2) + "|",
    ]
    for split, count in stats["images"].items():
        per_class = stats["instances"].get(split, {})
        cells = " | ".join(str(per_class.get(c, 0)) for c in classes)
        lines.append(f"| {split} | {count} | {cells} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_stats.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 1 passed; no issues.

```bash
git add src/claimlens/data/stats.py tests/unit/test_stats.py
git commit -m "feat: add dataset statistics and Markdown summary"
```

---

### Task 10: Source configuration and fetching

**Files:**
- Create: `config/datasets.toml`
- Create: `src/claimlens/data/config.py`
- Create: `src/claimlens/data/fetch.py`
- Test: `tests/unit/test_data_config_fetch.py`

**Interfaces:**
- Produces (`config.py`): `SplitLayout(images=None, labels=None, annotations=None)`; `SourceConfig(id, kind, format, terms, splits, class_names_file=None, url=None, workspace=None, project=None, version=None, export_format=None)`; `DatasetConfig(id, taxonomy, sources, protect_golden=None, phash_max_distance=6)`; `DataConfig(sources, datasets)`; `load_data_config(path) -> DataConfig`; `read_secret(name, env_file=Path(".env")) -> str | None`.
- Produces (`fetch.py`): `HttpGet = Callable[[str], bytes]`; `roboflow_download_url(source, api_key, http_get) -> str`; `extract_zip(data: bytes, dest: Path) -> None`; `write_manifest(root: Path) -> Path`; `fetch_source(source, raw_root, *, api_key=None, http_get=urllib_get) -> Path`.

- [ ] **Step 1: Write** `config/datasets.toml`

```toml
# Data sources and the datasets built from them. Split paths are relative to data/raw/<source id>/.

[source.cardd-roboflow-v6]
kind = "roboflow"
format = "coco-seg"
workspace = "auto-industry"
project = "car-damage-detection-vyhvw"
version = 6
export_format = "coco-segmentation"
terms = "CarDD (Wang et al., IEEE T-ITS 2023) re-uploaded to Roboflow Universe, labelled CC BY 4.0 there. Used under CarDD terms: non-commercial research and education only; images are not redistributed."

[source.cardd-roboflow-v6.splits]
train = { annotations = "train/_annotations.coco.json" }
valid = { annotations = "valid/_annotations.coco.json" }
test = { annotations = "test/_annotations.coco.json" }

[source.carparts-seg]
kind = "url"
format = "yolo-seg"
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
class_names_file = "carparts-seg.yaml"
terms = "Roboflow Universe car-seg by Gianmarco Russo, CC BY 4.0, as distributed by Ultralytics."

[source.carparts-seg.splits]
train = { images = "images/train", labels = "labels/train" }
valid = { images = "images/val", labels = "labels/val" }
test = { images = "images/test", labels = "labels/test" }

[source.legacy-course-subset]
kind = "local"
format = "yolo-seg"
class_names_file = "data.yaml"
terms = "Roboflow Universe exports used by the original course project, CC BY 4.0, mixed sources."

[source.legacy-course-subset.splits]
train = { images = "train/images", labels = "train/labels" }
valid = { images = "valid/images", labels = "valid/labels" }
test = { images = "test/images", labels = "test/labels" }

[dataset.damage-v1]
taxonomy = "damage"
sources = ["cardd-roboflow-v6"]
protect_golden = "evals/golden/v0/claims.jsonl"

[dataset.parts-v1]
taxonomy = "parts"
sources = ["carparts-seg"]
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_data_config_fetch.py`

```python
import io
import json
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.config import SourceConfig, load_data_config, read_secret
from claimlens.data.fetch import extract_zip, fetch_source, roboflow_download_url

ROOT = Path(__file__).resolve().parents[2]


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _url_source() -> SourceConfig:
    return SourceConfig.model_validate(
        {
            "id": "parts",
            "kind": "url",
            "format": "yolo-seg",
            "url": "https://example.test/parts.zip",
            "terms": "test",
            "splits": {"train": {"images": "images/train", "labels": "labels/train"}},
        }
    )


def test_repo_config_loads() -> None:
    config = load_data_config(ROOT / "config" / "datasets.toml")
    cardd = config.sources["cardd-roboflow-v6"]
    assert cardd.splits["test"].annotations == "test/_annotations.coco.json"
    assert config.datasets["damage-v1"].sources == ("cardd-roboflow-v6",)
    assert config.datasets["damage-v1"].phash_max_distance == 6


def test_yolo_source_requires_images_and_labels() -> None:
    with pytest.raises(ValidationError, match="images and labels"):
        SourceConfig.model_validate(
            {
                "id": "x",
                "kind": "local",
                "format": "yolo-seg",
                "terms": "t",
                "splits": {"train": {}},
            }
        )


def test_read_secret_prefers_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nROBOFLOW_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    assert read_secret("ROBOFLOW_API_KEY", env_file) == "from-file"
    monkeypatch.setenv("ROBOFLOW_API_KEY", "from-env")
    assert read_secret("ROBOFLOW_API_KEY", env_file) == "from-env"
    assert read_secret("MISSING_KEY", env_file) is None


def test_roboflow_url_comes_from_the_export_endpoint() -> None:
    config = load_data_config(ROOT / "config" / "datasets.toml")
    requested: list[str] = []

    def fake_get(url: str) -> bytes:
        requested.append(url)
        return json.dumps({"export": {"link": "https://download.test/x.zip"}}).encode()

    link = roboflow_download_url(config.sources["cardd-roboflow-v6"], "KEY", fake_get)
    assert link == "https://download.test/x.zip"
    assert requested[0].startswith(
        "https://api.roboflow.com/auto-industry/car-damage-detection-vyhvw/6/coco-segmentation"
    )


def test_fetch_extracts_and_writes_manifest(tmp_path: Path) -> None:
    archive = _zip({"images/train/a.jpg": b"jpeg-bytes", "labels/train/a.txt": b"0 0 0 1 0 1 1"})

    dest = fetch_source(_url_source(), tmp_path, http_get=lambda _url: archive)

    assert (dest / "images" / "train" / "a.jpg").read_bytes() == b"jpeg-bytes"
    manifest = json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in manifest] == ["images/train/a.jpg", "labels/train/a.txt"]
    assert manifest[0]["bytes"] == 10


def test_fetch_refuses_to_overwrite_raw_data(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        fetch_source(_url_source(), tmp_path, http_get=lambda _url: b"")


def test_extract_refuses_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        extract_zip(_zip({"../../evil.txt": b"x"}), tmp_path / "dest")
    assert not (tmp_path / "evil.txt").exists()


def test_roboflow_fetch_needs_a_key(tmp_path: Path) -> None:
    config = load_data_config(ROOT / "config" / "datasets.toml")
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        fetch_source(config.sources["cardd-roboflow-v6"], tmp_path, http_get=lambda _url: b"")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_data_config_fetch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.config'`

- [ ] **Step 4: Write** `src/claimlens/data/config.py`

```python
"""Data source and dataset configuration (config/datasets.toml) and secret lookup."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from claimlens.domain import Frozen


class SplitLayout(Frozen):
    images: str | None = None
    labels: str | None = None
    annotations: str | None = None


class SourceConfig(Frozen):
    id: str
    kind: Literal["roboflow", "url", "local"]
    format: Literal["coco-seg", "yolo-seg"]
    terms: str
    splits: dict[str, SplitLayout]
    class_names_file: str | None = None
    url: str | None = None
    workspace: str | None = None
    project: str | None = None
    version: int | None = None
    export_format: str | None = None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        for name, layout in self.splits.items():
            if self.format == "coco-seg" and not layout.annotations:
                raise ValueError(f"{self.id}/{name}: coco-seg splits need an annotations file")
            if self.format == "yolo-seg" and not (layout.images and layout.labels):
                raise ValueError(f"{self.id}/{name}: yolo-seg splits need images and labels")
        if self.kind == "url" and not self.url:
            raise ValueError(f"{self.id}: url sources need a url")
        if self.kind == "roboflow" and not (
            self.workspace and self.project and self.version and self.export_format
        ):
            raise ValueError(f"{self.id}: roboflow sources need workspace, project, version")
        return self


class DatasetConfig(Frozen):
    id: str
    taxonomy: str
    sources: tuple[str, ...]
    protect_golden: str | None = None
    phash_max_distance: int = 6


class DataConfig(Frozen):
    sources: dict[str, SourceConfig]
    datasets: dict[str, DatasetConfig]


def load_data_config(path: Path) -> DataConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return DataConfig(
        sources={
            key: SourceConfig.model_validate({"id": key, **body})
            for key, body in data.get("source", {}).items()
        },
        datasets={
            key: DatasetConfig.model_validate({"id": key, **body})
            for key, body in data.get("dataset", {}).items()
        },
    )


def read_secret(name: str, env_file: Path = Path(".env")) -> str | None:
    """Environment variable first, then a KEY=VALUE line in the .env file."""
    value = os.environ.get(name)
    if value:
        return value
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, raw = line.strip().partition("=")
            if sep and key.strip() == name and not key.startswith("#"):
                return raw.strip().strip('"').strip("'") or None
    return None
```

- [ ] **Step 5: Write** `src/claimlens/data/fetch.py`

```python
"""Download raw data sources into data/raw/<id>/ and record a checksum manifest."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from claimlens.data.config import SourceConfig

HttpGet = Callable[[str], bytes]


def urllib_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as response:
        data: bytes = response.read()
    return data


def roboflow_download_url(source: SourceConfig, api_key: str, http_get: HttpGet) -> str:
    endpoint = (
        f"https://api.roboflow.com/{source.workspace}/{source.project}/"
        f"{source.version}/{source.export_format}?api_key={api_key}"
    )
    try:
        payload = json.loads(http_get(endpoint))
    except Exception as exc:
        # Never echo the endpoint: it contains the API key.
        raise RuntimeError(f"Roboflow export request failed: {type(exc).__name__}") from None
    return str(payload["export"]["link"])


def extract_zip(data: bytes, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"zip entry {member.filename!r} would extract outside {dest}")
        archive.extractall(root)


def write_manifest(root: Path) -> Path:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    ]
    manifest = root / "MANIFEST.json"
    manifest.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    return manifest


def fetch_source(
    source: SourceConfig,
    raw_root: Path,
    *,
    api_key: str | None = None,
    http_get: HttpGet = urllib_get,
) -> Path:
    dest = raw_root / source.id
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} already exists; raw data is immutable, delete it to refetch")
    if source.kind == "local":
        raise ValueError(f"{source.id} is a local source and cannot be downloaded")
    if source.kind == "roboflow":
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set (environment or .env)")
        url = roboflow_download_url(source, api_key, http_get)
    else:
        url = str(source.url)
    extract_zip(http_get(url), dest)
    write_manifest(dest)
    return dest
```

- [ ] **Step 6: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_data_config_fetch.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 8 passed; no issues.

```bash
git add config/datasets.toml src/claimlens/data/config.py src/claimlens/data/fetch.py tests/unit/test_data_config_fetch.py
git commit -m "feat: add data source config and safe, immutable fetching"
```

---

### Task 11: Build pipeline and `claimlens data` commands

**Files:**
- Create: `src/claimlens/data/pipeline.py`
- Modify: `src/claimlens/cli.py` (add `data fetch` and `data build`)
- Test: `tests/unit/test_data_pipeline.py`

**Interfaces:**
- Consumes: Tasks 2-10, `load_golden` (M1).
- Produces: `DataContractError(Exception)` with `.report`; `BuildResult(dataset_id, stats, validation)`; `convert_source(source, taxonomy, *, repo_root) -> ConversionResult`; `golden_image_paths(golden_file, repo_root) -> list[Path]`; `build_dataset(dataset_id, *, repo_root, config_dir) -> BuildResult`. Writes `data/interim/<id>/{records.jsonl, clusters.json, splits.json}`, `data/processed/<id>/`, `reports/data/<id>-{validation.json, stats.json, stats.md}`. CLI: `claimlens data fetch <source>` (exit 2 if the folder exists), `claimlens data build <dataset>` (exit 1 on a contract violation, listing up to 10 errors).

- [ ] **Step 1: Write the failing test** `tests/unit/test_data_pipeline.py`

```python
import json
from pathlib import Path

import pytest
from PIL import Image

from claimlens.cli import main
from claimlens.data.pipeline import DataContractError, build_dataset
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, write_golden
from tests.data_helpers import make_pattern_image

ROOT = Path(__file__).resolve().parents[2]
SQUARE = "0 0.1 0.1 0.5 0.1 0.5 0.5\n"


def _repo(tmp_path: Path) -> Path:
    """A tiny repo: one YOLO source with a train/test duplicate and a golden lookalike."""
    raw = tmp_path / "data" / "raw" / "toy"
    for split, names in {"train": ["a", "b", "g"], "test": ["c"]}.items():
        for name in names:
            seed = {"a": 1, "b": 2, "c": 1, "g": 3}[name]
            make_pattern_image(raw / split / "images" / f"{name}.jpg", seed=seed)
            (raw / split / "labels").mkdir(parents=True, exist_ok=True)
            (raw / split / "labels" / f"{name}.txt").write_text(SQUARE, encoding="utf-8")
    (raw / "data.yaml").write_text("names: ['dent']\n", encoding="utf-8")

    golden_photo = tmp_path / "golden" / "g.jpg"
    golden_photo.parent.mkdir()
    with Image.open(raw / "train" / "images" / "g.jpg") as image:
        image.resize((300, 225)).save(golden_photo, quality=85)
    write_golden(
        tmp_path / "golden.jsonl",
        [
            GoldenClaim(
                case_id="g001",
                scenario="s",
                policy_id="P-1001",
                description="",
                photos=("golden/g.jpg",),
                expected_route=Route.FAST_TRACK,
                label_source="scenario",
            )
        ],
    )

    config = tmp_path / "config"
    config.mkdir()
    (config / "taxonomy.toml").write_text(
        'version = "t"\n[damage]\nclasses = ["dent"]\n', encoding="utf-8"
    )
    (config / "datasets.toml").write_text(
        """
[source.toy]
kind = "local"
format = "yolo-seg"
class_names_file = "data.yaml"
terms = "test"
[source.toy.splits]
train = { images = "train/images", labels = "train/labels" }
test = { images = "test/images", labels = "test/labels" }

[dataset.toy-v1]
taxonomy = "damage"
sources = ["toy"]
protect_golden = "golden.jsonl"
""",
        encoding="utf-8",
    )
    return config


def test_build_moves_cross_split_duplicates_to_test(tmp_path: Path) -> None:
    config = _repo(tmp_path)
    result = build_dataset("toy-v1", repo_root=tmp_path, config_dir=config)

    splits = json.loads((tmp_path / "data" / "interim" / "toy-v1" / "splits.json").read_text())
    assert splits["toy:a"] == "test"
    assert splits["toy:c"] == "test"
    assert splits["toy:b"] == "train"
    assert result.stats["leaks_prevented"] == 1
    assert (
        tmp_path / "data" / "processed" / "toy-v1" / "images" / "train" / "toy__b.jpg"
    ).is_file()
    assert (tmp_path / "reports" / "data" / "toy-v1-stats.md").is_file()


def test_build_excludes_images_near_golden_photos(tmp_path: Path) -> None:
    config = _repo(tmp_path)
    build_dataset("toy-v1", repo_root=tmp_path, config_dir=config)

    splits = json.loads((tmp_path / "data" / "interim" / "toy-v1" / "splits.json").read_text())
    assert splits["toy:g"] == "excluded"
    exported = list((tmp_path / "data" / "processed" / "toy-v1" / "images").rglob("toy__g.*"))
    assert exported == []


def test_build_stops_on_contract_violation(tmp_path: Path) -> None:
    config = _repo(tmp_path)
    labels = tmp_path / "data" / "raw" / "toy" / "train" / "labels" / "b.txt"
    labels.write_text("0 0 0 1.8 0 1.8 1\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="1 error"):
        build_dataset("toy-v1", repo_root=tmp_path, config_dir=config)


def test_data_build_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config), "data", "build", "toy-v1"]) == 0
    assert "Built toy-v1" in capsys.readouterr().out


def test_data_fetch_refuses_existing_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "data" / "raw" / "carparts-seg").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "carparts-seg" / "x").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(ROOT / "config"), "data", "fetch", "carparts-seg"]) == 2
    assert "immutable" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_data_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.data.pipeline'`

- [ ] **Step 3: Write** `src/claimlens/data/pipeline.py`

```python
"""Build a dataset end to end: convert, validate, deduplicate, split, export, report."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claimlens.data.config import SourceConfig, load_data_config
from claimlens.data.convert_coco import convert_coco_split
from claimlens.data.convert_yolo import convert_yolo_split, read_class_names
from claimlens.data.dedupe import find_clusters, hash_file
from claimlens.data.export import export_yolo_seg
from claimlens.data.records import ConversionResult, ImageRecord, relative_posix, write_records
from claimlens.data.split import assign_splits
from claimlens.data.stats import dataset_stats, stats_markdown
from claimlens.data.taxonomy import Taxonomy, load_taxonomies
from claimlens.data.validate import ValidationReport, validate_records
from claimlens.evals.golden import load_golden

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class DataContractError(Exception):
    def __init__(self, report: ValidationReport) -> None:
        super().__init__(f"data contract failed with {len(report.errors)} error(s)")
        self.report = report


@dataclass(frozen=True)
class BuildResult:
    dataset_id: str
    stats: dict[str, Any]
    validation: ValidationReport


def convert_source(
    source: SourceConfig, taxonomy: Taxonomy, *, repo_root: Path
) -> ConversionResult:
    root = repo_root / "data" / "raw" / source.id
    combined = ConversionResult(records=[])
    for split, layout in source.splits.items():
        if source.format == "coco-seg":
            result = convert_coco_split(
                root / str(layout.annotations),
                source=source.id,
                split=split,
                taxonomy=taxonomy,
                repo_root=repo_root,
            )
        else:
            names = read_class_names(root / str(source.class_names_file))
            result = convert_yolo_split(
                root / str(layout.images),
                root / str(layout.labels),
                source=source.id,
                split=split,
                class_names=names,
                taxonomy=taxonomy,
                repo_root=repo_root,
            )
        combined.records.extend(result.records)
        combined.ignored_labels.update(result.ignored_labels)
    return combined


def golden_image_paths(golden_file: Path, repo_root: Path) -> list[Path]:
    paths: dict[str, Path] = {}
    for case in load_golden(golden_file):
        photos = [*case.photos, *(p for prior in case.prior_claims for p in prior.photos)]
        for photo in photos:
            path = repo_root / photo
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                paths.setdefault(path.as_posix(), path)
    return list(paths.values())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def build_dataset(dataset_id: str, *, repo_root: Path, config_dir: Path) -> BuildResult:
    config = load_data_config(config_dir / "datasets.toml")
    dataset = config.datasets[dataset_id]
    taxonomy = load_taxonomies(config_dir / "taxonomy.toml")[dataset.taxonomy]
    reports = repo_root / "reports" / "data"

    records: list[ImageRecord] = []
    ignored: Counter[str] = Counter()
    for source_id in dataset.sources:
        converted = convert_source(config.sources[source_id], taxonomy, repo_root=repo_root)
        records.extend(converted.records)
        ignored.update(converted.ignored_labels)
    records.sort(key=lambda r: r.image_id)

    validation = validate_records(records, repo_root, taxonomy)
    _write_json(reports / f"{dataset_id}-validation.json", validation.model_dump(mode="json"))
    if not validation.ok:
        raise DataContractError(validation)

    hashes = [hash_file(r.image_id, repo_root / r.path) for r in records]
    golden_ids: list[str] = []
    if dataset.protect_golden:
        for path in golden_image_paths(repo_root / dataset.protect_golden, repo_root):
            golden_id = f"golden:{relative_posix(path, repo_root)}"
            hashes.append(hash_file(golden_id, path))
            golden_ids.append(golden_id)
    clusters = find_clusters(hashes, dataset.phash_max_distance)
    protected = {clusters[g] for g in golden_ids}
    splits = assign_splits(records, clusters, protected)

    interim = repo_root / "data" / "interim" / dataset_id
    write_records(interim / "records.jsonl", records)
    _write_json(
        interim / "clusters.json",
        {
            h.image_id: {
                "cluster": clusters[h.image_id],
                "sha256": h.sha256,
                "phash": f"{h.phash:016x}",
            }
            for h in hashes
        },
    )
    _write_json(interim / "splits.json", splits)

    export_yolo_seg(
        records,
        splits,
        taxonomy,
        repo_root=repo_root,
        out_dir=repo_root / "data" / "processed" / dataset_id,
    )
    stats = dataset_stats(
        dataset_id,
        records,
        splits,
        taxonomy,
        clusters=clusters,
        ignored_labels=ignored,
        warning_counts=validation.warning_counts,
    )
    _write_json(reports / f"{dataset_id}-stats.json", stats)
    (reports / f"{dataset_id}-stats.md").write_text(stats_markdown(stats), encoding="utf-8")
    return BuildResult(dataset_id=dataset_id, stats=stats, validation=validation)
```

- [ ] **Step 4: Add the `data` commands to** `src/claimlens/cli.py`

Add imports (merge with existing):

```python
from claimlens.data.config import load_data_config, read_secret
from claimlens.data.fetch import fetch_source
from claimlens.data.pipeline import DataContractError, build_dataset
```

In `build_parser()`, before `return parser`, add:

```python
    data = sub.add_parser("data", help="fetch and build datasets")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="download a raw data source into data/raw/")
    fetch.add_argument("source_id")
    build = data_sub.add_parser("build", help="build a dataset from raw sources")
    build.add_argument("dataset_id")
```

In `main()`, directly after the `eval-triage` dispatch, add:

```python
    if args.command == "data":
        return _data(args)
```

Append:

```python
def _data(args: argparse.Namespace) -> int:
    repo_root = Path.cwd()
    if args.data_command == "fetch":
        source = load_data_config(args.config / "datasets.toml").sources[args.source_id]
        try:
            dest = fetch_source(
                source, repo_root / "data" / "raw", api_key=read_secret("ROBOFLOW_API_KEY")
            )
        except (FileExistsError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Fetched {args.source_id} into {dest}")
        return 0
    try:
        result = build_dataset(args.dataset_id, repo_root=repo_root, config_dir=args.config)
    except DataContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for issue in exc.report.errors[:10]:
            print(f"  {issue.image_id}: {issue.code}: {issue.message}", file=sys.stderr)
        return 1
    print(f"Built {result.dataset_id}: images {result.stats['images']}")
    print(f"Leaks prevented: {result.stats['leaks_prevented']}")
    return 0
```

Update the module docstring to `"""Command-line interface: `claimlens run | resume | show | verify | eval-triage | data`."""`.

- [ ] **Step 5: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_data_pipeline.py -q && uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q`
Expected: 5 passed; full suite passes; coverage gate holds.

```bash
git add src/claimlens/data/pipeline.py src/claimlens/cli.py tests/unit/test_data_pipeline.py
git commit -m "feat: add dataset build pipeline and claimlens data commands"
```

---

### Task 12: DVC, real builds and the data card

**Files:**
- Modify: `pyproject.toml` (dev dependency `dvc>=3.50`)
- Modify: `.gitignore` (allow `.dvc` pointer files under `data/`)
- Create: `.dvc/` (via `dvc init`), `.dvcignore`, `dvc.yaml`, `dvc.lock`, `data/raw/*.dvc`
- Create: `reports/data/*` (generated, committed)
- Create: `docs/data-card.md`, `docs/adr/0004-data-sources-and-licensing.md`, `docs/adr/0005-dvc-for-data-versioning.md`
- Modify: `data/README.md`, `docs/roadmap.md`

**Interfaces:**
- Consumes: everything above.
- Produces: versioned `damage-v1` and `parts-v1`, reproducible with `uv run dvc repro`.

- [ ] **Step 1: Allow DVC pointer files through `.gitignore`**

Replace the `/data/*` and `!/data/README.md` lines with:

```
/data/**
!/data/**/
!/data/README.md
!/data/**/*.dvc
!/data/**/.gitignore
```

- [ ] **Step 2: Install and initialise DVC with a local remote**

Add `"dvc>=3.50",` to the `dev` group, then run:

```bash
uv sync
uv run dvc init
uv run dvc remote add -d localstore ../claimlens-dvc-remote
```

Expected: `.dvc/config` contains `remote = localstore`.

- [ ] **Step 3: Fetch the sources and track the raw snapshots**

```bash
uv run claimlens data fetch cardd-roboflow-v6
uv run claimlens data fetch carparts-seg
uv run dvc add data/raw/cardd-roboflow-v6 data/raw/carparts-seg data/raw/legacy-course-subset
```

Expected: three `.dvc` files under `data/raw/` and `data/raw/.gitignore`.

- [ ] **Step 4: Write** `dvc.yaml`

```yaml
stages:
  build-damage-v1:
    cmd: uv run claimlens data build damage-v1
    deps:
      - data/raw/cardd-roboflow-v6
      - data/raw/legacy-course-subset
      - evals/golden/v0/claims.jsonl
      - config/datasets.toml
      - config/taxonomy.toml
      - src/claimlens/data
    outs:
      - data/interim/damage-v1
      - data/processed/damage-v1
      - reports/data/damage-v1-validation.json:
          cache: false
      - reports/data/damage-v1-stats.md:
          cache: false
    metrics:
      - reports/data/damage-v1-stats.json:
          cache: false
  build-parts-v1:
    cmd: uv run claimlens data build parts-v1
    deps:
      - data/raw/carparts-seg
      - config/datasets.toml
      - config/taxonomy.toml
      - src/claimlens/data
    outs:
      - data/interim/parts-v1
      - data/processed/parts-v1
      - reports/data/parts-v1-validation.json:
          cache: false
      - reports/data/parts-v1-stats.md:
          cache: false
    metrics:
      - reports/data/parts-v1-stats.json:
          cache: false
```

- [ ] **Step 5: Build both datasets and push to the remote**

Run: `uv run dvc repro`
Expected: both stages run; `reports/data/damage-v1-stats.md` and `parts-v1-stats.md` exist. If a build stops on a contract violation, read the listed errors, decide whether the data or the contract is wrong, record the ruling, and fix that one cause.

Run: `uv run dvc metrics show && uv run dvc push`
Expected: metrics table printed; push succeeds.

- [ ] **Step 6: Write the documentation from the real numbers**

Write `docs/data-card.md` following *Datasheets for Datasets* (motivation, composition with the real counts from `reports/data/*-stats.md`, collection and sources, preprocessing and cleaning (taxonomy, contract, dedupe, split rule), uses and out-of-scope uses, distribution and licence terms per source, maintenance). Write ADR 0004 (data sources and licensing: the Roboflow v6 = CarDD finding, terms applied, legacy subset role, parts source, Ultralytics AGPL noted for M3) and ADR 0005 (DVC with a local remote; DagsHub as the one-command off-site option: `uv run dvc remote add -d dagshub https://dagshub.com/<user>/claimlens.dvc`; Google Drive blocked). Update `data/README.md` with the three sources and the `dvc pull` / `dvc repro` workflow, and tick the M2 items in `docs/roadmap.md` that this plan delivered.

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml uv.lock .dvc .dvcignore dvc.yaml dvc.lock data/raw/*.dvc data/raw/.gitignore reports/data docs data/README.md
git commit -m "feat: version datasets with DVC and build damage-v1 and parts-v1"
```
