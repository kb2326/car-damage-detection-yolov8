# M2b Review, Verified Auto-Labelling and Golden v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Part-group masks on 100 frozen CarDD test photos, proposed by Grounding DINO + SAM 2 and approved or rejected by a person in FiftyOne (`fusion-eval-v1`); golden claims v1 with duplicates removed and 50 CarDD test claims added; a human review of golden cases; a re-run baseline.

**Architecture:** A pure post-processing module turns open-vocabulary detections into part-group annotations (exact phrase mapping, per-group NMS, mask → polygon). A thin, coverage-omitted adapter runs the two foundation models on CPU. A job runner samples images deterministically and writes proposals as `ImageRecord`s with scores. Review decisions are plain JSON files committed to git; pure functions apply them. FiftyOne is used only as the review UI behind a thin adapter. Golden-photo protection is corrected to never alter the frozen test split.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, OpenCV (headless), Hugging Face `transformers` 5 (Grounding DINO tiny, SAM 2.1 tiny), PyTorch (CPU), FiftyOne 1.x, DVC.

**Spec:** [`docs/specs/2026-10-01-m2-data-engine-design.md`](../specs/2026-10-01-m2-data-engine-design.md) section 5, as refined by the approved M2b design in conversation (CPU, coarse part groups, approve/reject review, golden v1 from the frozen test split, protection keeps test intact).

## Global Constraints

- All M1 and M2a Global Constraints apply.
- New optional dependency groups: `autolabel` = `transformers>=5.0`, `torch>=2.4`, `opencv-python-headless>=4.9`; `review` = `fiftyone>=1.0`. `opencv-python-headless>=4.9` is also added to `dev` so mask tests run in CI. Nothing new in runtime dependencies.
- Code that imports `torch`, `transformers` or `fiftyone` lives only in `src/claimlens/autolabel/grounded_sam.py` and `src/claimlens/review/fiftyone_app.py`, imports lazily, and is omitted from coverage.
- Review decisions are committed under `reviews/` (human labour is versioned data); auto-label proposals are DVC outputs.
- Golden-photo protection must never move or exclude an image whose cluster contains a test-split image.
- Phrases from Grounding DINO map to part groups only by exact match; merged or unknown phrases are dropped and counted, never guessed.

## Review Focus

1. **A merged phrase such as `"front bumper rear bumper"`** must be dropped and counted, not mapped to either group (Task 3, `test_merged_phrase_is_dropped`).
2. **An annotation the reviewer did not decide** must keep its whole image out of the evaluation set, not be treated as approved (Task 6, `test_unreviewed_image_is_left_out`).
3. **A CarDD test image used in golden v1** must stay in the `damage-v1` test split (Task 1, `test_protected_cluster_with_test_member_stays_test`).
4. **A SAM mask that is empty or smaller than 3 points** must be dropped with a count, not written as a polygon (Task 3, `test_empty_mask_gives_no_polygon`; Task 4 counts `no_mask`).
5. **Running the review commands without FiftyOne installed** must print how to install it and exit 2, not raise a traceback (Task 7, `test_review_launch_without_fiftyone`).

---

## File structure

```
config/taxonomy.toml               + [part_groups] (Task 2)
config/autolabel.toml              auto-labelling job fusion-eval-v1 (Task 5)
src/claimlens/data/records.py      Annotation.score (Task 2)
src/claimlens/data/taxonomy.py     PartGroups, load_part_groups (Task 2)
src/claimlens/data/split.py        protection keeps test (Task 1)
src/claimlens/autolabel/
  __init__.py
  postprocess.py                   Detection, select_detections, mask_to_polygon (Task 3)
  labeller.py                      PartLabeller protocol (Task 4)
  grounded_sam.py                  GroundedSamLabeller adapter (Task 4, coverage omitted)
  job.py                           AutolabelJob, select_sample, run_autolabel (Task 5)
src/claimlens/review/
  __init__.py
  decisions.py                     PartReview, GoldenReview, apply_* (Task 6)
  fiftyone_app.py                  FiftyOne launch/export (Task 7, coverage omitted)
src/claimlens/evals/oracle.py      findings_from_record (Task 8)
src/claimlens/cli.py               data autolabel, review launch|export|apply (Tasks 5, 7)
scripts/build_golden_v1.py         (Task 8)
evals/golden/v1/                   claims.jsonl, README.md (Task 8)
reviews/                           fusion-eval-v1.json, golden-v1.json (Task 10)
```

---

### Task 1: Golden protection keeps the frozen test split

**Files:**
- Modify: `src/claimlens/data/split.py`
- Modify: `tests/unit/test_dedupe_split.py`
- Modify: `docs/data-card.md` (split rule sentence)

**Interfaces:**
- Produces: `assign_splits` unchanged in signature; a protected cluster that contains a test member is assigned `test`.

- [ ] **Step 1: Add the failing test** to `tests/unit/test_dedupe_split.py`

```python
def test_protected_cluster_with_test_member_stays_test() -> None:
    records = [_record("a", "train"), _record("b", "test")]
    assert assign_splits(records, {"a": 0, "b": 0}, protected_clusters={0}) == {
        "a": "test",
        "b": "test",
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_dedupe_split.py -q`
Expected: 1 failed (`'excluded' != 'test'`).

- [ ] **Step 3: Change** `target()` in `src/claimlens/data/split.py` to:

```python
    def target(cluster: int) -> str:
        splits = source_splits[cluster]
        if "test" in splits:
            # The frozen benchmark split is never altered, even next to a golden photo.
            return "test"
        if cluster in protected_clusters:
            return "excluded"
        if "valid" in splits:
            return "valid"
        return "train"
```

and its docstring to `"""Test wins over everything; protected clusters leave train and valid; valid wins over train."""`.

- [ ] **Step 4: Update the data card** split-rule line (Preprocessing step 4) to: `a whole cluster goes to test if any member is in the source test split, else it is excluded if it contains a golden photo, else validation, else train. The frozen test split is never altered.`

- [ ] **Step 5: Run the suite and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add src/claimlens/data/split.py tests/unit/test_dedupe_split.py docs/data-card.md
git commit -m "fix: golden protection never alters the frozen test split"
```

---

### Task 2: Annotation scores and part groups

**Files:**
- Modify: `src/claimlens/data/records.py` (`Annotation.score`)
- Modify: `config/taxonomy.toml` (`[part_groups]`)
- Modify: `src/claimlens/data/taxonomy.py` (`PartGroups`, `load_part_groups`, skip `part_groups` in `load_taxonomies`)
- Test: `tests/unit/test_part_groups.py`

**Interfaces:**
- Produces: `Annotation(label, polygon, score: float | None = None)`; `PartGroups(classes, members)` with `group_of(part: str) -> str` and `as_taxonomy() -> Taxonomy`; `load_part_groups(path) -> PartGroups`.

- [ ] **Step 1: Add to** `config/taxonomy.toml`

```toml

[part_groups]
# Coarse groups for damage-to-part fusion and pricing. Zero-shot models cannot reliably tell
# left from right, and pricing does not need it. Every parts class belongs to exactly one group.
classes = ["front_bumper", "rear_bumper", "door", "hood", "trunk", "light", "glass", "mirror", "wheel"]

[part_groups.members]
front_bumper = ["front_bumper"]
rear_bumper = ["back_bumper"]
door = ["back_door", "back_left_door", "back_right_door", "front_door", "front_left_door", "front_right_door"]
hood = ["hood"]
trunk = ["tailgate", "trunk"]
light = ["back_left_light", "back_light", "back_right_light", "front_left_light", "front_light", "front_right_light"]
glass = ["back_glass", "front_glass"]
mirror = ["left_mirror", "right_mirror"]
wheel = ["wheel"]
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_part_groups.py`

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.records import Annotation
from claimlens.data.taxonomy import PartGroups, UnknownLabelError, load_part_groups, load_taxonomies

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_FILE = ROOT / "config" / "taxonomy.toml"


def test_every_part_class_belongs_to_exactly_one_group() -> None:
    groups = load_part_groups(TAXONOMY_FILE)
    parts = load_taxonomies(TAXONOMY_FILE)["parts"]
    members = [part for group in groups.members.values() for part in group]
    assert sorted(members) == sorted(parts.classes)
    assert groups.group_of("front_left_door") == "door"
    assert groups.as_taxonomy().classes == groups.classes


def test_unknown_part_raises() -> None:
    with pytest.raises(UnknownLabelError, match="fender"):
        load_part_groups(TAXONOMY_FILE).group_of("fender")


def test_groups_must_match_members() -> None:
    with pytest.raises(ValidationError, match="members"):
        PartGroups(classes=("door",), members={"hood": ("hood",)})
    with pytest.raises(ValidationError, match="more than one group"):
        PartGroups(classes=("a", "b"), members={"a": ("x",), "b": ("x",)})


def test_taxonomies_ignore_the_part_groups_section() -> None:
    assert "part_groups" not in load_taxonomies(TAXONOMY_FILE)


def test_annotation_score_is_optional() -> None:
    assert Annotation(label="door", polygon=(0, 0, 1, 0, 1, 1)).score is None
    assert Annotation(label="door", polygon=(0, 0, 1, 0, 1, 1), score=0.5).score == 0.5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_part_groups.py -q`
Expected: FAIL with `ImportError: cannot import name 'PartGroups'`.

- [ ] **Step 4: Implement**

In `src/claimlens/data/records.py` add to `Annotation`:

```python
    score: float | None = None
```

In `src/claimlens/data/taxonomy.py`, change the comprehension condition in `load_taxonomies` to `if isinstance(body, dict) and name != "part_groups"`, and append:

```python
class PartGroups(Frozen):
    """Coarse part groups, each a set of fine-grained part classes."""

    classes: tuple[str, ...]
    members: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.members) != set(self.classes):
            raise ValueError("part group members must be defined for exactly the group classes")
        seen: set[str] = set()
        for parts in self.members.values():
            for part in parts:
                if part in seen:
                    raise ValueError(f"part {part!r} is in more than one group")
                seen.add(part)
        return self

    def group_of(self, part: str) -> str:
        for group, parts in self.members.items():
            if part in parts:
                return group
        raise UnknownLabelError("part_groups", part)

    def as_taxonomy(self) -> Taxonomy:
        return Taxonomy(name="part_groups", classes=self.classes)


def load_part_groups(path: Path) -> PartGroups:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return PartGroups.model_validate(data["part_groups"])
```

- [ ] **Step 5: Run tests, checks and commit**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: all pass.

```bash
git add config/taxonomy.toml src/claimlens/data/records.py src/claimlens/data/taxonomy.py tests/unit/test_part_groups.py
git commit -m "feat: add coarse part groups and optional annotation scores"
```

---

### Task 3: Auto-label post-processing

**Files:**
- Modify: `pyproject.toml` (groups `autolabel`, `review`; `opencv-python-headless` in `dev`; mypy overrides for `cv2`, `torch`, `transformers`, `fiftyone` with `follow_imports = "skip"`; coverage omit for the two adapters)
- Create: `src/claimlens/autolabel/__init__.py` (docstring only: `"""Verified auto-labelling with open-vocabulary foundation models."""`)
- Create: `src/claimlens/autolabel/postprocess.py`
- Test: `tests/unit/test_autolabel_postprocess.py`

**Interfaces:**
- Produces: `Detection(phrase, score, box)` (frozen dataclass, `box = (x1, y1, x2, y2)` pixels); `box_iou(a, b) -> float`; `select_detections(detections, prompts, *, min_score, iou_threshold=0.5, max_per_group=2) -> tuple[list[tuple[str, Detection]], Counter[str]]` (drop reasons `ambiguous_phrase`, `low_score`, `duplicate`); `mask_to_polygon(mask, *, epsilon_fraction=0.005) -> tuple[float, ...] | None` (normalised, largest contour).

- [ ] **Step 1: Update `pyproject.toml`**

Add to `dev`: `"opencv-python-headless>=4.9",`. Add groups:

```toml
autolabel = [
  "opencv-python-headless>=4.9",
  "torch>=2.4",
  "transformers>=5.0",
]
review = [
  "fiftyone>=1.0",
]
```

Add a second mypy override and extend coverage omit:

```toml
[[tool.mypy.overrides]]
module = ["cv2", "cv2.*", "torch", "torch.*", "transformers", "transformers.*", "fiftyone", "fiftyone.*"]
ignore_missing_imports = true
follow_imports = "skip"
```

```toml
[tool.coverage.run]
omit = [
  "*/claimlens/vision/legacy_yolo.py",
  "*/claimlens/autolabel/grounded_sam.py",
  "*/claimlens/review/fiftyone_app.py",
]
```

Run: `uv sync`
Expected: installs `opencv-python-headless`.

- [ ] **Step 2: Write the failing test** `tests/unit/test_autolabel_postprocess.py`

```python
import numpy as np
import pytest

from claimlens.autolabel.postprocess import Detection, box_iou, mask_to_polygon, select_detections

PROMPTS = {"car door": "door", "hood": "hood", "headlight": "light"}


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_merged_phrase_is_dropped() -> None:
    kept, dropped = select_detections(
        [Detection("headlight side mirror", 0.9, (0, 0, 10, 10))], PROMPTS, min_score=0.3
    )
    assert kept == []
    assert dropped == {"ambiguous_phrase": 1}


def test_low_score_and_duplicates_are_dropped() -> None:
    detections = [
        Detection("car door", 0.8, (0, 0, 100, 100)),
        Detection("car door", 0.7, (5, 5, 100, 100)),
        Detection("Car Door ", 0.6, (200, 0, 300, 100)),
        Detection("hood", 0.2, (0, 0, 50, 50)),
        Detection("hood", 0.5, (0, 0, 100, 100)),
    ]
    kept, dropped = select_detections(detections, PROMPTS, min_score=0.3, max_per_group=2)
    assert [(g, d.score) for g, d in kept] == [("door", 0.8), ("door", 0.6), ("hood", 0.5)]
    assert dropped == {"duplicate": 1, "low_score": 1}


def test_max_per_group_caps_detections() -> None:
    detections = [
        Detection("car door", 0.9 - i / 10, (i * 200, 0, i * 200 + 100, 100)) for i in range(3)
    ]
    kept, dropped = select_detections(detections, PROMPTS, min_score=0.3, max_per_group=2)
    assert len(kept) == 2
    assert dropped == {"duplicate": 1}


def test_rectangle_mask_becomes_four_corner_polygon() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 50:150] = True
    polygon = mask_to_polygon(mask)
    assert polygon is not None
    xs, ys = polygon[0::2], polygon[1::2]
    assert len(xs) == 4
    assert min(xs) == pytest.approx(0.25)
    assert max(xs) == pytest.approx(149 / 200)
    assert min(ys) == pytest.approx(0.2)
    assert max(ys) == pytest.approx(59 / 100)


def test_empty_mask_gives_no_polygon() -> None:
    assert mask_to_polygon(np.zeros((10, 10), dtype=bool)) is None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_autolabel_postprocess.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.autolabel'`.

- [ ] **Step 4: Write** `src/claimlens/autolabel/postprocess.py`

```python
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
```

- [ ] **Step 5: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_autolabel_postprocess.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 6 passed; no issues.

```bash
git add pyproject.toml uv.lock src/claimlens/autolabel tests/unit/test_autolabel_postprocess.py
git commit -m "feat: add auto-label post-processing for open-vocabulary detections"
```

---

### Task 4: Labeller interface and Grounding DINO + SAM 2 adapter

**Files:**
- Create: `src/claimlens/autolabel/labeller.py`
- Create: `src/claimlens/autolabel/grounded_sam.py`
- Test: `tests/integration/test_grounded_sam.py`

**Interfaces:**
- Consumes: Task 3, `Annotation` (Task 2).
- Produces: `PartLabeller` protocol (`model_version: str` property; `label(image_path: Path) -> tuple[list[Annotation], Counter[str]]`); `GroundedSamLabeller(prompts, *, box_threshold, text_threshold, min_score, max_per_group, detector, segmenter)`. Annotations carry the part group as `label` and the detection score as `score`; a mask that yields no polygon counts as `no_mask`.

- [ ] **Step 1: Write** `src/claimlens/autolabel/labeller.py`

```python
"""Interface for anything that proposes part-group masks for an image."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Protocol

from claimlens.data.records import Annotation


class PartLabeller(Protocol):
    @property
    def model_version(self) -> str: ...

    def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]: ...
```

- [ ] **Step 2: Write** `src/claimlens/autolabel/grounded_sam.py`

```python
"""Grounding DINO (open-vocabulary boxes) + SAM 2 (masks from boxes), on CPU.

Requires `uv sync --group autolabel`. Excluded from coverage; see the integration test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from claimlens.autolabel.postprocess import Detection, mask_to_polygon, select_detections
from claimlens.data.records import Annotation


class GroundedSamLabeller:
    def __init__(
        self,
        prompts: Mapping[str, str],
        *,
        box_threshold: float,
        text_threshold: float,
        min_score: float,
        max_per_group: int,
        detector: str,
        segmenter: str,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self._torch: Any = torch
        self._prompts = dict(prompts)
        self._text = " ".join(f"{phrase}." for phrase in self._prompts)
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._min_score = min_score
        self._max_per_group = max_per_group
        self._det_processor: Any = AutoProcessor.from_pretrained(detector)
        self._det_model: Any = AutoModelForZeroShotObjectDetection.from_pretrained(detector)
        self._seg_processor: Any = Sam2Processor.from_pretrained(segmenter)
        self._seg_model: Any = Sam2Model.from_pretrained(segmenter)
        self.model_version = f"{detector}+{segmenter}"

    def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]:
        torch = self._torch
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self._det_processor(images=image, text=self._text, return_tensors="pt")
        with torch.no_grad():
            outputs = self._det_model(**inputs)
        result = self._det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        detections = [
            Detection(phrase=str(label), score=float(score), box=tuple(float(v) for v in box))
            for box, score, label in zip(
                result["boxes"].tolist(),
                result["scores"].tolist(),
                result["text_labels"],
                strict=True,
            )
        ]
        kept, dropped = select_detections(
            detections, self._prompts, min_score=self._min_score, max_per_group=self._max_per_group
        )
        if not kept:
            return [], dropped
        seg_inputs = self._seg_processor(
            images=image, input_boxes=[[list(d.box) for _, d in kept]], return_tensors="pt"
        )
        with torch.no_grad():
            seg_outputs = self._seg_model(**seg_inputs, multimask_output=False)
        masks = self._seg_processor.post_process_masks(
            seg_outputs.pred_masks, seg_inputs["original_sizes"]
        )[0]
        annotations: list[Annotation] = []
        for (group, detection), mask in zip(kept, masks, strict=True):
            polygon = mask_to_polygon(mask[0].numpy())
            if polygon is None:
                dropped["no_mask"] += 1
                continue
            annotations.append(
                Annotation(label=group, polygon=polygon, score=round(detection.score, 4))
            )
        return annotations, dropped
```

- [ ] **Step 3: Write the opt-in integration test** `tests/integration/test_grounded_sam.py`

```python
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("transformers") is None or os.environ.get("CLAIMLENS_SLOW") != "1",
        reason="needs `uv sync --group autolabel` and CLAIMLENS_SLOW=1 (downloads ~800 MB)",
    ),
]


def test_grounded_sam_proposes_known_part_groups() -> None:
    from claimlens.autolabel.grounded_sam import GroundedSamLabeller

    labeller = GroundedSamLabeller(
        {"car door": "door", "wheel": "wheel"},
        box_threshold=0.3,
        text_threshold=0.25,
        min_score=0.3,
        max_per_group=2,
        detector="IDEA-Research/grounding-dino-tiny",
        segmenter="facebook/sam2.1-hiera-tiny",
    )
    annotations, _dropped = labeller.label(SAMPLE)
    assert all(a.label in {"door", "wheel"} for a in annotations)
    assert all(a.score is not None and len(a.polygon) >= 6 for a in annotations)
```

- [ ] **Step 4: Run checks and commit**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: all pass; the new integration test is skipped.

```bash
git add src/claimlens/autolabel/labeller.py src/claimlens/autolabel/grounded_sam.py tests/integration/test_grounded_sam.py
git commit -m "feat: add Grounding DINO + SAM 2 part labeller behind a labeller interface"
```

---

### Task 5: Auto-labelling job and `claimlens data autolabel`

**Files:**
- Create: `config/autolabel.toml`
- Create: `src/claimlens/autolabel/job.py`
- Modify: `src/claimlens/cli.py`
- Test: `tests/unit/test_autolabel_job.py`

**Interfaces:**
- Consumes: Tasks 2-4, `read_records`/`write_records` (M2a).
- Produces: `AutolabelJob(id, dataset, split, sample_size, seed, box_threshold, text_threshold, min_score, max_per_group, detector, segmenter, prompts)`; `load_autolabel_jobs(path) -> dict[str, AutolabelJob]`; `select_sample(records, splits, *, split, size, seed) -> list[ImageRecord]` (sorted by image id); `run_autolabel(job, *, repo_root, labeller, part_groups, progress=None) -> dict[str, Any]` (writes `data/interim/<job>/autolabels.jsonl` and `reports/data/<job>-autolabel.json`, returns that report). CLI: `claimlens data autolabel <job-id>`; `main(..., labeller_factory=...)`.

- [ ] **Step 1: Write** `config/autolabel.toml`

```toml
# Verified auto-labelling: part-group masks on frozen CarDD test photos (M2b).

[fusion-eval-v1]
dataset = "damage-v1"
split = "test"
sample_size = 100
seed = 20261001
box_threshold = 0.30
text_threshold = 0.25
min_score = 0.30
max_per_group = 2
detector = "IDEA-Research/grounding-dino-tiny"
segmenter = "facebook/sam2.1-hiera-tiny"

[fusion-eval-v1.prompts]
"front bumper" = "front_bumper"
"rear bumper" = "rear_bumper"
"car door" = "door"
"hood" = "hood"
"trunk" = "trunk"
"headlight" = "light"
"taillight" = "light"
"windshield" = "glass"
"side mirror" = "mirror"
"wheel" = "wheel"
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_autolabel_job.py`

```python
import json
from collections import Counter
from pathlib import Path

import pytest

from claimlens.autolabel.job import load_autolabel_jobs, run_autolabel, select_sample
from claimlens.cli import main
from claimlens.data.records import Annotation, ImageRecord, write_records
from claimlens.data.taxonomy import load_part_groups
from tests.data_helpers import make_pattern_image

ROOT = Path(__file__).resolve().parents[2]
DOOR = Annotation(label="door", polygon=(0.1, 0.1, 0.5, 0.1, 0.5, 0.5), score=0.6)


class FakeLabeller:
    model_version = "fake-labeller"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]:
        self.seen.append(image_path.name)
        return [DOOR], Counter({"ambiguous_phrase": 1})


def _record(name: str, split: str) -> ImageRecord:
    return ImageRecord(
        image_id=f"s:{name}",
        source="s",
        source_split=split,
        path=f"img/{name}.jpg",
        width=320,
        height=240,
    )


def _repo(tmp_path: Path) -> list[ImageRecord]:
    records = [_record(f"t{i}", "test") for i in range(5)] + [_record("a", "train")]
    for record in records:
        make_pattern_image(tmp_path / record.path, seed=len(record.image_id))
    interim = tmp_path / "data" / "interim" / "damage-v1"
    write_records(interim / "records.jsonl", records)
    splits = {r.image_id: r.source_split for r in records}
    (interim / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return records


def test_repo_job_config_loads_and_maps_to_part_groups() -> None:
    job = load_autolabel_jobs(ROOT / "config" / "autolabel.toml")["fusion-eval-v1"]
    groups = load_part_groups(ROOT / "config" / "taxonomy.toml")
    assert job.sample_size == 100
    assert set(job.prompts.values()) <= set(groups.classes)


def test_sample_is_deterministic_and_respects_split() -> None:
    records = [_record(f"t{i}", "test") for i in range(10)] + [_record("a", "train")]
    splits = {r.image_id: r.source_split for r in records}
    first = select_sample(records, splits, split="test", size=4, seed=1)
    again = select_sample(records, splits, split="test", size=4, seed=1)
    assert first == again
    assert len(first) == 4
    assert all(r.source_split == "test" for r in first)
    assert [r.image_id for r in first] == sorted(r.image_id for r in first)


def test_run_autolabel_writes_proposals_and_report(tmp_path: Path) -> None:
    _repo(tmp_path)
    job = load_autolabel_jobs(ROOT / "config" / "autolabel.toml")["fusion-eval-v1"]
    job = job.model_copy(update={"sample_size": 3})
    labeller = FakeLabeller()

    report = run_autolabel(
        job,
        repo_root=tmp_path,
        labeller=labeller,
        part_groups=load_part_groups(ROOT / "config" / "taxonomy.toml"),
    )

    assert len(labeller.seen) == 3
    lines = (
        (tmp_path / "data" / "interim" / "fusion-eval-v1" / "autolabels.jsonl")
        .read_text()
        .splitlines()
    )
    assert len(lines) == 3
    assert report["model_version"] == "fake-labeller"
    assert report["proposals"] == {"door": 3}
    assert report["dropped"] == {"ambiguous_phrase": 3}
    assert (tmp_path / "reports" / "data" / "fusion-eval-v1-autolabel.json").is_file()


def test_labeller_output_outside_part_groups_is_rejected(tmp_path: Path) -> None:
    _repo(tmp_path)
    job = load_autolabel_jobs(ROOT / "config" / "autolabel.toml")["fusion-eval-v1"]

    class BadLabeller(FakeLabeller):
        def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]:
            return [DOOR.model_copy(update={"label": "fender"})], Counter()

    with pytest.raises(ValueError, match="fender"):
        run_autolabel(
            job,
            repo_root=tmp_path,
            labeller=BadLabeller(),
            part_groups=load_part_groups(ROOT / "config" / "taxonomy.toml"),
        )


def test_autolabel_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = main(
        ["--config", str(ROOT / "config"), "data", "autolabel", "fusion-eval-v1"],
        labeller_factory=lambda _job: FakeLabeller(),
    )
    assert code == 0
    assert "Auto-labelled 5 images" in capsys.readouterr().out


def test_autolabel_unknown_job(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(ROOT / "config"), "data", "autolabel", "nope"]) == 2
    assert "unknown auto-label job 'nope'" in capsys.readouterr().err
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_autolabel_job.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.autolabel.job'`.

- [ ] **Step 4: Write** `src/claimlens/autolabel/job.py`

```python
"""Run a verified auto-labelling job over a deterministic sample of a built dataset."""

from __future__ import annotations

import json
import random
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from claimlens.autolabel.labeller import PartLabeller
from claimlens.data.records import ImageRecord, read_records, write_records
from claimlens.data.taxonomy import PartGroups
from claimlens.domain import Frozen


class AutolabelJob(Frozen):
    id: str
    dataset: str
    split: str
    sample_size: int
    seed: int
    box_threshold: float
    text_threshold: float
    min_score: float
    max_per_group: int
    detector: str
    segmenter: str
    prompts: dict[str, str]


def load_autolabel_jobs(path: Path) -> dict[str, AutolabelJob]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {key: AutolabelJob.model_validate({"id": key, **body}) for key, body in data.items()}


def select_sample(
    records: Sequence[ImageRecord],
    splits: Mapping[str, str],
    *,
    split: str,
    size: int,
    seed: int,
) -> list[ImageRecord]:
    candidates = sorted(
        (r for r in records if splits.get(r.image_id) == split), key=lambda r: r.image_id
    )
    chosen = random.Random(seed).sample(candidates, min(size, len(candidates)))
    return sorted(chosen, key=lambda r: r.image_id)


def run_autolabel(
    job: AutolabelJob,
    *,
    repo_root: Path,
    labeller: PartLabeller,
    part_groups: PartGroups,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    interim = repo_root / "data" / "interim" / job.dataset
    records = read_records(interim / "records.jsonl")
    splits: dict[str, str] = json.loads((interim / "splits.json").read_text(encoding="utf-8"))
    sample = select_sample(records, splits, split=job.split, size=job.sample_size, seed=job.seed)

    proposals: list[ImageRecord] = []
    counts: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    for number, record in enumerate(sample, start=1):
        annotations, reasons = labeller.label(repo_root / record.path)
        for annotation in annotations:
            if annotation.label not in part_groups.classes:
                raise ValueError(f"labeller produced unknown part group {annotation.label!r}")
            counts[annotation.label] += 1
        dropped.update(reasons)
        proposals.append(record.model_copy(update={"annotations": tuple(annotations)}))
        if progress:
            progress(number, len(sample))

    write_records(repo_root / "data" / "interim" / job.id / "autolabels.jsonl", proposals)
    report: dict[str, Any] = {
        "job": job.id,
        "model_version": labeller.model_version,
        "images": len(proposals),
        "images_with_proposals": sum(1 for p in proposals if p.annotations),
        "proposals": dict(sorted(counts.items())),
        "dropped": dict(sorted(dropped.items())),
    }
    report_path = repo_root / "reports" / "data" / f"{job.id}-autolabel.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return report
```

- [ ] **Step 5: Add `data autolabel` to** `src/claimlens/cli.py`

Imports (merge): `from claimlens.autolabel.job import AutolabelJob, load_autolabel_jobs, run_autolabel`, `from claimlens.autolabel.labeller import PartLabeller`, `from claimlens.data.taxonomy import load_part_groups`.

Below `DetectorFactory`, add:

```python
LabellerFactory = Callable[[AutolabelJob], PartLabeller]


def _grounded_sam(job: AutolabelJob) -> PartLabeller:
    from claimlens.autolabel.grounded_sam import GroundedSamLabeller

    return GroundedSamLabeller(
        job.prompts,
        box_threshold=job.box_threshold,
        text_threshold=job.text_threshold,
        min_score=job.min_score,
        max_per_group=job.max_per_group,
        detector=job.detector,
        segmenter=job.segmenter,
    )
```

In `build_parser()`, in the `data` subparsers, add:

```python
    autolabel = data_sub.add_parser("autolabel", help="propose part masks with foundation models")
    autolabel.add_argument("job_id")
```

Change the `main` signature to accept `labeller_factory: LabellerFactory = _grounded_sam` (keyword-only, after `detector_factory`) and the data dispatch to `return _data(args, labeller_factory)`. Change `def _data(args: argparse.Namespace) -> int:` to `def _data(args: argparse.Namespace, labeller_factory: LabellerFactory) -> int:` and add, right after `repo_root = Path.cwd()`:

```python
    if args.data_command == "autolabel":
        return _autolabel(args, repo_root, labeller_factory)
```

Append:

```python
def _autolabel(args: argparse.Namespace, repo_root: Path, factory: LabellerFactory) -> int:
    jobs = load_autolabel_jobs(args.config / "autolabel.toml")
    if args.job_id not in jobs:
        known = ", ".join(sorted(jobs))
        print(f"error: unknown auto-label job {args.job_id!r} (known: {known})", file=sys.stderr)
        return 2
    job = jobs[args.job_id]

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} images", flush=True)

    report = run_autolabel(
        job,
        repo_root=repo_root,
        labeller=factory(job),
        part_groups=load_part_groups(args.config / "taxonomy.toml"),
        progress=progress,
    )
    print(f"Auto-labelled {report['images']} images: proposals {report['proposals']}")
    print(f"Dropped: {report['dropped']}")
    return 0
```

- [ ] **Step 6: Run tests, checks and commit**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: all pass.

```bash
git add config/autolabel.toml src/claimlens/autolabel/job.py src/claimlens/cli.py tests/unit/test_autolabel_job.py
git commit -m "feat: add auto-labelling job and claimlens data autolabel"
```

---

### Task 6: Review decisions

**Files:**
- Create: `src/claimlens/review/__init__.py` (docstring only: `"""Human review: decisions as versioned data, applied by pure functions."""`)
- Create: `src/claimlens/review/decisions.py`
- Test: `tests/unit/test_review_decisions.py`

**Interfaces:**
- Consumes: `ImageRecord` (M2a), `GoldenClaim` (M1), `Route` (M1).
- Produces: `annotation_key(image_id, index) -> str` (`"<image_id>#<index>"`); `PartReview(job_id, reviewer, decisions: dict[str, Literal["approved","rejected"]])`; `apply_part_review(records, review) -> tuple[list[ImageRecord], dict[str, int]]` (counts `approved`, `rejected`, `unreviewed_images`, `images_all_rejected`, `images_without_proposals`); `GoldenReview(reviewer, decisions: dict[str, str])` (value `agree` or a `Route` value); `apply_golden_review(cases, review) -> list[GoldenClaim]`; `read_review[T](path, model) -> T`; `write_review(path, review)`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_review_decisions.py`

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.records import Annotation, ImageRecord
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim
from claimlens.review.decisions import (
    GoldenReview,
    PartReview,
    annotation_key,
    apply_golden_review,
    apply_part_review,
    read_review,
    write_review,
)

SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5)


def _record(image_id: str, *labels: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source="s",
        source_split="test",
        path="x.jpg",
        width=1,
        height=1,
        annotations=tuple(Annotation(label=label, polygon=SQUARE, score=0.5) for label in labels),
    )


def _review(**decisions: str) -> PartReview:
    return PartReview.model_validate({"job_id": "j", "reviewer": "me", "decisions": decisions})


def test_only_approved_annotations_are_kept() -> None:
    records = [_record("a", "door", "hood")]
    review = _review(**{annotation_key("a", 0): "approved", annotation_key("a", 1): "rejected"})
    kept, counts = apply_part_review(records, review)
    assert [a.label for a in kept[0].annotations] == ["door"]
    assert counts == {"approved": 1, "rejected": 1}


def test_unreviewed_image_is_left_out() -> None:
    records = [_record("a", "door", "hood")]
    kept, counts = apply_part_review(records, _review(**{annotation_key("a", 0): "approved"}))
    assert kept == []
    assert counts == {"unreviewed_images": 1}


def test_all_rejected_and_empty_images_are_counted() -> None:
    records = [_record("a", "door"), _record("b")]
    kept, counts = apply_part_review(records, _review(**{annotation_key("a", 0): "rejected"}))
    assert kept == []
    assert counts == {"rejected": 1, "images_all_rejected": 1, "images_without_proposals": 1}


def test_invalid_part_decision_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _review(**{"a#0": "maybe"})


def _case(case_id: str, route: Route = Route.FAST_TRACK) -> GoldenClaim:
    return GoldenClaim(
        case_id=case_id,
        scenario="s",
        policy_id="P-1001",
        description="",
        photos=("x.jpg",),
        expected_route=route,
        label_source="oracle",
    )


def test_golden_review_marks_and_corrects_cases() -> None:
    review = GoldenReview(reviewer="me", decisions={"g1": "agree", "g2": "ADJUSTER_REVIEW"})
    g1, g2, g3 = apply_golden_review([_case("g1"), _case("g2"), _case("g3")], review)
    assert g1.reviewed and g1.expected_route is Route.FAST_TRACK
    assert g2.reviewed and g2.expected_route is Route.ADJUSTER_REVIEW
    assert "FAST_TRACK to ADJUSTER_REVIEW" in g2.notes
    assert not g3.reviewed


def test_unknown_golden_decision_raises() -> None:
    review = GoldenReview(reviewer="me", decisions={"g1": "DENY"})
    with pytest.raises(ValueError, match="DENY"):
        apply_golden_review([_case("g1")], review)


def test_reviews_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "reviews" / "j.json"
    review = _review(**{"a#0": "approved"})
    write_review(path, review)
    assert read_review(path, PartReview) == review
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_review_decisions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.review'`.

- [ ] **Step 3: Write** `src/claimlens/review/decisions.py`

```python
"""Review decisions are data: stored as JSON in git and applied by pure functions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from claimlens.data.records import ImageRecord
from claimlens.domain import Frozen, Route
from claimlens.evals.golden import GoldenClaim

PartDecision = Literal["approved", "rejected"]


def annotation_key(image_id: str, index: int) -> str:
    return f"{image_id}#{index}"


class PartReview(Frozen):
    job_id: str
    reviewer: str
    decisions: dict[str, PartDecision]


class GoldenReview(Frozen):
    reviewer: str
    decisions: dict[str, str]


def apply_part_review(
    records: Sequence[ImageRecord], review: PartReview
) -> tuple[list[ImageRecord], dict[str, int]]:
    """Keep approved annotations; an image is used only if every proposal on it was decided."""
    kept: list[ImageRecord] = []
    counts: Counter[str] = Counter()
    for record in records:
        keys = [annotation_key(record.image_id, i) for i in range(len(record.annotations))]
        if not keys:
            counts["images_without_proposals"] += 1
            continue
        if any(key not in review.decisions for key in keys):
            counts["unreviewed_images"] += 1
            continue
        approved = tuple(
            annotation
            for key, annotation in zip(keys, record.annotations, strict=True)
            if review.decisions[key] == "approved"
        )
        counts["approved"] += len(approved)
        counts["rejected"] += len(keys) - len(approved)
        if approved:
            kept.append(record.model_copy(update={"annotations": approved}))
        else:
            counts["images_all_rejected"] += 1
    return kept, {key: value for key, value in counts.items() if value}


def apply_golden_review(cases: Sequence[GoldenClaim], review: GoldenReview) -> list[GoldenClaim]:
    updated: list[GoldenClaim] = []
    for case in cases:
        decision = review.decisions.get(case.case_id)
        if decision is None:
            updated.append(case)
        elif decision == "agree":
            updated.append(case.model_copy(update={"reviewed": True}))
        else:
            try:
                route = Route(decision)
            except ValueError:
                raise ValueError(
                    f"{case.case_id}: decision {decision!r} is neither 'agree' nor a route"
                ) from None
            note = f"Reviewer changed expected route from {case.expected_route.value} to {route.value}."
            updated.append(
                case.model_copy(update={"reviewed": True, "expected_route": route, "notes": note})
            )
    return updated


def read_review[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_review(path: Path, review: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(review.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
```

- [ ] **Step 4: Run tests, checks and commit**

Run: `uv run pytest tests/unit/test_review_decisions.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: 7 passed; no issues.

```bash
git add src/claimlens/review tests/unit/test_review_decisions.py
git commit -m "feat: add review decisions as versioned data with pure apply functions"
```

---

### Task 7: FiftyOne review UI and `claimlens review` commands

**Files:**
- Create: `src/claimlens/review/fiftyone_app.py`
- Modify: `src/claimlens/cli.py`
- Test: `tests/unit/test_review_cli.py`

**Interfaces:**
- Consumes: Task 6, `read_records`/`write_records`, `load_golden`/`write_golden`.
- Produces (`fiftyone_app.py`): `ReviewUnavailableError(RuntimeError)`; `launch_parts_review(records, *, repo_root, name, damage)`; `export_parts_review(name, *, job_id, reviewer) -> PartReview`; `launch_golden_review(cases, *, repo_root, name)`; `export_golden_review(name, *, reviewer) -> GoldenReview`. Reviewers tag each part polyline `approved` or `rejected`; each golden sample `agree` or `route:<ROUTE>`.
- Produces (CLI): `claimlens review {launch,export,apply} {parts,golden} [--job ID] [--golden PATH] [--reviewer NAME]`. Files: `reviews/<job>.json`, `reviews/golden-<golden folder name>.json`; `apply parts` writes `data/processed/<job>/parts.jsonl` and `reports/data/<job>-review.json`; `apply golden` rewrites the golden file in place.

- [ ] **Step 1: Write the failing test** `tests/unit/test_review_cli.py`

```python
import importlib.util
import json
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.data.records import Annotation, ImageRecord, read_records, write_records
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, load_golden, write_golden
from claimlens.review.decisions import GoldenReview, PartReview, write_review

ROOT = Path(__file__).resolve().parents[2]
SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5)


def test_apply_parts_writes_the_eval_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = ImageRecord(
        image_id="s:a",
        source="s",
        source_split="test",
        path="a.jpg",
        width=1,
        height=1,
        annotations=(
            Annotation(label="door", polygon=SQUARE, score=0.7),
            Annotation(label="hood", polygon=SQUARE, score=0.4),
        ),
    )
    write_records(tmp_path / "data" / "interim" / "fusion-eval-v1" / "autolabels.jsonl", [record])
    write_review(
        tmp_path / "reviews" / "fusion-eval-v1.json",
        PartReview(
            job_id="fusion-eval-v1",
            reviewer="me",
            decisions={"s:a#0": "approved", "s:a#1": "rejected"},
        ),
    )
    monkeypatch.chdir(tmp_path)

    assert main(["--config", str(ROOT / "config"), "review", "apply", "parts"]) == 0

    kept = read_records(tmp_path / "data" / "processed" / "fusion-eval-v1" / "parts.jsonl")
    assert [a.label for a in kept[0].annotations] == ["door"]
    report = json.loads((tmp_path / "reports" / "data" / "fusion-eval-v1-review.json").read_text())
    assert report["counts"] == {"approved": 1, "rejected": 1}
    assert report["images"] == 1
    assert "1 approved" in capsys.readouterr().out


def test_apply_golden_marks_reviewed_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    golden = tmp_path / "evals" / "golden" / "v1" / "claims.jsonl"
    case = GoldenClaim(
        case_id="g1",
        scenario="s",
        policy_id="P-1001",
        description="",
        photos=("x.jpg",),
        expected_route=Route.FAST_TRACK,
        label_source="oracle",
    )
    write_golden(golden, [case])
    write_review(
        tmp_path / "reviews" / "golden-v1.json",
        GoldenReview(reviewer="me", decisions={"g1": "agree"}),
    )
    monkeypatch.chdir(tmp_path)

    assert main(["review", "apply", "golden", "--golden", str(golden)]) == 0
    assert load_golden(golden)[0].reviewed


@pytest.mark.skipif(
    importlib.util.find_spec("fiftyone") is not None, reason="FiftyOne is installed"
)
def test_review_launch_without_fiftyone(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["review", "launch", "golden", "--golden", str(ROOT / "evals/golden/v0/claims.jsonl")])
        == 2
    )
    assert "uv sync --group review" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_review_cli.py -q`
Expected: FAIL (argparse error: invalid choice `review`).

- [ ] **Step 3: Write** `src/claimlens/review/fiftyone_app.py`

```python
"""FiftyOne as the review UI. Requires `uv sync --group review`. Excluded from coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from claimlens.data.records import Annotation, ImageRecord
from claimlens.evals.golden import GoldenClaim
from claimlens.review.decisions import GoldenReview, PartReview, annotation_key


class ReviewUnavailableError(RuntimeError):
    pass


def _fo() -> Any:
    try:
        import fiftyone as fo
    except ImportError:
        raise ReviewUnavailableError(
            "FiftyOne is not installed: run `uv sync --group review`"
        ) from None
    return fo


def _polyline(fo: Any, annotation: Annotation, **attributes: object) -> Any:
    points = list(zip(annotation.polygon[0::2], annotation.polygon[1::2], strict=True))
    return fo.Polyline(
        label=annotation.label, points=[points], closed=True, filled=True, **attributes
    )


def launch_parts_review(
    records: Sequence[ImageRecord],
    *,
    repo_root: Path,
    name: str,
    damage: Mapping[str, ImageRecord],
) -> None:
    fo = _fo()
    dataset = fo.Dataset(name, overwrite=True, persistent=True)
    samples = []
    for record in records:
        sample = fo.Sample(
            filepath=str((repo_root / record.path).resolve()), image_id=record.image_id
        )
        sample["parts"] = fo.Polylines(
            polylines=[
                _polyline(fo, a, key=annotation_key(record.image_id, i), confidence=a.score)
                for i, a in enumerate(record.annotations)
            ]
        )
        context = damage.get(record.image_id)
        if context is not None:
            sample["damage"] = fo.Polylines(
                polylines=[_polyline(fo, a) for a in context.annotations]
            )
        samples.append(sample)
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    session.wait()


def export_parts_review(name: str, *, job_id: str, reviewer: str) -> PartReview:
    fo = _fo()
    decisions: dict[str, str] = {}
    for sample in fo.load_dataset(name):
        for polyline in sample["parts"].polylines:
            if "approved" in polyline.tags:
                decisions[polyline.key] = "approved"
            elif "rejected" in polyline.tags:
                decisions[polyline.key] = "rejected"
    return PartReview.model_validate(
        {"job_id": job_id, "reviewer": reviewer, "decisions": decisions}
    )


def launch_golden_review(cases: Sequence[GoldenClaim], *, repo_root: Path, name: str) -> None:
    fo = _fo()
    dataset = fo.Dataset(name, overwrite=True, persistent=True)
    samples = [
        fo.Sample(
            filepath=str((repo_root / case.photos[0]).resolve()),
            case_id=case.case_id,
            scenario=case.scenario,
            policy_id=case.policy_id,
            expected_route=case.expected_route.value,
        )
        for case in cases
    ]
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    session.wait()


def export_golden_review(name: str, *, reviewer: str) -> GoldenReview:
    fo = _fo()
    decisions: dict[str, str] = {}
    for sample in fo.load_dataset(name):
        routes = [tag.removeprefix("route:") for tag in sample.tags if tag.startswith("route:")]
        if routes:
            decisions[sample["case_id"]] = routes[0]
        elif "agree" in sample.tags:
            decisions[sample["case_id"]] = "agree"
    return GoldenReview(reviewer=reviewer, decisions=decisions)
```

- [ ] **Step 4: Add `review` commands to** `src/claimlens/cli.py`

Imports (merge): `from claimlens.data.records import read_records, write_records`, `from claimlens.evals.golden import write_golden`, `from claimlens.review.decisions import GoldenReview, PartReview, apply_golden_review, apply_part_review, read_review, write_review`.

In `build_parser()`, before `return parser`, add:

```python
    review = sub.add_parser("review", help="human review in FiftyOne")
    review.add_argument("action", choices=["launch", "export", "apply"])
    review.add_argument("target", choices=["parts", "golden"])
    review.add_argument("--job", default="fusion-eval-v1", help="auto-label job id")
    review.add_argument("--golden", type=Path, default=Path("evals/golden/v1/claims.jsonl"))
    review.add_argument("--reviewer", default="reviewer")
```

In `main()`, after the `data` dispatch, add:

```python
    if args.command == "review":
        return _review(args)
```

Append:

```python
def _review(args: argparse.Namespace) -> int:
    from claimlens.review import fiftyone_app

    repo_root = Path.cwd()
    parts_file = repo_root / "data" / "interim" / args.job / "autolabels.jsonl"
    parts_review = repo_root / "reviews" / f"{args.job}.json"
    golden_review = repo_root / "reviews" / f"golden-{args.golden.parent.name}.json"
    try:
        if args.action == "launch" and args.target == "parts":
            damage = {
                r.image_id: r
                for r in read_records(
                    repo_root / "data" / "interim" / "damage-v1" / "records.jsonl"
                )
            }
            fiftyone_app.launch_parts_review(
                read_records(parts_file),
                repo_root=repo_root,
                name=f"claimlens-{args.job}",
                damage=damage,
            )
        elif args.action == "launch":
            fiftyone_app.launch_golden_review(
                load_golden(args.golden),
                repo_root=repo_root,
                name=f"claimlens-golden-{args.golden.parent.name}",
            )
        elif args.action == "export" and args.target == "parts":
            review = fiftyone_app.export_parts_review(
                f"claimlens-{args.job}", job_id=args.job, reviewer=args.reviewer
            )
            write_review(parts_review, review)
            print(f"Wrote {len(review.decisions)} decisions to {parts_review}")
        elif args.action == "export":
            golden = fiftyone_app.export_golden_review(
                f"claimlens-golden-{args.golden.parent.name}", reviewer=args.reviewer
            )
            write_review(golden_review, golden)
            print(f"Wrote {len(golden.decisions)} decisions to {golden_review}")
        elif args.target == "parts":
            kept, counts = apply_part_review(
                read_records(parts_file), read_review(parts_review, PartReview)
            )
            out = repo_root / "data" / "processed" / args.job / "parts.jsonl"
            write_records(out, kept)
            report = {"job": args.job, "images": len(kept), "counts": counts}
            report_path = repo_root / "reports" / "data" / f"{args.job}-review.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
            )
            print(
                f"{counts.get('approved', 0)} approved, {counts.get('rejected', 0)} rejected; {len(kept)} images in {out}"
            )
        else:
            cases = apply_golden_review(
                load_golden(args.golden), read_review(golden_review, GoldenReview)
            )
            write_golden(args.golden, cases)
            print(f"{sum(c.reviewed for c in cases)} of {len(cases)} golden cases reviewed")
    except fiftyone_app.ReviewUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
```

Add `import json` to the stdlib imports if absent. Let `ruff format` wrap long lines.

- [ ] **Step 5: Run tests, checks and commit**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: all pass.

```bash
git add src/claimlens/review/fiftyone_app.py src/claimlens/cli.py tests/unit/test_review_cli.py
git commit -m "feat: add FiftyOne review UI and claimlens review commands"
```

---

### Task 8: Golden claims v1

**Files:**
- Modify: `src/claimlens/evals/oracle.py` (`findings_from_record`)
- Modify: `tests/unit/test_oracle.py`
- Create: `scripts/build_golden_v1.py`
- Create (generated): `evals/golden/v1/claims.jsonl`, `evals/golden/v1/README.md`
- Modify: `config/datasets.toml` (`protect_golden` → v1), `dvc.yaml` (dependency → v1 claims)

**Interfaces:**
- Produces: `findings_from_record(record: ImageRecord, photo_id: str = "p1") -> list[DamageFinding]` (pixel box from polygon extents, confidence 1.0, area fraction from the box).

- [ ] **Step 1: Add the failing test** to `tests/unit/test_oracle.py`

```python
def test_findings_from_record_uses_polygon_extents() -> None:
    from claimlens.data.records import Annotation, ImageRecord
    from claimlens.evals.oracle import findings_from_record

    record = ImageRecord(
        image_id="s:a",
        source="s",
        source_split="test",
        path="a.jpg",
        width=1000,
        height=500,
        annotations=(Annotation(label="scratch", polygon=(0.1, 0.2, 0.3, 0.2, 0.3, 0.6)),),
    )
    (finding,) = findings_from_record(record)
    assert finding.damage_type is DamageType.SCRATCH
    assert finding.bbox == BoundingBox(x1=100, y1=100, x2=300, y2=300)
    assert finding.image_area_fraction == pytest.approx(0.08)
```

Run: `uv run pytest tests/unit/test_oracle.py -q`
Expected: 1 failed (`ImportError`).

- [ ] **Step 2: Add to** `src/claimlens/evals/oracle.py` (import `ImageRecord` from `claimlens.data.records`):

```python
def findings_from_record(record: ImageRecord, photo_id: str = "p1") -> list[DamageFinding]:
    findings: list[DamageFinding] = []
    for annotation in record.annotations:
        xs, ys = annotation.polygon[0::2], annotation.polygon[1::2]
        x1, x2 = max(min(xs), 0.0), min(max(xs), 1.0)
        y1, y2 = max(min(ys), 0.0), min(max(ys), 1.0)
        findings.append(
            DamageFinding(
                photo_id=photo_id,
                damage_type=normalize_class_name(annotation.label),
                confidence=1.0,
                bbox=BoundingBox(
                    x1=x1 * record.width,
                    y1=y1 * record.height,
                    x2=x2 * record.width,
                    y2=y2 * record.height,
                ),
                image_area_fraction=(x2 - x1) * (y2 - y1),
            )
        )
    return findings
```

Run: `uv run pytest tests/unit/test_oracle.py -q`
Expected: all pass.

- [ ] **Step 3: Write** `scripts/build_golden_v1.py`

```python
"""Build golden claims v1: v0 without near-duplicate photos, plus 50 frozen CarDD test claims.

Run from the repository root after `dvc repro`:  uv run python scripts/build_golden_v1.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from claimlens.data.dedupe import find_clusters, hash_file
from claimlens.data.records import read_records
from claimlens.decision import load_decision_config
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, load_golden, write_golden
from claimlens.evals.oracle import findings_from_record, oracle_route
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card

V0 = Path("evals/golden/v0/claims.jsonl")
OUT = Path("evals/golden/v1/claims.jsonl")
INTERIM = Path("data/interim/damage-v1")
ACTIVE_POLICIES = ["P-1001", "P-1002", "P-1003", "P-1004", "P-1005", "P-1006"]
DESCRIPTION = "Damage reported after a low-speed collision."
SEED = 20261001
FAST_TRACK_CASES = 20
OTHER_CASES = 30


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def dedupe_v0(cases: list[GoldenClaim]) -> tuple[list[GoldenClaim], list[str]]:
    """Drop oracle cases whose first photo repeats an earlier one in the same scenario.

    Scenario cases are left alone: their photos are deliberate test inputs (for example two
    blank "unusable" images that share a perceptual hash but test different failures).
    """
    hashes = [
        hash_file(c.case_id, Path(c.photos[0]))
        for c in cases
        if c.label_source == "oracle" and _readable(Path(c.photos[0]))
    ]
    clusters = find_clusters(hashes, max_distance=6)
    seen: set[tuple[str, int]] = set()
    kept: list[GoldenClaim] = []
    dropped: list[str] = []
    for case in cases:
        cluster = clusters.get(case.case_id)
        key = (case.scenario, cluster) if cluster is not None else None
        if key is not None and key in seen:
            dropped.append(case.case_id)
            continue
        if key is not None:
            seen.add(key)
        kept.append(case)
    return kept, dropped


def main() -> int:
    v0, dropped = dedupe_v0(load_golden(V0))
    records = read_records(INTERIM / "records.jsonl")
    splits: dict[str, str] = json.loads((INTERIM / "splits.json").read_text(encoding="utf-8"))
    test = sorted((r for r in records if splits[r.image_id] == "test"), key=lambda r: r.image_id)
    random.Random(SEED).shuffle(test)

    policies = load_policies(Path("config/policies.toml"))
    card = load_rate_card(Path("config/rate_card.toml"))
    config = load_decision_config(Path("config/decision_policy.toml"))
    fast: list[GoldenClaim] = []
    other: list[GoldenClaim] = []
    for i, record in enumerate(test):
        if len(fast) >= FAST_TRACK_CASES and len(other) >= OTHER_CASES:
            break
        policy = ACTIVE_POLICIES[i % len(ACTIVE_POLICIES)]
        route = oracle_route(
            findings_from_record(record), policies.get_coverage(policy), card, config
        )
        bucket = fast if route is Route.FAST_TRACK else other
        limit = FAST_TRACK_CASES if route is Route.FAST_TRACK else OTHER_CASES
        if len(bucket) < limit:
            bucket.append(
                GoldenClaim(
                    case_id="pending",
                    scenario="cardd_test_oracle",
                    policy_id=policy,
                    description=DESCRIPTION,
                    photos=(record.path,),
                    expected_route=route,
                    label_source="oracle",
                )
            )
    new = sorted(fast + other, key=lambda c: c.photos[0])
    numbered = [c.model_copy(update={"case_id": f"g{101 + i:03d}"}) for i, c in enumerate(new)]

    write_golden(OUT, v0 + numbered)
    print(f"Kept {len(v0)} v0 cases (dropped duplicates: {', '.join(dropped) or 'none'})")
    print(f"Added {len(fast)} fast-track and {len(other)} other CarDD test cases -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate golden v1 and switch protection to it**

Run: `uv run python scripts/build_golden_v1.py`
Expected: prints the dropped duplicate case ids (the `crashed454` copies) and `Added 20 fast-track and 30 other ...`.

In `config/datasets.toml` set `protect_golden = "evals/golden/v1/claims.jsonl"`; in `dvc.yaml` replace the dependency `evals/golden/v0/claims.jsonl` with `evals/golden/v1/claims.jsonl`.

Run: `uv run dvc repro build-damage-v1 && git diff --stat reports/data/damage-v1-stats.json`
Expected: the rebuild completes; image counts per split are unchanged (2,812 / 814 / 374) because protection now never alters the test split.

- [ ] **Step 5: Write** `evals/golden/v1/README.md`

```markdown
# Golden claims v1

Built by `scripts/build_golden_v1.py`:

- **v0 cases** (ids `g001`-`g050`) with near-duplicate photos removed: when two oracle-labelled
  cases in the same scenario use near-identical first photos (perceptual hash distance at most 6), only the first is
  kept. This removes repeated Roboflow augmentations such as the `crashed454` copies.
- **50 new cases** (ids `g101`-`g150`) from the frozen CarDD test split of `damage-v1`, which is
  never used for training: 20 whose oracle route is FAST_TRACK and 30 others, chosen by a seeded
  shuffle. Expected routes come from the decision policy applied to ground-truth masks
  (`label_source: oracle`).

`reviewed: true` marks cases a person checked in FiftyOne (`claimlens review launch golden`).
Reviewer corrections are recorded in each case's `notes`.
```

- [ ] **Step 6: Run checks and commit**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: all pass.

```bash
git add src/claimlens/evals/oracle.py tests/unit/test_oracle.py scripts/build_golden_v1.py evals/golden/v1 config/datasets.toml dvc.yaml dvc.lock reports/data
git commit -m "feat: add golden claims v1 with duplicates removed and CarDD test cases"
```

---

### Task 9: Run the auto-labelling job under DVC

**Files:**
- Modify: `dvc.yaml` (stage `autolabel-fusion-eval-v1`)
- Create (generated): `reports/data/fusion-eval-v1-autolabel.json`, `dvc.lock`

- [ ] **Step 1: Add the stage to** `dvc.yaml`

```yaml
  autolabel-fusion-eval-v1:
    cmd: uv run --group autolabel claimlens data autolabel fusion-eval-v1
    deps:
      - data/interim/damage-v1
      - config/autolabel.toml
      - config/taxonomy.toml
      - src/claimlens/autolabel
    outs:
      - data/interim/fusion-eval-v1
    metrics:
      - reports/data/fusion-eval-v1-autolabel.json:
          cache: false
```

- [ ] **Step 2: Run it** (about 35 minutes on CPU; run in the background)

Run: `HF_HUB_DISABLE_SYMLINKS_WARNING=1 uv run dvc repro autolabel-fusion-eval-v1`
Expected: progress every 10 images; ends with `Auto-labelled 100 images: proposals {...}` and the drop counts.

- [ ] **Step 3: Push and commit**

```bash
uv run dvc push
git add dvc.yaml dvc.lock reports/data/fusion-eval-v1-autolabel.json
git commit -m "feat: propose part-group masks on 100 frozen CarDD test photos"
```

---

### Task 10: Human review, evaluation set, baseline v1 and documentation

**This task needs the human reviewer.** Steps 1-2 are theirs.

**Files:**
- Create: `reviews/fusion-eval-v1.json`, `reviews/golden-v1.json` (exported decisions)
- Modify: `evals/golden/v1/claims.jsonl` (reviewed flags)
- Modify: `dvc.yaml` (stage `build-fusion-eval-v1`)
- Create: `evals/reports/<date>-triage-baseline-v1.md`, `docs/adr/0006-verified-auto-labelling.md`, `docs/retros/m2-data-engine.md`
- Modify: `docs/data-card.md`, `docs/roadmap.md`

- [ ] **Step 1 (reviewer): Review part proposals**

Run: `uv sync --group review` then `uv run claimlens review launch parts`. In the FiftyOne App, for each image select each `parts` polyline and add the label tag `approved` or `rejected` (damage polylines are shown for context). Close the App when done.

- [ ] **Step 2 (reviewer): Review golden claims**

Run: `uv run claimlens review launch golden`. For each sample add the sample tag `agree`, or `route:FAST_TRACK` / `route:ADJUSTER_REVIEW` / `route:FRAUD_REVIEW` to correct it.

- [ ] **Step 3: Export and apply the decisions**

```bash
uv run claimlens review export parts --reviewer <name>
uv run claimlens review export golden --reviewer <name>
uv run claimlens review apply golden
```

Add to `dvc.yaml`:

```yaml
  build-fusion-eval-v1:
    cmd: uv run claimlens review apply parts --job fusion-eval-v1
    deps:
      - data/interim/fusion-eval-v1
      - reviews/fusion-eval-v1.json
    outs:
      - data/processed/fusion-eval-v1
    metrics:
      - reports/data/fusion-eval-v1-review.json:
          cache: false
```

Run: `uv run dvc repro build-fusion-eval-v1 && uv run dvc push`
Expected: approved and rejected counts printed.

- [ ] **Step 4: Re-run the baseline on golden v1**

Run: `uv sync --group vision && uv run claimlens eval-triage --golden evals/golden/v1/claims.jsonl --report evals/reports/<date>-triage-baseline-v1.md`
Expected: report written; fast-track cases now number about 33.

- [ ] **Step 5: Documentation**

Write ADR 0006 (verified auto-labelling: CPU Grounding DINO + SAM 2, exact phrase mapping, approve/reject review as data, observed failure modes from the reports), update the data card with `fusion-eval-v1` and golden v1, write the M2 retro from the real numbers, and tick M2b in `docs/roadmap.md` (add story material to the LinkedIn table).

- [ ] **Step 6: Commit**

```bash
git add reviews evals docs dvc.yaml dvc.lock reports/data
git commit -m "feat: build human-verified fusion evaluation set and baseline on golden v1"
```
