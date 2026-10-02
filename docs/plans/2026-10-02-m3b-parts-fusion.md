# M3b: Part Model, Fusion, Calibration and ONNX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure each damage against the car part it sits on, calibrate the damage model's
confidence, export both models to ONNX, and document them in a model card.

**Architecture:**
- A `Segmenter` protocol returns `SegInstance`s (label, confidence, box, polygon) for one photo.
  `claimlens.vision.masks` rasterises polygons on a 256 × 256 grid.
- `claimlens.fusion.fuse` assigns each damage instance to the part with the largest
  intersection ÷ damage area. `FusedDetector` turns the result into `DamageFinding`s with `part`
  and `part_area_ratio`, so the workflow does not change.
- Pricing uses part-ratio bands when a part is known.
- `claimlens.training.calibration` fits one temperature on validation and recommends a threshold.
- Training and registry code becomes task-aware (damage or parts).

**Tech Stack:** Python 3.12, Ultralytics 8.4, ONNX Runtime, MLflow 3, NumPy, Pillow, Pydantic 2, pytest.

**Spec:** [`docs/specs/2026-10-02-m3b-parts-fusion-design.md`](../specs/2026-10-02-m3b-parts-fusion-design.md)
(builds on [M3a](../specs/2026-10-02-m3a-damage-model-design.md); plan [M3a](2026-10-02-m3a-damage-model.md)).

---

## Plain-language briefing (read first)

**What we're building:**
- A second model that outlines car parts.
- A "fusion" step that says *which part each damage is on* and *how much of that part it covers*.
- Pricing that uses that share, instead of how much of the photo the damage fills.
- A calibration step, so a confidence of 0.8 really means right about 80% of the time.
- Fast ONNX versions of both models, and a model card.

**Why:**
- Today a close-up of a small scratch looks "severe", because it fills the photo.
- The 0.40 confidence rule (R6) was a guess made for a different model.

**Where it sits in the story:** step 3 ("rear bumper, dent, 8% of the bumper") and step 5 (pricing).

**To-do:**
1. Task-aware training registry (damage and parts) (Task 1)
2. Mask geometry and segmentation instances (Task 2)
3. Fusion and `FusedDetector` (Task 3)
4. Part-ratio severity in pricing (Task 4)
5. Calibration maths and `train calibrate` (Task 5)
6. ONNX export and benchmark (Task 6)
7. `--detector fused` and part-agreement evaluation (Task 7)
8. **Results:** import the part model trained on Kaggle, calibrate, export, evaluate (Task 8)
9. Golden baseline with fusion, model card, ADR, retro (Task 9)

**Done looks like:** `uv run claimlens run --policy P-1001 photo.jpg` reports something like
"dent on rear_bumper, 6% of the part, minor". The model card states what both models can and cannot
do. The golden report shows the fused system next to legacy and M3a.

**New terms:**
- **Fusion:** combining two models' outputs (damage plus parts) into one finding.
- **Intersection over area:** how much of the damage outline lies inside a part outline.
- **Calibration:** making confidence scores honest.
- **Temperature:** one number that softens or sharpens all scores. Above 1 means "less sure".
- **ECE (expected calibration error):** the average gap between stated confidence and actual
  accuracy. Lower is better.
- **ONNX:** a standard model file format that runs fast on a CPU without PyTorch.
- **Model card:** a one-page "nutrition label" for a model.

---

## Global Constraints

- Python `>=3.12,<3.13`; every command is `uv run …`. Checks: `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy`, `uv run pytest`.
- Line length 100, mypy strict, coverage ≥ 95%. Model adapters (`vision/ultralytics_segmenter.py`)
  are excluded from coverage.
- Never commit `data/`, `models/` content, `var/` or `.env`. Weights and ONNX files are private
  (ADR 0004).
- **Do not change `min_finding_confidence`** in `config/decision_policy.toml`. Calibration only
  recommends a value.
- Gate: escalation recall on golden v1 with the fused detector must be **1.00**. If it is not, stop
  and report.
- Calibration is fitted on the **validation** split only; the test split is never used to fit
  anything.
- CI needs no GPU, data, weights or network.
- Heavy work runs on Kaggle. On the laptop, run only calibration over validation (814 photos with
  the n model, a few minutes), the benchmarks, part agreement (77 images) and the golden run.
- Old events must still load: new fields on `DamageFinding` have defaults.

## Review Focus

- **A damage outline that touches two parts equally:** the tie goes to the smaller part. Pinned in
  Task 3 (`test_tie_goes_to_the_smaller_part`).
- **A damage polygon with fewer than 3 points:** it has zero area, is never fused, and keeps the
  image fraction. Pinned in Task 3 (`test_degenerate_damage_is_not_fused`).
- **Calibration with zero matched predictions** (for example a wrong label folder): it must be a
  clear error, not a temperature of 0.25. Pinned in Task 5 (`test_fit_needs_predictions`).
- **A damage instance larger than its part** (a ratio above 1, when one dent spans a whole
  mirror): severity must be severe and nothing may crash. Pinned in Task 4
  (`test_ratio_above_one_is_severe`).
- **`models.toml` updated for one task:** the other task's champion must survive. Pinned in Task 1
  (`test_updating_one_task_keeps_the_other`).
- **Old claim events without `part_area_ratio`:** they must still fold. Pinned in Task 3
  (`test_old_findings_without_ratio_still_load`).

---

## File structure

| File | Responsibility |
|---|---|
| `src/claimlens/training/manifest.py` | `ModelReport.task`, `cpu_ms_per_image_onnx` |
| `src/claimlens/training/tracking.py` | Registered model per task |
| `src/claimlens/training/select.py` | `ModelsConfig.parts`, temperature, `update_models_config`, per-task reports |
| `src/claimlens/vision/masks.py` | `rasterize`, `area`, `intersection`, `mask_iou` |
| `src/claimlens/vision/instances.py` | `SegInstance`, `Segmentation`, `Segmenter` |
| `src/claimlens/vision/ultralytics_segmenter.py` | Ultralytics `.pt`/`.onnx` segmenter and ONNX export (coverage-excluded) |
| `src/claimlens/fusion.py` | `fuse`, `FusedDamage`, `FusedDetector` |
| `src/claimlens/domain.py` | `DamageFinding.part_area_ratio` |
| `src/claimlens/pricing.py`, `config/rate_card.toml` | Part-ratio bands, `severity_of` |
| `src/claimlens/training/calibration.py` | Matching, temperature, ECE, threshold, YOLO label reader, part agreement |
| `src/claimlens/training/commands.py` | `--task`, `calibrate`, `export`, `fusion-eval`, ONNX benchmark |
| `src/claimlens/cli.py` | `DetectorSpec`, `resolve_detector`, `--detector fused` |

---

### Task 1: Task-aware registry and models config

**Files:**
- Modify: `src/claimlens/training/manifest.py`, `src/claimlens/training/tracking.py`, `src/claimlens/training/select.py`, `src/claimlens/training/commands.py`
- Test: `tests/unit/test_training_select.py`, `tests/unit/test_training_tracking.py`, `tests/unit/test_training_cli.py`

**Interfaces:**
- Consumes: `TrainingRun.task` (already on this branch).
- Produces:
  - `ModelReport.task: Literal["damage", "parts"] = "damage"` and `ModelReport.cpu_ms_per_image_onnx: float | None = None`.
  - `registered_model(task: str) -> str`, returning `"claimlens-<task>"`.
  - `set_champion_alias(tracking_uri: str, version: str, task: str = "damage")`.
  - `ChampionModel` gains `temperature: float | None = None` and `recommended_threshold: float | None = None`.
  - `ModelsConfig(damage: ChampionModel | None = None, parts: ChampionModel | None = None)`.
  - `update_models_config(path: Path, task: str, champion: ChampionModel) -> ModelsConfig`, which keeps the other task.
  - `champion_from_report(report: ModelReport) -> ChampionModel`.
  - `load_model_reports(reports_dir: Path, task: str | None = None) -> list[ModelReport]`.
  - CLI: `train select --task`, `train report --task`.

`write_models_config` from M3a is replaced by `update_models_config`. Update its callers and tests.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_training_select.py`:

```python
from claimlens.training.select import ChampionModel, champion_from_report, update_models_config


def test_updating_one_task_keeps_the_other(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    damage = champion_from_report(_report("d", 0.6, 0.5))
    parts = ChampionModel(run="p", weights="models/parts/p/best.pt", mlflow_version="1")
    update_models_config(path, "damage", damage)
    config = update_models_config(path, "parts", parts)
    assert config.damage == damage
    assert config.parts == parts
    assert load_models_config(path) == config


def test_temperature_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    champion = champion_from_report(_report("d", 0.6, 0.5)).model_copy(
        update={"temperature": 1.35, "recommended_threshold": 0.45}
    )
    update_models_config(path, "damage", champion)
    loaded = load_models_config(path)
    assert loaded is not None and loaded.damage is not None
    assert loaded.damage.temperature == 1.35
    assert loaded.damage.recommended_threshold == 0.45


def test_reports_can_be_filtered_by_task(tmp_path: Path) -> None:
    write_json(tmp_path / "d.json", _report("d", 0.6, 0.5))
    write_json(tmp_path / "p.json", _report("p", 0.6, 0.5).model_copy(update={"task": "parts"}))
    assert [r.run for r in load_model_reports(tmp_path, task="parts")] == ["p"]
    assert [r.run for r in load_model_reports(tmp_path, task="damage")] == ["d"]
```

In the same file, change `test_models_config_round_trips` to use `update_models_config(path, "damage", champion_from_report(_report("s", 0.6, 0.5)))`, and assert
`written.damage is not None and written.damage.weights == "models/damage/s/best.pt"`. Change
`write_models_config` in the import list to `update_models_config`.

Append to `tests/unit/test_training_tracking.py`:

```python
def test_parts_runs_register_a_parts_model(tmp_path: Path) -> None:
    from claimlens.training.tracking import registered_model
    from tests.training_helpers import make_run_dir

    run_dir = make_run_dir(tmp_path, name="p1", task="parts")
    report = _import(tmp_path, run_dir)
    assert report.task == "parts"
    client = mlflow.MlflowClient(_uri(tmp_path))
    assert client.search_model_versions(f"name='{registered_model('parts')}'")
```

In `tests/training_helpers.py`, give `make_run` and `make_run_dir` a keyword `task: str = "damage"`.
Pass it into `TrainingRun(task=task, …)`, and from `make_run_dir` pass it into `make_run(name, task=task)`.

In `tests/unit/test_training_cli.py`, change the `train select` test's `models.toml` assertion to
`'[damage]' in text and 'run = "damage-yolo11n-v1"' in text`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_training_select.py tests/unit/test_training_tracking.py -q`
Expected: FAIL with `ImportError: cannot import name 'update_models_config'`

- [ ] **Step 3: Implement**

In `manifest.py`, change `ModelReport`. Add `task: Literal["damage", "parts"] = "damage"` after `run`,
and `cpu_ms_per_image_onnx: float | None = None` at the end.

In `tracking.py`:
- Replace `REGISTERED_MODEL = "claimlens-damage"` with `REGISTERED_MODEL = "claimlens-damage"  # kept for M3a tests`.
- Add `def registered_model(task: str) -> str: return f"claimlens-{task}"`.
- In `import_run`, use `name = registered_model(manifest.config.task)` wherever `REGISTERED_MODEL` was
  used, and set `task=manifest.config.task` in the `ModelReport`.
- Keep `EXPERIMENT` as is. One experiment holds both tasks, tagged `claimlens_task`, so add
  `"claimlens_task": manifest.config.task` to the run tags.
- Make `set_champion_alias(tracking_uri, version, task="damage")` use `registered_model(task)`.

In `select.py`, replace the classes and the writer with:

```python
class ChampionModel(Frozen):
    run: str
    weights: str
    mlflow_version: str
    temperature: float | None = None
    recommended_threshold: float | None = None


class ModelsConfig(Frozen):
    damage: ChampionModel | None = None
    parts: ChampionModel | None = None


def load_model_reports(reports_dir: Path, task: str | None = None) -> list[ModelReport]:
    reports = [read_json(path, ModelReport) for path in sorted(reports_dir.glob("*.json"))]
    return [r for r in reports if task is None or r.task == task]


def champion_from_report(report: ModelReport) -> ChampionModel:
    return ChampionModel(
        run=report.run,
        weights=f"models/{report.task}/{report.run}/best.pt",
        mlflow_version=report.model_version,
    )


def _section(task: str, champion: ChampionModel) -> list[str]:
    lines = [
        f"[{task}]",
        f'run = "{champion.run}"',
        f'weights = "{champion.weights}"',
        f'mlflow_version = "{champion.mlflow_version}"',
    ]
    if champion.temperature is not None:
        lines.append(f"temperature = {champion.temperature}")
    if champion.recommended_threshold is not None:
        lines.append(f"recommended_threshold = {champion.recommended_threshold}")
    return lines


def update_models_config(path: Path, task: str, champion: ChampionModel) -> ModelsConfig:
    current = load_models_config(path) or ModelsConfig()
    config = current.model_copy(update={task: champion})
    lines = [
        "# Written by `claimlens train select` and `train calibrate`. The pipeline's defaults."
    ]
    for name in ("damage", "parts"):
        section: ChampionModel | None = getattr(config, name)
        if section is not None:
            lines += ["", *_section(name, section)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return config
```

Delete `write_models_config`. Change `render_model_report`'s title to
`f"# {'Damage' if champion.task == 'damage' else 'Part'} model v1: YOLO11-seg on {champion.dataset}"`.

In `commands.py`:
- Add `--task` (choices `damage` and `parts`, default `damage`) to the `select` and `report` parsers.
- `_select`:

```python
    champion = select_champion(load_model_reports(repo_root / "reports" / "models", args.task))
    update_models_config(args.config / "models.toml", args.task, champion_from_report(champion))
    tracking_uri, _ = default_tracking(repo_root)
    set_champion_alias(tracking_uri, champion.model_version, args.task)
```

- `_report`: load reports for `args.task`, and read `getattr(models, args.task)`. Raise the
  "no champion yet: run `claimlens train select --task …` first" error if that is `None`.
- `_import`: use `models_dir=repo_root / "models"`. In `tracking.import_run`, copy the weights to
  `models_dir / manifest.config.task / run_name / "best.pt"`, and update the M3a tracking tests'
  expected path to `tmp_path / "models" / "damage" / "damage" / "r1"`. **Simpler:** keep `models_dir`
  meaning the task folder and pass `repo_root / "models" / run_task`. To do that, `_import` reads the
  manifest task first with `read_json(args.source / "manifest.json", RunManifest).config.task`. Use
  this simpler option.

In `cli.py`, change `resolve_detector` to read `models.damage` defensively (it is now optional):

```python
    if models is None or models.damage is None:
        raise ValueError("no champion model: run `claimlens train select` or pass --weights")
    return kind, Path(models.damage.weights)
```

and default `kind` to `"yolo-seg"` only when `models is not None and models.damage is not None`.
Task 7 replaces this function.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src tests
git commit -m "feat: make the model registry and models config task-aware"
```

---

### Task 2: Mask geometry and segmentation instances

**Files:**
- Create: `src/claimlens/vision/masks.py`, `src/claimlens/vision/instances.py`, `src/claimlens/vision/ultralytics_segmenter.py`
- Modify: `pyproject.toml` (coverage omit)
- Test: `tests/unit/test_masks.py`

**Interfaces:**
- Produces:
  - `GRID = 256`.
  - `rasterize(polygon_xyn: Sequence[float], size: int = GRID) -> npt.NDArray[np.bool_]`.
  - `area(mask) -> int`, `intersection(a, b) -> int`, `mask_iou(a, b) -> float`.
  - `SegInstance(label, confidence, box_xyxy, polygon_xyn)`.
  - `Segmentation(instances: tuple[SegInstance, ...], width: int, height: int)`.
  - The `Segmenter` protocol: `model_version: str` and `segment(image_path: Path) -> Segmentation`.
  - `UltralyticsSegmenter(weights: Path, *, name: str, conf_floor: float = 0.10)`.
  - `export_onnx(weights: Path, imgsz: int = 640) -> Path`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_masks.py`:

```python
import pytest
from pydantic import ValidationError

from claimlens.vision.instances import SegInstance
from claimlens.vision.masks import GRID, area, intersection, mask_iou, rasterize

QUARTER = (0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5)
RIGHT_QUARTER = (0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5)


def test_quarter_square_covers_a_quarter_of_the_grid() -> None:
    assert area(rasterize(QUARTER)) / GRID**2 == pytest.approx(0.25, abs=0.01)


def test_degenerate_polygons_are_empty() -> None:
    assert area(rasterize(())) == 0
    assert area(rasterize((0.1, 0.1, 0.2, 0.2))) == 0


def test_odd_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="pairs"):
        rasterize((0.1, 0.2, 0.3))


def test_identical_masks_have_iou_one_and_disjoint_zero() -> None:
    a = rasterize(QUARTER)
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, rasterize((0.6, 0.6, 0.9, 0.6, 0.9, 0.9))) == 0.0
    assert mask_iou(rasterize(()), rasterize(())) == 0.0


def test_neighbouring_squares_share_only_their_edge() -> None:
    a, b = rasterize(QUARTER), rasterize(RIGHT_QUARTER)
    assert intersection(a, b) <= GRID // 2 + 1


def test_instances_validate_confidence() -> None:
    with pytest.raises(ValidationError):
        SegInstance(label="dent", confidence=1.5, box_xyxy=(0, 0, 1, 1), polygon_xyn=QUARTER)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_masks.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.vision.instances'`

- [ ] **Step 3: Implement**

`src/claimlens/vision/masks.py`:

```python
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
```

`src/claimlens/vision/instances.py`:

```python
"""Model-agnostic segmentation output: one labelled outline per detected object."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import Field

from claimlens.domain import Frozen


class SegInstance(Frozen):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box_xyxy: tuple[float, float, float, float]
    polygon_xyn: tuple[float, ...]


class Segmentation(Frozen):
    instances: tuple[SegInstance, ...]
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Segmenter(Protocol):
    model_version: str

    def segment(self, image_path: Path) -> Segmentation: ...
```

`src/claimlens/vision/ultralytics_segmenter.py`:

```python
"""Ultralytics YOLO-seg as a `Segmenter`, from `.pt` or `.onnx`; plus ONNX export.

Requires `uv sync --group vision`. Excluded from coverage; see tests/integration/test_fused.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from claimlens.vision.instances import SegInstance, Segmentation


class UltralyticsSegmenter:
    def __init__(self, weights: Path, *, name: str, conf_floor: float = 0.10) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        onnx = weights.suffix == ".onnx"
        if onnx and importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError("onnxruntime is not installed: run `uv sync --group vision`")
        from ultralytics import YOLO

        self._model: Any = YOLO(str(weights), task="segment")
        self._conf_floor = conf_floor
        self.model_version = f"{name}{'-onnx' if onnx else ''}"

    def segment(self, image_path: Path) -> Segmentation:
        result = self._model.predict(
            source=str(image_path), conf=self._conf_floor, imgsz=640, verbose=False
        )[0]
        height, width = result.orig_shape
        if result.masks is None:
            return Segmentation(instances=(), width=int(width), height=int(height))
        names: dict[int, str] = result.names
        instances = tuple(
            SegInstance(
                label=names[int(cls)],
                confidence=float(conf),
                box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                polygon_xyn=tuple(float(v) for v in polygon.reshape(-1).tolist()),
            )
            for box, conf, cls, polygon in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                result.masks.xyn,
                strict=True,
            )
        )
        return Segmentation(instances=instances, width=int(width), height=int(height))


def export_onnx(weights: Path, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    model: Any = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False)
    return Path(str(exported))
```

In `pyproject.toml`, add `"*/claimlens/vision/ultralytics_segmenter.py",` to the coverage omit list.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_masks.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src/claimlens/vision pyproject.toml tests/unit/test_masks.py
git commit -m "feat: add mask geometry and model-agnostic segmentation instances"
```

---

### Task 3: Fusion and `FusedDetector`

**Files:**
- Create: `src/claimlens/fusion.py`
- Modify: `src/claimlens/domain.py` (`part_area_ratio`)
- Test: `tests/unit/test_fusion.py`

**Interfaces:**
- Consumes: `rasterize`, `area`, `intersection` (Task 2); `SegInstance`, `Segmentation`, `Segmenter` (Task 2); `PartGroups` (existing); `normalize_class_name`, `polygon_area_fraction` (existing).
- Produces:
  - `DamageFinding.part_area_ratio: float | None = Field(default=None, ge=0.0)`.
  - `FusedDamage(damage: SegInstance, part_group: str | None, part_area_ratio: float | None)`.
  - `fuse(damage: Sequence[SegInstance], parts: Sequence[SegInstance], groups: PartGroups, *, min_overlap: float = 0.10) -> list[FusedDamage]`.
  - `calibrate_confidence(p: float, temperature: float) -> float`.
  - `FusedDetector(damage: Segmenter, parts: Segmenter, groups: PartGroups, *, temperature: float | None = None, min_overlap: float = 0.10)`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_fusion.py`:

```python
from pathlib import Path

import pytest

from claimlens.data.taxonomy import load_part_groups
from claimlens.domain import BoundingBox, DamageFinding, DamageType
from claimlens.fusion import FusedDetector, calibrate_confidence, fuse
from claimlens.vision.instances import SegInstance, Segmentation

ROOT = Path(__file__).resolve().parents[2]
GROUPS = load_part_groups(ROOT / "config" / "taxonomy.toml")


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _inst(label: str, polygon: tuple[float, ...], confidence: float = 0.9) -> SegInstance:
    return SegInstance(
        label=label, confidence=confidence, box_xyxy=(0, 0, 10, 10), polygon_xyn=polygon
    )


def test_damage_inside_a_part_gets_its_group_and_ratio() -> None:
    dent = _inst("dent", _square(0.1, 0.1, 0.2, 0.2))
    bumper = _inst("back_bumper", _square(0.0, 0.0, 0.5, 0.5))
    (fused,) = fuse([dent], [bumper], GROUPS)
    assert fused.part_group == "rear_bumper"
    assert fused.part_area_ratio == pytest.approx(0.04, abs=0.01)


def test_damage_goes_to_the_part_with_the_largest_overlap() -> None:
    dent = _inst("dent", _square(0.1, 0.1, 0.3, 0.3))
    door = _inst("front_door", _square(0.0, 0.0, 0.25, 1.0))
    fender_light = _inst("front_light", _square(0.25, 0.0, 1.0, 1.0))
    (fused,) = fuse([dent], [door, fender_light], GROUPS)
    assert fused.part_group == "door"


def test_tie_goes_to_the_smaller_part() -> None:
    dent = _inst("dent", _square(0.4, 0.4, 0.6, 0.6))
    big = _inst("hood", _square(0.0, 0.0, 1.0, 1.0))
    small = _inst("left_mirror", _square(0.3, 0.3, 0.7, 0.7))
    (fused,) = fuse([dent], [big, small], GROUPS)
    assert fused.part_group == "mirror"


def test_no_overlap_leaves_the_part_unknown() -> None:
    dent = _inst("dent", _square(0.8, 0.8, 0.9, 0.9))
    (fused,) = fuse([dent], [_inst("hood", _square(0.0, 0.0, 0.3, 0.3))], GROUPS)
    assert fused.part_group is None
    assert fused.part_area_ratio is None


def test_overlap_below_the_minimum_is_ignored() -> None:
    dent = _inst("dent", _square(0.0, 0.0, 0.5, 0.5))
    sliver = _inst("hood", _square(0.45, 0.45, 1.0, 1.0))
    (fused,) = fuse([dent], [sliver], GROUPS, min_overlap=0.10)
    assert fused.part_group is None


def test_degenerate_damage_is_not_fused() -> None:
    (fused,) = fuse([_inst("dent", (0.1, 0.1))], [_inst("hood", _square(0, 0, 1, 1))], GROUPS)
    assert fused.part_group is None


def test_calibration_is_identity_at_temperature_one_and_softens_above() -> None:
    assert calibrate_confidence(0.8, 1.0) == pytest.approx(0.8)
    assert 0.5 < calibrate_confidence(0.8, 2.0) < 0.8
    assert calibrate_confidence(0.2, 2.0) > 0.2


class _FakeSegmenter:
    def __init__(self, name: str, instances: list[SegInstance]) -> None:
        self.model_version = name
        self._instances = tuple(instances)

    def segment(self, image_path: Path) -> Segmentation:
        return Segmentation(instances=self._instances, width=200, height=100)


def test_fused_detector_returns_findings_with_parts() -> None:
    damage = _FakeSegmenter("d", [_inst("dent", _square(0.1, 0.1, 0.2, 0.2), 0.8)])
    parts = _FakeSegmenter("p", [_inst("back_bumper", _square(0.0, 0.0, 0.5, 0.5))])
    detector = FusedDetector(damage, parts, GROUPS, temperature=2.0)
    (finding,) = detector.detect(Path("x.jpg"), "p1")
    assert finding.damage_type is DamageType.DENT
    assert finding.part == "rear_bumper"
    assert finding.part_area_ratio == pytest.approx(0.04, abs=0.01)
    assert finding.confidence == pytest.approx(calibrate_confidence(0.8, 2.0))
    assert finding.image_area_fraction == pytest.approx(0.01)
    assert detector.model_version == "fused:d+p+T2.00"


def test_old_findings_without_ratio_still_load() -> None:
    data = {
        "photo_id": "p1",
        "damage_type": "dent",
        "confidence": 0.9,
        "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        "image_area_fraction": 0.1,
        "part": None,
    }
    finding = DamageFinding.model_validate(data)
    assert finding.part_area_ratio is None
    assert finding.bbox == BoundingBox(x1=0, y1=0, x2=1, y2=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_fusion.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.fusion'`

- [ ] **Step 3: Implement**

In `src/claimlens/domain.py`, add to `DamageFinding` after `part`:

```python
    part_area_ratio: float | None = Field(default=None, ge=0.0)
```

`src/claimlens/fusion.py`:

```python
"""Damage-to-part fusion: which part is each damage on, and how much of that part does it cover."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from claimlens.data.taxonomy import PartGroups
from claimlens.domain import BoundingBox, DamageFinding, Frozen
from claimlens.vision.base import normalize_class_name, polygon_area_fraction
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
    """Each damage goes to the part covering most of it (ties: the smaller part)."""
    part_masks = [(part, rasterize(part.polygon_xyn)) for part in parts]
    fused: list[FusedDamage] = []
    for instance in damage:
        damage_mask = rasterize(instance.polygon_xyn)
        damage_area = area(damage_mask)
        best: tuple[tuple[float, int], SegInstance, int] | None = None
        if damage_area > 0:
            for part, part_mask in part_masks:
                part_area = area(part_mask)
                if part_area == 0:
                    continue
                overlap = intersection(damage_mask, part_mask) / damage_area
                if overlap < min_overlap:
                    continue
                key = (overlap, -part_area)
                if best is None or key > best[0]:
                    best = (key, part, part_area)
        if best is None:
            fused.append(FusedDamage(damage=instance, part_group=None, part_area_ratio=None))
        else:
            _, part, part_area = best
            fused.append(
                FusedDamage(
                    damage=instance,
                    part_group=groups.group_of(part.label),
                    part_area_ratio=damage_area / part_area,
                )
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
            x1, y1, x2, y2 = item.damage.box_xyxy
            confidence = item.damage.confidence
            if self._temperature is not None:
                confidence = calibrate_confidence(confidence, self._temperature)
            findings.append(
                DamageFinding(
                    photo_id=photo_id,
                    damage_type=normalize_class_name(item.damage.label),
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=min(max(x1, 0.0), width),
                        y1=min(max(y1, 0.0), height),
                        x2=min(max(x2, 0.0), width),
                        y2=min(max(y2, 0.0), height),
                    ),
                    image_area_fraction=polygon_area_fraction(item.damage.polygon_xyn),
                    part=item.part_group,
                    part_area_ratio=item.part_area_ratio,
                )
            )
        return findings
```

Check that `BoundingBox` accepts these values. If it rejects `x2 < x1`, the fake's box `(0, 0, 10, 10)`
is valid anyway.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_fusion.py -q && uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src/claimlens/fusion.py src/claimlens/domain.py tests/unit/test_fusion.py
git commit -m "feat: fuse damage with parts and add FusedDetector"
```

---

### Task 4: Part-ratio severity in pricing

**Files:**
- Modify: `src/claimlens/pricing.py`, `config/rate_card.toml`
- Test: `tests/unit/test_pricing.py`

**Interfaces:**
- Consumes: `DamageFinding.part`, `part_area_ratio` (Task 3).
- Produces:
  - `PartBands(minor_max_ratio: float, moderate_max_ratio: float)`.
  - `RateCard.part_bands: PartBands | None = None`.
  - `severity_of(finding: DamageFinding, card: RateCard) -> Severity`.
  - `estimate_cost` uses `severity_of`.
  - Config version `rate-card-2026.10-v1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pricing.py`:

```python
from claimlens.domain import BoundingBox, DamageFinding, DamageType
from claimlens.pricing import PartBands, severity_of


def _finding(fraction: float, part: str | None = None, ratio: float | None = None) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=DamageType.DENT,
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        image_area_fraction=fraction,
        part=part,
        part_area_ratio=ratio,
    )


def test_repo_card_has_part_bands() -> None:
    assert CARD.part_bands == PartBands(minor_max_ratio=0.10, moderate_max_ratio=0.30)
    assert CARD.version == "rate-card-2026.10-v1"


def test_part_ratio_decides_when_a_part_is_known() -> None:
    # A close-up fills 60% of the photo but covers only 5% of the bumper: minor.
    assert severity_of(_finding(0.60, "rear_bumper", 0.05), CARD) is Severity.MINOR
    assert severity_of(_finding(0.01, "door", 0.20), CARD) is Severity.MODERATE


def test_ratio_above_one_is_severe() -> None:
    assert severity_of(_finding(0.30, "mirror", 1.7), CARD) is Severity.SEVERE


def test_without_a_part_the_image_fraction_decides() -> None:
    assert severity_of(_finding(0.60), CARD) is Severity.SEVERE
    assert severity_of(_finding(0.01), CARD) is Severity.MINOR


def test_part_bands_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="minor_max_ratio"):
        PartBands(minor_max_ratio=0.5, moderate_max_ratio=0.3)
```

If `pytest` is not yet imported in that file, add `import pytest`. Make `CARD` the repo card, which
the existing tests already load.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_pricing.py -q`
Expected: FAIL with `ImportError: cannot import name 'PartBands'`

- [ ] **Step 3: Implement**

In `pricing.py`:

```python
class PartBands(Frozen):
    """Severity from damage area / part area."""

    minor_max_ratio: float = Field(gt=0.0)
    moderate_max_ratio: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.minor_max_ratio >= self.moderate_max_ratio:
            raise ValueError("minor_max_ratio must be below moderate_max_ratio")
        return self
```

Add `part_bands: PartBands | None = None` to `RateCard`. In `load_rate_card`, pass
`"part_bands": data.get("part_severity")`. Add:

```python
def severity_of(finding: DamageFinding, card: RateCard) -> Severity:
    bands = card.part_bands
    if bands is not None and finding.part is not None and finding.part_area_ratio is not None:
        if finding.part_area_ratio < bands.minor_max_ratio:
            return Severity.MINOR
        if finding.part_area_ratio < bands.moderate_max_ratio:
            return Severity.MODERATE
        return Severity.SEVERE
    return severity_for(finding.image_area_fraction, card)
```

In `estimate_cost`, replace `severity_for(finding.image_area_fraction, card)` with
`severity_of(finding, card)`.

In `config/rate_card.toml`, set `version = "rate-card-2026.10-v1"` and add after `[severity]`:

```toml
[part_severity]
# Used when fusion knows the part: damage area as a fraction of that part's area.
minor_max_ratio = 0.10
moderate_max_ratio = 0.30
```

Search the tests for `rate-card-2026.10-v0` (`grep -rn "rate-card-2026.10-v0" tests src`) and update any
assertion to `v1`. Golden v1 oracle routes do not change, because oracle findings have no part.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src/claimlens/pricing.py config/rate_card.toml tests
git commit -m "feat: price damage by its share of the part when the part is known"
```

---

### Task 5: Calibration and `train calibrate`

**Files:**
- Create: `src/claimlens/training/calibration.py`
- Modify: `src/claimlens/training/commands.py`, `src/claimlens/cli.py` (`segmenter_factory`)
- Test: `tests/unit/test_calibration.py`, `tests/unit/test_training_cli.py`

**Interfaces:**
- Consumes: `SegInstance`, `Segmenter` (Task 2); `rasterize`, `mask_iou` (Task 2); `calibrate_confidence` (Task 3); `update_models_config`, `load_models_config` (Task 1).
- Produces:
  - `Truth = tuple[str, tuple[float, ...]]`.
  - `read_yolo_labels(path: Path, names: Mapping[int, str]) -> list[Truth]`.
  - `match_predictions(preds: Sequence[SegInstance], truths: Sequence[Truth], *, iou: float = 0.5) -> list[tuple[float, bool]]`.
  - `fit_temperature(pairs: Sequence[tuple[float, bool]]) -> float`.
  - `ece(pairs, *, temperature: float = 1.0, bins: int = 10) -> float`.
  - `recommend_threshold(pairs, temperature, *, precision: float = 0.80) -> float | None`.
  - `CalibrationResult` (`run, images, predictions, correct, temperature, ece_before, ece_after, recommended_threshold, precision_at_threshold, kept_at_threshold`).
  - `SegmenterFactory = Callable[[Path, str], Segmenter]`.
  - `main(..., segmenter_factory=...)` and `run_train_command(..., segmenter_factory=...)`.
  - CLI: `train calibrate <run> [--limit N]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_calibration.py`:

```python
from pathlib import Path

import pytest

from claimlens.fusion import calibrate_confidence
from claimlens.training.calibration import (
    ece,
    fit_temperature,
    match_predictions,
    read_yolo_labels,
    recommend_threshold,
)
from claimlens.vision.instances import SegInstance

SQUARE = (0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4)
OTHER = (0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9)


def _pred(label: str, polygon: tuple[float, ...], confidence: float) -> SegInstance:
    return SegInstance(
        label=label, confidence=confidence, box_xyxy=(0, 0, 1, 1), polygon_xyn=polygon
    )


def test_yolo_labels_are_read_with_class_names(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("1 0.1 0.1 0.4 0.1 0.4 0.4\n\n0 0.5 0.5 0.6 0.5 0.6 0.6\n", encoding="utf-8")
    truths = read_yolo_labels(path, {0: "crack", 1: "dent"})
    assert [t[0] for t in truths] == ["dent", "crack"]
    assert len(truths[0][1]) == 6


def test_matching_needs_same_class_and_overlap() -> None:
    preds = [_pred("dent", SQUARE, 0.9), _pred("scratch", SQUARE, 0.8), _pred("dent", OTHER, 0.7)]
    pairs = match_predictions(preds, [("dent", SQUARE)])
    assert pairs == [(0.9, True), (0.8, False), (0.7, False)]


def test_each_truth_is_matched_once_highest_confidence_first() -> None:
    preds = [_pred("dent", SQUARE, 0.6), _pred("dent", SQUARE, 0.9)]
    assert match_predictions(preds, [("dent", SQUARE)]) == [(0.9, True), (0.6, False)]


def _synthetic(true_temperature: float) -> list[tuple[float, bool]]:
    pairs: list[tuple[float, bool]] = []
    for step in range(5, 100, 5):
        p = step / 100
        positives = round(calibrate_confidence(p, true_temperature) * 200)
        pairs += [(p, True)] * positives + [(p, False)] * (200 - positives)
    return pairs


def test_fit_recovers_a_known_temperature() -> None:
    assert fit_temperature(_synthetic(2.0)) == pytest.approx(2.0, abs=0.1)
    assert fit_temperature(_synthetic(0.5)) == pytest.approx(0.5, abs=0.1)


def test_fit_needs_predictions() -> None:
    with pytest.raises(ValueError, match="no matched predictions"):
        fit_temperature([])


def test_calibration_lowers_ece() -> None:
    pairs = _synthetic(2.0)
    fitted = fit_temperature(pairs)
    assert ece(pairs, temperature=fitted) < ece(pairs)


def test_threshold_is_the_lowest_meeting_the_precision() -> None:
    pairs = [(0.3, False)] * 5 + [(0.5, True)] * 8 + [(0.5, False)] * 2 + [(0.9, True)] * 10
    assert recommend_threshold(pairs, 1.0, precision=0.80) == pytest.approx(0.35)
    assert recommend_threshold([(0.9, False)], 1.0) is None
```

Add to `tests/unit/test_training_cli.py`:

```python
def test_train_calibrate_writes_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    from claimlens.training.manifest import write_json
    from claimlens.training.select import load_models_config
    from claimlens.vision.instances import SegInstance, Segmentation
    from tests.unit.test_training_select import _report

    config = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config)
    (config / "models.toml").unlink(missing_ok=True)
    write_json(tmp_path / "reports" / "models" / "d1.json", _report("d1", 0.6, 0.5))
    dataset = tmp_path / "data" / "processed" / "damage-v1"
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "data.yaml").write_text("names:\n  0: dent\n", encoding="utf-8")
    square = (0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4)
    for i in range(4):
        (dataset / "images" / "val" / f"{i}.jpg").write_bytes(b"x")
        (dataset / "labels" / "val" / f"{i}.txt").write_text(
            "0 " + " ".join(map(str, square)) + "\n", encoding="utf-8"
        )

    class _Seg:
        model_version = "fake"

        def segment(self, image_path: Path) -> Segmentation:
            hit = SegInstance(
                label="dent", confidence=0.9, box_xyxy=(0, 0, 1, 1), polygon_xyn=square
            )
            miss = SegInstance(
                label="dent",
                confidence=0.6,
                box_xyxy=(0, 0, 1, 1),
                polygon_xyn=(0.6, 0.6, 0.9, 0.6, 0.9, 0.9),
            )
            return Segmentation(instances=(hit, miss), width=10, height=10)

    monkeypatch.chdir(tmp_path)
    from claimlens.training.select import champion_from_report, update_models_config

    update_models_config(
        config / "models.toml", "damage", champion_from_report(_report("d1", 0.6, 0.5))
    )
    code = main(
        ["--config", str(config), "train", "calibrate", "d1"],
        segmenter_factory=lambda weights, name: _Seg(),
    )
    assert code == 0
    models = load_models_config(config / "models.toml")
    assert models is not None and models.damage is not None
    assert models.damage.temperature is not None
    assert (tmp_path / "reports" / "models" / "calibration" / "d1.json").is_file()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_calibration.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.training.calibration'`

- [ ] **Step 3: Implement**

`src/claimlens/training/calibration.py`:

```python
"""Confidence calibration: match predictions to truth, fit one temperature, measure ECE."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

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
```

In `test_threshold_is_the_lowest_meeting_the_precision`, check by hand that 0.35 keeps exactly the
0.5 and 0.9 predictions: 18 of 20 are correct, which is 0.90, at or above 0.80. 0.30 keeps all 25:
18 of 25 is 0.72, below 0.80. So 0.35 is the expected answer, and the test is consistent.

In `commands.py`, add the parser and the handler:

```python
    cal = train_sub.add_parser("calibrate", help="fit a temperature on validation predictions")
    cal.add_argument("run")
    cal.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all)")
```

```python
def _calibrate(args: argparse.Namespace, segmenter_factory: SegmenterFactory) -> int:
    import yaml

    from claimlens.training.calibration import (
        CalibrationResult,
        ece,
        fit_temperature,
        match_predictions,
        read_yolo_labels,
        recommend_threshold,
    )
    from claimlens.training.manifest import ModelReport, read_json, write_json
    from claimlens.training.select import load_models_config, update_models_config

    repo_root = Path.cwd()
    report = read_json(repo_root / "reports" / "models" / f"{args.run}.json", ModelReport)
    dataset = repo_root / "data" / "processed" / report.dataset
    names = {
        int(k): str(v)
        for k, v in yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))[
            "names"
        ].items()
    }
    images = sorted((dataset / "images" / "val").glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    segmenter = segmenter_factory(
        repo_root / "models" / report.task / args.run / "best.pt", args.run
    )
    pairs: list[tuple[float, bool]] = []
    for image in images:
        label_file = dataset / "labels" / "val" / f"{image.stem}.txt"
        truths = read_yolo_labels(label_file, names) if label_file.is_file() else []
        pairs += match_predictions(segmenter.segment(image).instances, truths)
    temperature = fit_temperature(pairs)
    threshold = recommend_threshold(pairs, temperature)
    from claimlens.fusion import calibrate_confidence

    kept = [
        c
        for p, c in pairs
        if threshold is not None and calibrate_confidence(p, temperature) >= threshold
    ]
    result = CalibrationResult(
        run=args.run,
        images=len(images),
        predictions=len(pairs),
        correct=sum(c for _, c in pairs),
        temperature=temperature,
        ece_before=round(ece(pairs), 4),
        ece_after=round(ece(pairs, temperature=temperature), 4),
        recommended_threshold=threshold,
        precision_at_threshold=round(sum(kept) / len(kept), 4) if kept else None,
        kept_at_threshold=len(kept),
    )
    write_json(repo_root / "reports" / "models" / "calibration" / f"{args.run}.json", result)
    models = load_models_config(args.config / "models.toml")
    if models is not None and models.damage is not None and models.damage.run == args.run:
        update_models_config(
            args.config / "models.toml",
            "damage",
            models.damage.model_copy(
                update={"temperature": temperature, "recommended_threshold": threshold}
            ),
        )
    print(
        f"{args.run}: T={temperature:.2f}, ECE {result.ece_before:.3f} -> {result.ece_after:.3f}, "
        f"recommended threshold {threshold}"
    )
    return 0
```

Let ruff format wrap the long lines. Define
`SegmenterFactory = Callable[[Path, str], Segmenter]` in `commands.py`, importing `Segmenter` from
`claimlens.vision.instances`. Add a `segmenter_factory: SegmenterFactory` keyword to
`run_train_command`, and dispatch `"calibrate"`.

In `cli.py`, add:

```python
def _ultralytics_segmenter(weights: Path, name: str) -> Segmenter:
    from claimlens.vision.ultralytics_segmenter import UltralyticsSegmenter

    return UltralyticsSegmenter(weights, name=name)
```

Add `segmenter_factory: SegmenterFactory = _ultralytics_segmenter` to `main`, and pass it to
`run_train_command`.

Calibration files live in `reports/models/calibration/`. `load_model_reports` globs only
`reports/models/*.json`, so they are never read as model reports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_calibration.py tests/unit/test_training_cli.py -q && uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src tests
git commit -m "feat: calibrate damage confidence with temperature scaling on validation"
```

---

### Task 6: ONNX export and benchmark

**Files:**
- Modify: `pyproject.toml` (`vision` group: `onnx`, `onnxruntime`), `uv.lock`, `src/claimlens/training/commands.py`, `src/claimlens/cli.py`
- Test: `tests/unit/test_training_cli.py`

**Interfaces:**
- Consumes: `export_onnx` (Task 2); `benchmark_detector` (M3a); `SegmenterFactory` (Task 5); `ModelReport.task`, `cpu_ms_per_image_onnx` (Task 1).
- Produces:
  - `benchmark_callable(fn: Callable[[Path], object], images: Sequence[Path]) -> float`, in `benchmark.py`; `benchmark_detector` calls it.
  - `ExporterFactory = Callable[[Path], Path]`.
  - `main(..., exporter=...)`.
  - CLI: `train export <run>`, and `train benchmark <run> [--format pt|onnx]`. The benchmark uses the segmenter, so it works for both tasks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_training_cli.py`:

```python
def test_train_export_and_onnx_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from claimlens.training.manifest import ModelReport, read_json, write_json
    from claimlens.vision.instances import Segmentation
    from tests.unit.test_training_select import _report

    write_json(
        tmp_path / "reports" / "models" / "p1.json",
        _report("p1", 0.6, 0.5).model_copy(update={"task": "parts", "dataset": "parts-v1"}),
    )
    weights = tmp_path / "models" / "parts" / "p1" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"pt")
    test_dir = tmp_path / "data" / "processed" / "parts-v1" / "images" / "test"
    test_dir.mkdir(parents=True)
    for i in range(3):
        (test_dir / f"{i}.jpg").write_bytes(b"x")

    def fake_export(path: Path) -> Path:
        out = path.with_suffix(".onnx")
        out.write_bytes(b"onnx")
        return out

    class _Seg:
        model_version = "fake"

        def segment(self, image_path: Path) -> Segmentation:
            return Segmentation(instances=(), width=1, height=1)

    seen: list[Path] = []

    def factory(path: Path, name: str) -> _Seg:
        seen.append(path)
        return _Seg()

    monkeypatch.chdir(tmp_path)
    assert main(["train", "export", "p1"], exporter=fake_export) == 0
    assert (tmp_path / "models" / "parts" / "p1" / "best.onnx").read_bytes() == b"onnx"
    assert main(["train", "benchmark", "p1", "--format", "onnx"], segmenter_factory=factory) == 0
    assert seen[-1].suffix == ".onnx"
    report = read_json(tmp_path / "reports" / "models" / "p1.json", ModelReport)
    assert report.cpu_ms_per_image_onnx is not None
    assert report.cpu_ms_per_image is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_training_cli.py -q`
Expected: FAIL (`main()` got an unexpected keyword argument `exporter`)

- [ ] **Step 3: Implement**

In `pyproject.toml`, set the `vision` group to:

```toml
vision = [
  "onnx>=1.16",
  "onnxruntime>=1.20",
  "ultralytics>=8.3,<9",
]
```

Then run `uv lock && uv sync --group training --group review --group vision`.
If the sync breaks `cv2` again, add `--reinstall-package opencv-python-headless`.

In `benchmark.py`, add:

```python
def benchmark_callable(fn: Callable[[Path], object], images: Sequence[Path]) -> float:
    if not images:
        raise ValueError("benchmark needs at least one image")
    fn(images[0])
    timings: list[float] = []
    for image in images:
        start = time.perf_counter()
        fn(image)
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)
```

Make `benchmark_detector(detector, images)` return
`benchmark_callable(lambda image: detector.detect(image, "bench"), images)`. The M3a test expects
4 calls: 1 warm-up plus 3, and that still holds.

In `commands.py`:
- Add the `export` subparser (`run`).
- Add `--format` (choices `pt` and `onnx`, default `pt`) to the `benchmark` subparser.
- Add the handlers:

```python
def _export(args: argparse.Namespace, exporter: ExporterFactory) -> int:
    from claimlens.training.manifest import ModelReport, read_json

    repo_root = Path.cwd()
    report = read_json(repo_root / "reports" / "models" / f"{args.run}.json", ModelReport)
    weights = repo_root / "models" / report.task / args.run / "best.pt"
    exported = exporter(weights)
    target = weights.with_suffix(".onnx")
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    print(f"Exported {args.run} to {target}")
    return 0
```

Rewrite `_benchmark` to use the segmenter, so it works for parts too:

```python
def _benchmark(args: argparse.Namespace, segmenter_factory: SegmenterFactory) -> int:
    from claimlens.training.benchmark import benchmark_callable
    from claimlens.training.manifest import ModelReport, read_json, write_json

    repo_root = Path.cwd()
    report_path = repo_root / "reports" / "models" / f"{args.run}.json"
    report = read_json(report_path, ModelReport)
    test_dir = repo_root / "data" / "processed" / report.dataset / "images" / "test"
    images = sorted(test_dir.glob("*.jpg"))[: args.images]
    weights = repo_root / "models" / report.task / args.run / f"best.{args.format}"
    segmenter = segmenter_factory(weights, args.run)
    ms = round(benchmark_callable(segmenter.segment, images), 1)
    field = "cpu_ms_per_image_onnx" if args.format == "onnx" else "cpu_ms_per_image"
    write_json(report_path, report.model_copy(update={field: ms}))
    print(
        f"{args.run} ({args.format}): median {ms:.0f} ms per image on CPU over {len(images)} images"
    )
    return 0
```

`run_train_command` gains `exporter: ExporterFactory` and dispatches `export`. The benchmark now
uses `segmenter_factory`, so `detector_factory` is no longer needed there; drop that parameter from
`run_train_command`. Update the M3a benchmark test, if one exists, to pass `segmenter_factory`.
Import `shutil` in `commands.py`.

In `cli.py`, add:

```python
def _export_onnx(weights: Path) -> Path:
    from claimlens.vision.ultralytics_segmenter import export_onnx

    return export_onnx(weights)
```

Then add `exporter: ExporterFactory = _export_onnx` to `main`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add pyproject.toml uv.lock src tests
git commit -m "feat: export models to ONNX and benchmark .pt against .onnx"
```

---

### Task 7: `--detector fused` and part-agreement evaluation

**Files:**
- Modify: `src/claimlens/cli.py`, `src/claimlens/training/calibration.py` (`part_agreement`), `src/claimlens/training/commands.py` (`fusion-eval`)
- Test: `tests/unit/test_yolo_seg_math.py` (resolver), `tests/unit/test_calibration.py`, `tests/unit/test_training_cli.py`, `tests/integration/test_fused.py`

**Interfaces:**
- Consumes: `FusedDetector` (Task 3); `UltralyticsSegmenter` (Task 2); `load_part_groups` (existing); `load_models_config` (Task 1).
- Produces:
  - `DetectorSpec(kind: Literal["legacy", "yolo-seg", "fused"], weights: Path, parts_weights: Path | None = None, temperature: float | None = None, taxonomy: Path | None = None)`.
  - `resolve_detector(detector: str | None, weights: Path | None, config_dir: Path) -> DetectorSpec`.
  - `DetectorFactory = Callable[[DetectorSpec], Detector]`.
  - `part_agreement(truths: Mapping[str, Sequence[Truth]], preds: Mapping[str, Sequence[SegInstance]], groups: PartGroups, *, iou: float = 0.5) -> dict[str, tuple[int, int]]`, mapping each group to (agreed, total).
  - CLI: `--detector fused`, and `train fusion-eval <parts-run>`, which writes `reports/models/fusion-eval/<run>.json`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_yolo_seg_math.py`, replace the four `resolve_*` tests with:

```python
from claimlens.cli import DetectorSpec

DAMAGE_ONLY = '[damage]\nrun = "s"\nweights = "models/damage/s/best.pt"\nmlflow_version = "2"\n'
BOTH = (
    DAMAGE_ONLY
    + "temperature = 1.4\n"
    + '[parts]\nrun = "p"\nweights = "models/parts/p/best.pt"\nmlflow_version = "1"\n'
)


def test_resolve_prefers_explicit_arguments(tmp_path: Path) -> None:
    spec = resolve_detector("legacy", tmp_path / "w.pt", tmp_path)
    assert spec == DetectorSpec(kind="legacy", weights=tmp_path / "w.pt")


def test_resolve_defaults_to_legacy_without_a_champion(tmp_path: Path) -> None:
    spec = resolve_detector(None, None, tmp_path)
    assert spec.kind == "legacy"
    assert spec.weights.name == "yolov8n-cardamage-v6.pt"


def test_resolve_uses_the_damage_champion(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(DAMAGE_ONLY, encoding="utf-8")
    spec = resolve_detector(None, None, tmp_path)
    assert (spec.kind, spec.weights) == ("yolo-seg", Path("models/damage/s/best.pt"))


def test_resolve_defaults_to_fused_with_both_champions(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(BOTH, encoding="utf-8")
    spec = resolve_detector(None, None, tmp_path)
    assert spec.kind == "fused"
    assert spec.parts_weights == Path("models/parts/p/best.pt")
    assert spec.temperature == 1.4
    assert spec.taxonomy == tmp_path / "taxonomy.toml"


def test_fused_without_parts_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(DAMAGE_ONLY, encoding="utf-8")
    with pytest.raises(ValueError, match="--task parts"):
        resolve_detector("fused", None, tmp_path)


def test_resolve_yolo_without_champion_or_weights_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train select"):
        resolve_detector("yolo-seg", None, tmp_path)
```

Update `tests/integration/test_yolo_seg.py`'s `_champion()` to use `resolve_detector(...).weights`.

Append to `tests/unit/test_calibration.py`:

```python
from claimlens.data.taxonomy import load_part_groups
from claimlens.training.calibration import part_agreement

GROUPS = load_part_groups(Path(__file__).resolve().parents[2] / "config" / "taxonomy.toml")


def test_part_agreement_counts_per_group() -> None:
    truths = {"img": [("door", SQUARE), ("wheel", OTHER)]}
    preds = {"img": [_pred("front_left_door", SQUARE, 0.9), _pred("hood", OTHER, 0.9)]}
    assert part_agreement(truths, preds, GROUPS) == {"door": (1, 1), "wheel": (0, 1)}


def test_images_without_predictions_count_as_misses() -> None:
    assert part_agreement({"img": [("door", SQUARE)]}, {}, GROUPS) == {"door": (0, 1)}
```

`tests/integration/test_fused.py`:

```python
import importlib.util
from pathlib import Path

import pytest

from claimlens.cli import _default_detector, resolve_detector

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"


def _spec():  # type: ignore[no-untyped-def]
    try:
        spec = resolve_detector("fused", None, ROOT / "config")
    except ValueError:
        return None
    ok = (
        (ROOT / spec.weights).is_file()
        and spec.parts_weights is not None
        and (ROOT / spec.parts_weights).is_file()
    )
    return spec if ok else None


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _spec() is None or importlib.util.find_spec("ultralytics") is None,
        reason="needs damage and parts champions, their weights and `uv sync --group vision`",
    ),
]


def test_fused_detector_runs_on_a_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    spec = _spec()
    detector = _default_detector(spec)
    findings = detector.detect(SAMPLE, "p1")
    assert detector.model_version.startswith("fused:")
    assert all(f.part_area_ratio is None or f.part_area_ratio >= 0 for f in findings)
```

(Ruff/mypy: type `_spec` as `-> DetectorSpec | None`, imported from `claimlens.cli`, instead of the
ignore comment.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_yolo_seg_math.py tests/unit/test_calibration.py -q`
Expected: FAIL with `ImportError: cannot import name 'DetectorSpec'` / `'part_agreement'`

- [ ] **Step 3: Implement**

In `cli.py`:

```python
class DetectorSpec(Frozen):
    kind: Literal["legacy", "yolo-seg", "fused"]
    weights: Path
    parts_weights: Path | None = None
    temperature: float | None = None
    taxonomy: Path | None = None


DetectorFactory = Callable[[DetectorSpec], Detector]


def _default_detector(spec: DetectorSpec) -> Detector:
    if spec.kind == "fused":
        from claimlens.fusion import FusedDetector
        from claimlens.vision.ultralytics_segmenter import UltralyticsSegmenter

        assert spec.parts_weights is not None and spec.taxonomy is not None
        return FusedDetector(
            UltralyticsSegmenter(spec.weights, name=spec.weights.parent.name),
            UltralyticsSegmenter(spec.parts_weights, name=spec.parts_weights.parent.name),
            load_part_groups(spec.taxonomy),
            temperature=spec.temperature,
        )
    if spec.kind == "yolo-seg":
        from claimlens.vision.yolo_seg import YoloSegDetector

        return YoloSegDetector(spec.weights, run=spec.weights.parent.name)
    from claimlens.vision.legacy_yolo import LegacyYoloDetector

    return LegacyYoloDetector(spec.weights)


def resolve_detector(detector: str | None, weights: Path | None, config_dir: Path) -> DetectorSpec:
    """Explicit flags win; otherwise fused (both champions), the damage champion, or legacy."""
    models = load_models_config(config_dir / "models.toml")
    damage = models.damage if models is not None else None
    parts = models.parts if models is not None else None
    default = "fused" if damage and parts else "yolo-seg" if damage else "legacy"
    kind = detector or default
    if kind == "legacy":
        return DetectorSpec(kind="legacy", weights=weights or DEFAULT_WEIGHTS)
    if weights is None and damage is None:
        raise ValueError("no champion model: run `claimlens train select` or pass --weights")
    damage_weights = weights or Path(damage.weights)  # type: ignore[union-attr]
    if kind == "yolo-seg":
        return DetectorSpec(kind="yolo-seg", weights=damage_weights)
    if parts is None:
        raise ValueError("fused needs a parts champion: run `claimlens train select --task parts`")
    return DetectorSpec(
        kind="fused",
        weights=damage_weights,
        parts_weights=Path(parts.weights),
        temperature=damage.temperature if damage is not None else None,
        taxonomy=config_dir / "taxonomy.toml",
    )
```

Avoid the `type: ignore` by writing the `damage_weights` line as an explicit `if` / `else`. Change
the `--detector` choices to `["legacy", "yolo-seg", "fused"]`. `_make_detector` becomes
`return factory(resolve_detector(args.detector, args.weights, args.config))`. Import
`Literal` and `Frozen`, plus `load_part_groups` from `claimlens.data.taxonomy`.

In `calibration.py`, add:

```python
def part_agreement(
    truths: Mapping[str, Sequence[Truth]],
    preds: Mapping[str, Sequence[SegInstance]],
    groups: PartGroups,
    *,
    iou: float = 0.5,
) -> dict[str, tuple[int, int]]:
    """Per part group: how many reviewed parts the part model also finds (same group, IoU >= iou)."""
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
```

with `from claimlens.data.taxonomy import PartGroups`.

In `commands.py`, add `fusion-eval` (argument `run`, the parts run):

```python
def _fusion_eval(args: argparse.Namespace, segmenter_factory: SegmenterFactory) -> int:
    import json

    from claimlens.data.records import read_records
    from claimlens.data.taxonomy import load_part_groups
    from claimlens.training.calibration import part_agreement

    repo_root = Path.cwd()
    records = read_records(repo_root / "data" / "processed" / "fusion-eval-v1" / "parts.jsonl")
    segmenter = segmenter_factory(repo_root / "models" / "parts" / args.run / "best.pt", args.run)
    truths = {r.image_id: [(a.label, a.polygon) for a in r.annotations] for r in records}
    preds = {r.image_id: list(segmenter.segment(repo_root / r.path).instances) for r in records}
    agreement = part_agreement(truths, preds, load_part_groups(args.config / "taxonomy.toml"))
    out = repo_root / "reports" / "models" / "fusion-eval" / f"{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {g: {"agreed": a, "total": t} for g, (a, t) in agreement.items()}
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    agreed, total = sum(a for a, _ in agreement.values()), sum(t for _, t in agreement.values())
    print(f"{args.run}: {agreed} of {total} reviewed parts found ({agreed / max(total, 1):.0%})")
    return 0
```

Check that `Annotation.polygon` is a tuple of floats in the records module (it is used that way in
M2b). Dispatch `fusion-eval`.

Add one CLI test with a fake segmenter: write a one-record `parts.jsonl` (via `write_records`, with an
`Annotation(label="door", polygon=…)`), run `train fusion-eval p1`, and assert the JSON has
`door: {agreed: 1, total: 1}` when the fake returns `front_left_door` with the same polygon.

Update `tests/unit/test_cli.py` and `tests/unit/test_eval_triage.py`: their `lambda *_: chosen`
factories still work with one `DetectorSpec` argument.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass (integration tests skipped)

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src tests
git commit -m "feat: add the fused detector as default and part-agreement evaluation"
```

---

### Task 8: Results — import the part model, calibrate, export, evaluate

Run this after M3a's Task 9 has imported the damage runs and chosen the damage champion, and after
`feat/m3a-damage-model` has been merged into this branch (`git merge feat/m3a-damage-model` or `main`).

- [ ] **Step 1: Download and import the part model**

```bash
export KAGGLE_API_TOKEN="$(grep -E '^KAGGLE_API_TOKEN=' .env | cut -d= -f2-)"
mkdir -p var/kaggle/output/train-parts && cd var/kaggle/output/train-parts && uvx --from kaggle kaggle kernels output karthickbalaje/claimlens-train-parts-v1 -p . && cd -
uv run claimlens train import parts-yolo11n-v1 --from var/kaggle/output/train-parts/parts-yolo11n-v1
uv run claimlens train select --task parts
uv run claimlens train select --task damage
```

Expected: the parts import registers `claimlens-parts` version 1, and both champions appear in
`config/models.toml`. The second `select` rewrites the damage section; it keeps nothing extra yet.

- [ ] **Step 2: Calibrate, export and benchmark**

```bash
uv run claimlens train calibrate <damage-champion>
for run in <damage-champion> parts-yolo11n-v1; do
  uv run claimlens train export $run
  uv run claimlens train benchmark $run --format pt
  uv run claimlens train benchmark $run --format onnx
done
uv run claimlens train fusion-eval parts-yolo11n-v1
```

Expected:
- A temperature, ECE before and after (after is lower), and a recommended threshold.
- Two `.onnx` files.
- Four CPU timings.
- A part-agreement line.

- [ ] **Step 3: Reports and DVC**

```bash
uv run claimlens train report --task damage --out evals/reports/<today>-damage-model-v1.md
uv run claimlens train report --task parts --out evals/reports/<today>-parts-model-v1.md
uv run dvc add models/damage models/parts && uv run dvc push
uv run pytest tests/integration -q
```

Append a "Calibration" section to the damage report and a "Part agreement on fusion-eval-v1" table
to the parts report, by hand from the two JSON files. Use only numbers from those files.

- [ ] **Step 4: Commit**

```bash
git add config/models.toml reports/models evals/reports models/damage.dvc models/parts.dvc models/.gitignore training/kaggle/train-parts
git commit -m "feat: import the part model, calibrate the damage model and export both to ONNX"
```

---

### Task 9: Golden baseline with fusion, model card, ADR and retro

- [ ] **Step 1: Golden run with the fused detector**

```bash
uv run claimlens eval-triage --golden evals/golden/v1/claims.jsonl --report evals/reports/<today>-triage-baseline-v1-fused.md --what-if 0.25,0.40,<recommended>,0.55
```

**Gate:** escalation recall must be 1.00. If not, stop and report the cases; do not change policy.

- [ ] **Step 2: Model card**

`docs/model-card.md`, following Mitchell et al. (2019). For both models, cover:
- details (architecture, version, MLflow version, date, owner);
- intended use and out-of-scope uses (no real claim decisions; no commercial use);
- training data (`damage-v1` and `parts-v1`, with links to the data card);
- metrics (test mask mAP50 overall and per class, from the reports);
- calibration (T, ECE, the recommended threshold, "not applied");
- performance on CPU (`.pt` vs `.onnx`);
- part agreement on `fusion-eval-v1`;
- limitations (stock photos, close-ups, panels, left/right confusion);
- ethical considerations (fraud flags only route to people; there is no auto-deny);
- licence ("private: trained on CarDD, non-commercial; weights are not released").

Every number must come from a report file.

- [ ] **Step 3: ADR 0009 and retro**

- `docs/adr/0009-fusion-and-calibration.md`: fusion by intersection over damage area on a 256-pixel
  grid; part-ratio severity with fallback; one-temperature calibration on validation; threshold
  recommended, not applied; ONNX for portable CPU inference.
- `docs/retros/m3-vision-models.md`: covers M3a and M3b together, with numbers only from the
  reports: damage n vs s, parts, calibration, ONNX speed, and golden legacy vs M3a vs fused.
- `docs/roadmap.md`:
  - tick the M3 items;
  - add story material to the LinkedIn table;
  - add a "**You:** approve or reject the recommended R6 threshold (separate PR)" item.

- [ ] **Step 4: Final checks and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
git add evals/reports docs
git commit -m "docs: add M3 model card, ADR 0009, fused baseline and retro"
```

---

## Self-review notes

- **Spec coverage:** §2 goals (Tasks 8–9), §3 decisions (Tasks 1–7), §5.1–5.2 (Task 2), §5.3
  (Task 3), §5.4 (Task 4), §5.5 (Task 5), §5.6 (Task 6), §5.7 (Task 1 plus the commits already on
  the branch), §5.8 (Task 7), §6 outputs (Tasks 8–9), §7 errors (Tasks 3, 5, 7), §8 tests (each
  task).
- **Deviations:**
  - `YoloSegDetector` (M3a) is kept as is, rather than rewritten over `UltralyticsSegmenter`; the
    fused path uses the segmenter. Cost if wrong: two small Ultralytics adapters.
  - The `train benchmark` command moves from the detector to the segmenter, so it works for parts.
- **Ordering:** the parts training already runs on Kaggle (commit `7b97f54`). Tasks 1–7 need no
  model. Task 8 needs both trainings and M3a's import.
