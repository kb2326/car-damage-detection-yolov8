# M3a: Damage Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train our own YOLO11 damage segmentation model on a free Kaggle GPU, record every run in a
local MLflow logbook, choose the best run on validation, and plug it into the claim pipeline.

**Architecture:** `config/training.toml` defines runs. A private Kaggle kernel checks out a pinned
commit and calls `claimlens train run`, which trains through a `Trainer` protocol and writes a
manifest and metrics. `claimlens train import` logs the run into MLflow (SQLite in `var/mlflow`),
registers it and copies the weights to `models/damage/`. `claimlens train select` picks the
champion into `config/models.toml`. `YoloSegDetector` measures damage size from mask polygons, and
the CLI chooses it with `--detector`.

**Tech Stack:** Python 3.12, Ultralytics 8.4 (YOLO11-seg), MLflow 3 (SQLite backend), Pydantic 2,
Kaggle CLI, DVC, pytest.

**Spec:** [`docs/specs/2026-10-02-m3a-damage-model-design.md`](../specs/2026-10-02-m3a-damage-model-design.md)

---

## Plain-language briefing (read first)

**What we're building:** the system's own "eyes" for damage. Today the pipeline uses the course
model, which over-sizes damage and is often unsure. We train a new model on the 4,000 cleaned
CarDD photos from M2, in two sizes (small "n" and medium "s"). The better one becomes the
**champion** that the claim pipeline uses.

**Why:** 23 of the 26 fast-track misses in baseline v1 come from rules R5 (too expensive, because
damage looks too big) and R6 (the model is unsure). A model trained on the right data, measuring
damage by its outline instead of its box, should fix many of them.

**Where it sits in the story:** step 3, "our own trained vision models find what kind of damage it
is and how big".

**To-do:**
1. Training config and run records (Task 1)
2. The training runner with a fake trainer for tests (Task 2)
3. The real Ultralytics trainer and the `train run` command (Task 3)
4. Import runs into the MLflow logbook (Task 4)
5. Choose the champion and write the model report (Task 5)
6. The new detector, mask-based size, `--detector` switch and CPU benchmark (Task 6)
7. "What-if" thresholds in the golden evaluation (Task 7)
8. Kaggle dataset bundle and training job (Task 8)
9. **Train on Kaggle**, import, select, benchmark (Task 9, about 3–4 hours of Kaggle GPU time; your
   laptop only waits)
10. Golden baseline with the champion, ADR, retro, docs (Task 10)

**Done looks like:** `uv run claimlens run --policy P-1001 photo.jpg` uses our own model, and a
report shows test mask mAP50 for n and s, the champion, CPU speed, and the golden-set result next
to the legacy baseline.

**New terms:**
- **Segmentation:** outlining the exact pixels of each damage, not just a box around it.
- **mAP50:** how well predicted outlines match the true ones (1.0 is perfect). "Mask mAP50" scores
  the outlines; "box mAP50" scores the boxes.
- **Validation and test splits:** validation is used to choose between models; test is touched
  once, to report the final number honestly.
- **Epoch:** one full pass over the training photos. **Early stopping:** stop when validation stops
  improving.
- **MLflow:** a logbook of training runs, plus a registry that records which model is the champion.
- **Champion:** the model version the pipeline uses.

---

## Global Constraints

- Python `>=3.12,<3.13`; every command is `uv run …`. Checks: `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy`, `uv run pytest`.
- Line length 100; mypy strict; coverage stays at or above 95%. Model adapters (`ultralytics_trainer.py`,
  `vision/yolo_seg.py`) are excluded from coverage, like `legacy_yolo.py`.
- Never commit `data/`, `models/` content, `var/` or `.env`. `.dvc` pointer files are allowed.
- Weights are private (CarDD terms, ADR 0004): never pushed to git, Hugging Face or a public Kaggle
  item.
- Selection uses **validation mask mAP50 only**. Test numbers are reported, never used to choose.
- The decision policy (`config/decision_policy.toml`) and the rate card are **not changed** in M3a.
- Gate: escalation recall on golden v1 with the champion must be **1.00**. If it is not, stop and
  report; do not tune the policy to pass.
- CI needs no GPU, data, weights or network. MLflow tests use a temporary SQLite database.
- Heavy model work runs on Kaggle. On the laptop, only the CPU benchmark (20 images) and the golden
  evaluation (97 claims) run, each a few minutes.

## Review Focus

- **Ultralytics `results.csv` headers padded with spaces** (older versions write
  `"      epoch"`): parsing must strip names and values. Pinned in Task 2
  (`test_results_with_padded_headers_are_parsed`).
- **A photo with no detections** (`result.masks is None`, zero boxes): it must give zero findings,
  not crash. Pinned in Task 6 (`test_no_predictions_give_no_findings`).
- **Degenerate or out-of-range mask polygons** (fewer than 3 points, or points slightly outside
  [0, 1]): the area fraction must be 0.0 or clamped to at most 1.0. Pinned in Task 6
  (`test_degenerate_polygon_has_zero_area`, `test_area_fraction_is_clamped`).
- **Importing the same run twice:** there must be no second MLflow run and no second model version.
  Pinned in Task 4 (`test_import_is_idempotent`).
- **Re-running `train run` into a folder that already has results:** it must refuse rather than
  mix two runs. Pinned in Task 2 (`test_existing_run_folder_is_refused`).
- **A golden case that errored** (no final state) in the what-if table: it must count as an error,
  not crash. Pinned in Task 7 (`test_what_if_counts_errored_cases_as_errors`).

---

## File structure

| File | Responsibility |
|---|---|
| `config/training.toml` | Run definitions (defaults plus one table per run) |
| `src/claimlens/training/__init__.py` | Package docstring |
| `src/claimlens/training/config.py` | `TrainingRun`, `load_training_runs`, `dvc_out_md5` |
| `src/claimlens/training/manifest.py` | `SplitMetrics`, `RunManifest`, `RunMetrics`, `ModelReport`, JSON helpers |
| `src/claimlens/training/run.py` | `Trainer` protocol, `run_training`, `epochs_from_results` |
| `src/claimlens/training/ultralytics_trainer.py` | Real trainer (coverage-excluded) |
| `src/claimlens/training/tracking.py` | MLflow import, registry and alias (`import_run`, `set_champion_alias`) |
| `src/claimlens/training/select.py` | Champion choice, `config/models.toml` read and write, model report |
| `src/claimlens/training/benchmark.py` | Median CPU ms per image |
| `src/claimlens/training/commands.py` | `claimlens train …` parser and handlers |
| `src/claimlens/vision/base.py` | Add `polygon_area_fraction`, `area_fractions` argument |
| `src/claimlens/vision/yolo_seg.py` | `YoloSegDetector` (coverage-excluded) |
| `src/claimlens/cli.py` | Wire `train`, `--detector`, `--what-if`, two-argument detector factory |
| `src/claimlens/evals/triage.py` | Final state on `CaseResult`, `what_if_thresholds`, report section |
| `scripts/make_train_bundle.py` | Zip `damage-v1` plus `dataset.json` for Kaggle |
| `training/kaggle/train/run_train.py`, `kernel-metadata.json` | Kaggle job |
| `tests/training_helpers.py` | `FakeTrainer`, `make_run_dir` |
| `tests/unit/test_training_*.py`, `tests/unit/test_yolo_seg_math.py`, `tests/integration/test_yolo_seg.py` | Tests |

---

### Task 1: Training config and run records

**Files:**
- Create: `config/training.toml`, `src/claimlens/training/__init__.py`, `src/claimlens/training/config.py`, `src/claimlens/training/manifest.py`
- Test: `tests/unit/test_training_config.py`

**Interfaces:**
- Produces:
  - `TrainingRun` (fields `name, model, dataset, epochs, patience, imgsz, batch, seed, workers`).
  - `load_training_runs(path: Path) -> dict[str, TrainingRun]`.
  - `dvc_out_md5(lock_path: Path, out_path: str) -> str`.
  - `SplitMetrics`, `RunManifest`, `RunMetrics`, `ModelReport`.
  - `write_json(path: Path, model: BaseModel) -> None` and `read_json[T: BaseModel](path: Path, model: type[T]) -> T`.

- [ ] **Step 1: Write the config file**

`config/training.toml`:

```toml
# Training runs for `claimlens train run <name>` (spec docs/specs/2026-10-02-m3a-damage-model-design.md).
# Each [runs.<name>] table overrides [defaults].

[defaults]
dataset = "damage-v1"
epochs = 100
patience = 20 # stop after 20 epochs without validation improvement
imgsz = 640
batch = 16
seed = 20261002
workers = 2

[runs.damage-yolo11n-v1]
model = "yolo11n-seg.pt"

[runs.damage-yolo11s-v1]
model = "yolo11s-seg.pt"
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_training_config.py`:

```python
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.training.config import TrainingRun, dvc_out_md5, load_training_runs
from claimlens.training.manifest import RunManifest, SplitMetrics, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


def test_repo_training_config_loads_both_runs() -> None:
    runs = load_training_runs(ROOT / "config" / "training.toml")
    assert set(runs) == {"damage-yolo11n-v1", "damage-yolo11s-v1"}
    small = runs["damage-yolo11s-v1"]
    assert small.model == "yolo11s-seg.pt"
    assert small.dataset == "damage-v1"
    assert small.epochs == 100
    assert small.name == "damage-yolo11s-v1"


def test_run_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "training.toml"
    path.write_text(
        '[defaults]\ndataset = "d"\nepochs = 10\npatience = 2\nimgsz = 320\nbatch = 4\n'
        'seed = 1\nworkers = 0\n[runs.quick]\nmodel = "m.pt"\nepochs = 3\n',
        encoding="utf-8",
    )
    assert load_training_runs(path)["quick"].epochs == 3


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "training.toml"
    path.write_text(
        '[defaults]\ndataset = "d"\nepochs = 10\npatience = 2\nimgsz = 320\nbatch = 4\n'
        'seed = 1\nworkers = 0\n[runs.quick]\nmodel = "m.pt"\nlearning_rate = 0.1\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_training_runs(path)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "training.toml"
    path.write_text("[extra]\na = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        load_training_runs(path)


def test_dvc_out_md5_reads_the_lock_file(tmp_path: Path) -> None:
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\nstages:\n  build-damage-v1:\n    outs:\n"
        "    - path: data/processed/damage-v1\n      hash: md5\n      md5: abc.dir\n",
        encoding="utf-8",
    )
    assert dvc_out_md5(lock, "data/processed/damage-v1") == "abc.dir"
    with pytest.raises(ValueError, match="not a stage output"):
        dvc_out_md5(lock, "data/processed/other")


def test_repo_lock_has_the_damage_dataset() -> None:
    assert dvc_out_md5(ROOT / "dvc.lock", "data/processed/damage-v1").endswith(".dir")


def _metrics() -> SplitMetrics:
    return SplitMetrics(
        box_map50=0.6,
        box_map50_95=0.4,
        mask_map50=0.55,
        mask_map50_95=0.35,
        per_class_mask_map50={"dent": 0.5},
    )


def test_manifest_round_trips(tmp_path: Path) -> None:
    run = TrainingRun(
        name="r",
        model="m.pt",
        dataset="d",
        epochs=1,
        patience=0,
        imgsz=64,
        batch=1,
        seed=0,
        workers=0,
    )
    now = datetime(2026, 10, 2, tzinfo=UTC)
    manifest = RunManifest(
        run="r",
        commit="c",
        dataset="d",
        dataset_md5="abc.dir",
        config=run,
        started_at=now,
        finished_at=now,
        status="complete",
        trainer_version="fake",
        epochs_run=1,
        best_epoch=1,
    )
    write_json(tmp_path / "m.json", manifest)
    assert read_json(tmp_path / "m.json", RunManifest) == manifest


def test_metric_values_must_be_fractions() -> None:
    with pytest.raises(ValidationError):
        SplitMetrics(
            box_map50=1.5,
            box_map50_95=0.4,
            mask_map50=0.5,
            mask_map50_95=0.3,
            per_class_mask_map50={},
        )
    assert _metrics().mask_map50 == 0.55
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_training_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.training'`

- [ ] **Step 4: Implement**

`src/claimlens/training/__init__.py`:

```python
"""Model training: run definitions, the training runner, experiment tracking and selection."""
```

`src/claimlens/training/config.py`:

```python
"""Training run definitions (`config/training.toml`) and dataset hashes from `dvc.lock`."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from claimlens.domain import Frozen


class TrainingRun(Frozen):
    name: str
    model: str
    dataset: str
    epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    imgsz: int = Field(gt=0)
    batch: int = Field(gt=0)
    seed: int
    workers: int = Field(ge=0)


def load_training_runs(path: Path) -> dict[str, TrainingRun]:
    data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(data) - {"defaults", "runs"})
    if unknown:
        raise ValueError(f"unknown sections in {path.name}: {unknown}")
    defaults: dict[str, Any] = data.get("defaults", {})
    runs: dict[str, dict[str, Any]] = data.get("runs", {})
    return {
        name: TrainingRun.model_validate({**defaults, **body, "name": name})
        for name, body in runs.items()
    }


def dvc_out_md5(lock_path: Path, out_path: str) -> str:
    """The md5 DVC recorded for a stage output, e.g. `data/processed/damage-v1`."""
    lock: dict[str, Any] = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    for stage in lock.get("stages", {}).values():
        for out in stage.get("outs", []):
            if out.get("path") == out_path:
                return str(out["md5"])
    raise ValueError(f"{out_path} is not a stage output in {lock_path}")
```

`src/claimlens/training/manifest.py`:

```python
"""What a training run leaves behind: its manifest, its metrics, and the imported model report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from claimlens.domain import Frozen
from claimlens.training.config import TrainingRun


class SplitMetrics(Frozen):
    box_map50: float = Field(ge=0.0, le=1.0)
    box_map50_95: float = Field(ge=0.0, le=1.0)
    mask_map50: float = Field(ge=0.0, le=1.0)
    mask_map50_95: float = Field(ge=0.0, le=1.0)
    per_class_mask_map50: dict[str, float]


class RunManifest(Frozen):
    run: str
    commit: str
    dataset: str
    dataset_md5: str
    config: TrainingRun
    started_at: datetime
    finished_at: datetime
    status: Literal["complete", "incomplete"]
    trainer_version: str
    epochs_run: int = Field(ge=0)
    best_epoch: int = Field(ge=0)
    note: str = ""


class RunMetrics(Frozen):
    run: str
    val: SplitMetrics
    test: SplitMetrics


class ModelReport(Frozen):
    """Git-tracked summary of an imported run (`reports/models/<run>.json`); no weights."""

    run: str
    base_model: str
    commit: str
    dataset: str
    dataset_md5: str
    status: Literal["complete", "incomplete"]
    epochs_run: int
    best_epoch: int
    val: SplitMetrics
    test: SplitMetrics
    mlflow_run_id: str
    model_version: str
    cpu_ms_per_image: float | None = None


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")


def read_json[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text(encoding="utf-8"))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_training_config.py -q`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add config/training.toml src/claimlens/training tests/unit/test_training_config.py
git commit -m "feat: add training run config and run records"
```

---

### Task 2: Training runner

**Files:**
- Create: `src/claimlens/training/run.py`, `tests/training_helpers.py`
- Test: `tests/unit/test_training_run.py`

**Interfaces:**
- Consumes: `TrainingRun`, `SplitMetrics`, `RunManifest`, `RunMetrics`, `write_json` (Task 1).
- Produces:
  - The `Trainer` protocol: attribute `version: str`; `train(run, *, data_yaml, project_dir) -> None`, which must leave `project_dir/weights/best.pt` and `project_dir/results.csv`; `evaluate(weights, *, data_yaml, split, imgsz) -> SplitMetrics`.
  - `DatasetMismatchError(ValueError)`.
  - `run_training(run: TrainingRun, *, dataset_dir: Path, bundle_md5: str, expected_md5: str, commit: str, out_dir: Path, trainer: Trainer, now: Callable[[], datetime] = ...) -> RunManifest`. It writes `out_dir/<run>/manifest.json` and `metrics.json`; the weights end up in `out_dir/<run>/train/weights/best.pt`.
  - `epochs_from_results(path: Path) -> tuple[int, int]`, returning `(epochs_run, best_epoch)`.
  - `tests.training_helpers.FakeTrainer`, `make_run(name="r1") -> TrainingRun`, and `make_run_dir(tmp_path, *, name="r1", fail=False) -> Path`.

- [ ] **Step 1: Write the shared test helpers**

`tests/training_helpers.py`:

```python
"""Fakes for training tests: no GPU, no Ultralytics."""

from __future__ import annotations

from pathlib import Path

from claimlens.training.config import TrainingRun
from claimlens.training.manifest import SplitMetrics
from claimlens.training.run import run_training

# Ultralytics pads header names with spaces in some versions.
RESULTS_CSV = (
    "                  epoch,      metrics/mAP50(B),   metrics/mAP50-95(B),"
    "      metrics/mAP50(M),   metrics/mAP50-95(M)\n"
    "                      1,                 0.2,                 0.1,"
    "                 0.2,                 0.1\n"
    "                      2,                 0.5,                 0.3,"
    "                 0.4,                 0.3\n"
    "                      3,                 0.4,                 0.2,"
    "                 0.3,                 0.2\n"
)


class FakeTrainer:
    version = "fake-trainer-1"

    def __init__(self, *, fail_after_weights: bool = False, write_weights: bool = True) -> None:
        self.fail_after_weights = fail_after_weights
        self.write_weights = write_weights
        self.evaluated: list[str] = []

    def train(self, run: TrainingRun, *, data_yaml: Path, project_dir: Path) -> None:
        weights = project_dir / "weights"
        weights.mkdir(parents=True)
        if self.write_weights:
            (weights / "best.pt").write_bytes(b"weights")
        (project_dir / "results.csv").write_text(RESULTS_CSV, encoding="utf-8")
        (project_dir / "results.png").write_bytes(b"png")
        if self.fail_after_weights:
            raise RuntimeError("CUDA out of memory")

    def evaluate(self, weights: Path, *, data_yaml: Path, split: str, imgsz: int) -> SplitMetrics:
        self.evaluated.append(split)
        return SplitMetrics(
            box_map50=0.6,
            box_map50_95=0.4,
            mask_map50=0.55 if split == "val" else 0.5,
            mask_map50_95=0.35,
            per_class_mask_map50={"dent": 0.5, "scratch": 0.45},
        )


def make_run(name: str = "r1", model: str = "yolo11n-seg.pt") -> TrainingRun:
    return TrainingRun(
        name=name,
        model=model,
        dataset="damage-v1",
        epochs=3,
        patience=0,
        imgsz=64,
        batch=1,
        seed=0,
        workers=0,
    )


def make_run_dir(tmp_path: Path, *, name: str = "r1", fail: bool = False) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "data.yaml").write_text("names: {0: dent}\n", encoding="utf-8")
    run_training(
        make_run(name),
        dataset_dir=dataset,
        bundle_md5="abc.dir",
        expected_md5="abc.dir",
        commit="0123abc",
        out_dir=tmp_path / "runs",
        trainer=FakeTrainer(fail_after_weights=fail),
    )
    return tmp_path / "runs" / name
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_training_run.py`:

```python
from pathlib import Path

import pytest

from claimlens.training.manifest import RunManifest, RunMetrics, read_json
from claimlens.training.run import DatasetMismatchError, epochs_from_results, run_training
from tests.training_helpers import RESULTS_CSV, FakeTrainer, make_run, make_run_dir


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: {0: dent}\n", encoding="utf-8")
    return dataset


def test_a_complete_run_writes_manifest_and_metrics(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    manifest = read_json(run_dir / "manifest.json", RunManifest)
    metrics = read_json(run_dir / "metrics.json", RunMetrics)
    assert manifest.status == "complete"
    assert manifest.commit == "0123abc"
    assert manifest.dataset_md5 == "abc.dir"
    assert manifest.trainer_version == "fake-trainer-1"
    assert (manifest.epochs_run, manifest.best_epoch) == (3, 2)
    assert metrics.val.mask_map50 == 0.55
    assert metrics.test.mask_map50 == 0.5
    assert (run_dir / "train" / "weights" / "best.pt").read_bytes() == b"weights"


def test_validation_and_test_are_each_evaluated_once(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    run_training(
        make_run(),
        dataset_dir=_dataset(tmp_path),
        bundle_md5="a",
        expected_md5="a",
        commit="c",
        out_dir=tmp_path / "runs",
        trainer=trainer,
    )
    assert trainer.evaluated == ["val", "test"]


def test_a_stale_bundle_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetMismatchError, match="old.dir"):
        run_training(
            make_run(),
            dataset_dir=_dataset(tmp_path),
            bundle_md5="old.dir",
            expected_md5="new.dir",
            commit="c",
            out_dir=tmp_path / "runs",
            trainer=FakeTrainer(),
        )


def test_missing_data_yaml_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data\.yaml"):
        run_training(
            make_run(),
            dataset_dir=tmp_path,
            bundle_md5="a",
            expected_md5="a",
            commit="c",
            out_dir=tmp_path / "runs",
            trainer=FakeTrainer(),
        )


def test_existing_run_folder_is_refused(tmp_path: Path) -> None:
    make_run_dir(tmp_path)
    with pytest.raises(FileExistsError, match="r1"):
        make_run_dir(tmp_path)


def test_a_crash_after_a_checkpoint_gives_an_incomplete_run(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, fail=True)
    manifest = read_json(run_dir / "manifest.json", RunManifest)
    assert manifest.status == "incomplete"
    assert "CUDA out of memory" in manifest.note


def test_a_crash_without_a_checkpoint_is_raised(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        run_training(
            make_run(),
            dataset_dir=_dataset(tmp_path),
            bundle_md5="a",
            expected_md5="a",
            commit="c",
            out_dir=tmp_path / "runs",
            trainer=FakeTrainer(fail_after_weights=True, write_weights=False),
        )


def test_results_with_padded_headers_are_parsed(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(RESULTS_CSV, encoding="utf-8")
    assert epochs_from_results(path) == (3, 2)


def test_empty_results_give_zero_epochs(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("epoch,metrics/mAP50(B)\n", encoding="utf-8")
    assert epochs_from_results(path) == (0, 0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_training_run.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.training.run'`

- [ ] **Step 4: Implement**

`src/claimlens/training/run.py`:

```python
"""Run one training job: check the dataset, train, evaluate validation and test once, record it."""

from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from claimlens.training.config import TrainingRun
from claimlens.training.manifest import RunManifest, RunMetrics, SplitMetrics, write_json

# Ultralytics' segmentation fitness: weights for mAP50 and mAP50-95 of boxes (B) and masks (M).
_FITNESS = {
    "metrics/mAP50(B)": 0.1,
    "metrics/mAP50-95(B)": 0.9,
    "metrics/mAP50(M)": 0.1,
    "metrics/mAP50-95(M)": 0.9,
}


class Trainer(Protocol):
    version: str

    def train(self, run: TrainingRun, *, data_yaml: Path, project_dir: Path) -> None:
        """Train; must leave `project_dir/weights/best.pt` and `project_dir/results.csv`."""
        ...

    def evaluate(
        self, weights: Path, *, data_yaml: Path, split: str, imgsz: int
    ) -> SplitMetrics: ...


class DatasetMismatchError(ValueError):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


def epochs_from_results(path: Path) -> tuple[int, int]:
    """(epochs run, best epoch by Ultralytics' fitness) from `results.csv`."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        return 0, 0

    def fitness(row: dict[str, str]) -> float:
        return sum(weight * float(row.get(key) or 0.0) for key, weight in _FITNESS.items())

    best = max(rows, key=fitness)
    return len(rows), int(float(best["epoch"]))


def run_training(
    run: TrainingRun,
    *,
    dataset_dir: Path,
    bundle_md5: str,
    expected_md5: str,
    commit: str,
    out_dir: Path,
    trainer: Trainer,
    now: Callable[[], datetime] = _utc_now,
) -> RunManifest:
    if bundle_md5 != expected_md5:
        raise DatasetMismatchError(
            f"dataset bundle md5 {bundle_md5} does not match dvc.lock ({expected_md5}); "
            "rebuild the bundle with scripts/make_train_bundle.py"
        )
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset has no data.yaml: {data_yaml}")
    run_dir = out_dir / run.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run folder already has results: {run_dir}")
    project_dir = run_dir / "train"
    best = project_dir / "weights" / "best.pt"

    started = now()
    status: Literal["complete", "incomplete"] = "complete"
    note = ""
    try:
        trainer.train(run, data_yaml=data_yaml, project_dir=project_dir)
    except Exception as exc:
        if not best.is_file():
            raise
        status, note = "incomplete", f"training stopped early: {type(exc).__name__}: {exc}"
    if not best.is_file():
        raise RuntimeError(f"training finished without weights at {best}")

    epochs_run, best_epoch = epochs_from_results(project_dir / "results.csv")
    val = trainer.evaluate(best, data_yaml=data_yaml, split="val", imgsz=run.imgsz)
    test = trainer.evaluate(best, data_yaml=data_yaml, split="test", imgsz=run.imgsz)
    manifest = RunManifest(
        run=run.name,
        commit=commit,
        dataset=run.dataset,
        dataset_md5=expected_md5,
        config=run,
        started_at=started,
        finished_at=now(),
        status=status,
        trainer_version=trainer.version,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        note=note,
    )
    write_json(run_dir / "metrics.json", RunMetrics(run=run.name, val=val, test=test))
    write_json(run_dir / "manifest.json", manifest)
    return manifest
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_training_run.py tests/unit/test_training_config.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src/claimlens/training/run.py tests/training_helpers.py tests/unit/test_training_run.py
git commit -m "feat: add training runner with dataset check and incomplete runs"
```

---

### Task 3: Ultralytics trainer and `claimlens train run`

**Files:**
- Create: `src/claimlens/training/ultralytics_trainer.py`, `src/claimlens/training/commands.py`
- Modify: `src/claimlens/cli.py` (register `train`, accept `trainer_factory`), `pyproject.toml` (coverage omit)
- Test: `tests/unit/test_training_cli.py`

**Interfaces:**
- Consumes: `load_training_runs`, `dvc_out_md5`, `run_training`, `Trainer` (Tasks 1–2).
- Produces:
  - `UltralyticsTrainer(device: str = "0")`.
  - `add_train_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None`.
  - `run_train_command(args: argparse.Namespace, *, trainer_factory: TrainerFactory, detector_factory: Callable[[str, Path], Detector]) -> int`.
  - `TrainerFactory = Callable[[], Trainer]`.
  - `BundleInfo(dataset: str, md5: str)`.
  - `main(..., trainer_factory: TrainerFactory = _ultralytics_trainer)`.

`detector_factory` is unused until Task 6 (benchmark), but it is accepted now so `cli.main` wiring does
not change again.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_training_cli.py`:

```python
import json
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.training.manifest import RunManifest, read_json
from tests.fakes import CONFIG_DIR
from tests.training_helpers import FakeTrainer

LOCK = (
    "schema: '2.0'\nstages:\n  build-damage-v1:\n    outs:\n"
    "    - path: data/processed/damage-v1\n      hash: md5\n      md5: abc.dir\n"
)


def _setup(tmp_path: Path, bundle_md5: str = "abc.dir") -> Path:
    (tmp_path / "dvc.lock").write_text(LOCK, encoding="utf-8")
    dataset = tmp_path / "data" / "processed" / "damage-v1"
    dataset.mkdir(parents=True)
    (dataset / "data.yaml").write_text("names: {0: dent}\n", encoding="utf-8")
    bundle = tmp_path / "dataset.json"
    bundle.write_text(json.dumps({"dataset": "damage-v1", "md5": bundle_md5}), encoding="utf-8")
    return bundle


def _train(tmp_path: Path, *extra: str) -> int:
    return main(
        [
            "--config",
            str(CONFIG_DIR),
            "train",
            "run",
            "damage-yolo11n-v1",
            "--dataset-dir",
            str(tmp_path / "data" / "processed" / "damage-v1"),
            "--bundle",
            str(tmp_path / "dataset.json"),
            "--out",
            str(tmp_path / "runs"),
            "--commit",
            "0123abc",
            *extra,
        ],
        trainer_factory=FakeTrainer,
    )


def test_train_run_writes_a_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _train(tmp_path) == 0
    manifest = read_json(tmp_path / "runs" / "damage-yolo11n-v1" / "manifest.json", RunManifest)
    assert manifest.config.model == "yolo11n-seg.pt"
    assert manifest.commit == "0123abc"


def test_train_run_refuses_a_stale_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(tmp_path, bundle_md5="old.dir")
    monkeypatch.chdir(tmp_path)
    assert _train(tmp_path) == 1
    assert "does not match dvc.lock" in capsys.readouterr().err


def test_unknown_run_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "--config",
            str(CONFIG_DIR),
            "train",
            "run",
            "no-such-run",
            "--dataset-dir",
            str(tmp_path),
            "--bundle",
            str(tmp_path / "dataset.json"),
            "--commit",
            "c",
        ],
        trainer_factory=FakeTrainer,
    )
    assert code == 1
    assert "unknown training run" in capsys.readouterr().err


def test_bundle_for_another_dataset_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _setup(tmp_path)
    bundle.write_text(json.dumps({"dataset": "parts-v1", "md5": "abc.dir"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert _train(tmp_path) == 1
    assert "parts-v1" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_training_cli.py -q`
Expected: FAIL (`main()` got an unexpected keyword argument `trainer_factory`, or the parser rejects `train`)

- [ ] **Step 3: Implement the real trainer**

`src/claimlens/training/ultralytics_trainer.py`:

```python
"""Ultralytics YOLO11-seg training and evaluation. Requires `uv sync --group vision` and a GPU.

Excluded from coverage: it is exercised by the Kaggle job, not by CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claimlens.training.config import TrainingRun
from claimlens.training.manifest import SplitMetrics


class UltralyticsTrainer:
    def __init__(self, device: str = "0") -> None:
        import ultralytics

        self._device = device
        self.version = f"ultralytics-{ultralytics.__version__}"

    def train(self, run: TrainingRun, *, data_yaml: Path, project_dir: Path) -> None:
        from ultralytics import YOLO

        model: Any = YOLO(run.model)
        model.train(
            data=str(data_yaml),
            epochs=run.epochs,
            patience=run.patience,
            imgsz=run.imgsz,
            batch=run.batch,
            seed=run.seed,
            deterministic=True,
            workers=run.workers,
            device=self._device,
            project=str(project_dir.parent),
            name=project_dir.name,
            exist_ok=True,
            plots=True,
            verbose=False,
        )

    def evaluate(self, weights: Path, *, data_yaml: Path, split: str, imgsz: int) -> SplitMetrics:
        from ultralytics import YOLO

        model: Any = YOLO(str(weights))
        result: Any = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            batch=16,
            device=self._device,
            plots=False,
            verbose=False,
            project=str(weights.parents[2] / "eval"),
            name=split,
            exist_ok=True,
        )
        names: dict[int, str] = model.names
        per_class = {
            names[int(index)]: float(ap)
            for index, ap in zip(result.seg.ap_class_index, result.seg.ap50, strict=True)
        }
        return SplitMetrics(
            box_map50=float(result.box.map50),
            box_map50_95=float(result.box.map),
            mask_map50=float(result.seg.map50),
            mask_map50_95=float(result.seg.map),
            per_class_mask_map50=per_class,
        )
```

- [ ] **Step 4: Implement the command module**

`src/claimlens/training/commands.py`:

```python
"""`claimlens train …`: run a training job, import it, choose the champion, benchmark it."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from claimlens.domain import Frozen
from claimlens.training.config import dvc_out_md5, load_training_runs
from claimlens.training.manifest import read_json
from claimlens.training.run import Trainer, run_training
from claimlens.vision.base import Detector

TrainerFactory = Callable[[], Trainer]
DetectorFactory = Callable[[str, Path], Detector]


class BundleInfo(Frozen):
    dataset: str
    md5: str


def add_train_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train = sub.add_parser("train", help="train models and manage the model registry")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    run = train_sub.add_parser("run", help="train one run (used inside the Kaggle job)")
    run.add_argument("run")
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--bundle", type=Path, required=True, help="dataset.json from the bundle")
    run.add_argument("--out", type=Path, default=Path("var/runs"))
    run.add_argument("--commit", default=None, help="defaults to `git rev-parse HEAD`")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _run(args: argparse.Namespace, trainer_factory: TrainerFactory) -> int:
    runs = load_training_runs(args.config / "training.toml")
    if args.run not in runs:
        raise ValueError(f"unknown training run {args.run!r}; known: {sorted(runs)}")
    run = runs[args.run]
    bundle = read_json(args.bundle, BundleInfo)
    if bundle.dataset != run.dataset:
        raise ValueError(
            f"bundle holds {bundle.dataset!r} but run {run.name} needs {run.dataset!r}"
        )
    expected = dvc_out_md5(Path.cwd() / "dvc.lock", f"data/processed/{run.dataset}")
    manifest = run_training(
        run,
        dataset_dir=args.dataset_dir,
        bundle_md5=bundle.md5,
        expected_md5=expected,
        commit=args.commit or _git_commit(),
        out_dir=args.out,
        trainer=trainer_factory(),
    )
    print(
        f"Trained {manifest.run}: {manifest.status}, {manifest.epochs_run} epochs, "
        f"best epoch {manifest.best_epoch}"
    )
    return 0


def run_train_command(
    args: argparse.Namespace,
    *,
    trainer_factory: TrainerFactory,
    detector_factory: DetectorFactory,
) -> int:
    try:
        if args.train_command == "run":
            return _run(args, trainer_factory)
        raise ValueError(f"unknown train command {args.train_command!r}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
```

- [ ] **Step 5: Wire it into `cli.py`**

In `src/claimlens/cli.py`:

1. Add the import `from claimlens.training.commands import TrainerFactory, add_train_parser, run_train_command`.
2. Add this beside `_legacy_detector`:

```python
def _ultralytics_trainer() -> Trainer:
    from claimlens.training.ultralytics_trainer import UltralyticsTrainer

    return UltralyticsTrainer()
```

   with `from claimlens.training.run import Trainer` imported.
3. Call `add_train_parser(sub)` in `build_parser()` just before `return parser`.
4. Add the parameter `trainer_factory: TrainerFactory = _ultralytics_trainer` to `main`. Then add this branch after the `review` branch:

```python
    if args.command == "train":
        return run_train_command(
            args, trainer_factory=trainer_factory, detector_factory=lambda kind, w: detector_factory(w)
        )
```

   Task 6 replaces this lambda with `detector_factory` directly, once the factory takes two arguments.

In `pyproject.toml`, add `"*/claimlens/training/ultralytics_trainer.py",` to `[tool.coverage.run] omit`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_training_cli.py tests/unit/test_cli.py -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src/claimlens/training src/claimlens/cli.py pyproject.toml tests/unit/test_training_cli.py
git commit -m "feat: add Ultralytics trainer and claimlens train run"
```

---

### Task 4: Import runs into the MLflow logbook

**Files:**
- Create: `src/claimlens/training/tracking.py`
- Modify: `pyproject.toml` (group `training`, mypy override), `.github/workflows/ci.yml` (install `--group training`), `src/claimlens/training/commands.py` (`train import`), `uv.lock`
- Test: `tests/unit/test_training_tracking.py`

**Interfaces:**
- Consumes: `RunManifest`, `RunMetrics`, `ModelReport`, `read_json`, `write_json` (Task 1); `make_run_dir` (Task 2).
- Produces:
  - `REGISTERED_MODEL = "claimlens-damage"`.
  - `TrainingUnavailableError(RuntimeError)`.
  - `default_tracking(repo_root: Path) -> tuple[str, Path]`, returning `(tracking_uri, artifact_root)`.
  - `import_run(run_dir: Path, *, run_name: str, tracking_uri: str, artifact_root: Path, models_dir: Path, reports_dir: Path, allow_incomplete: bool = False) -> ModelReport`. It writes `reports_dir/<run>.json` and `models_dir/<run>/best.pt`.
  - `set_champion_alias(tracking_uri: str, version: str) -> None`.

- [ ] **Step 1: Add the dependency group**

In `pyproject.toml` `[dependency-groups]`, add:

```toml
training = [
  "mlflow>=3.0",
]
```

and add `"mlflow", "mlflow.*"` to the mypy override that lists `ultralytics`. Then run:

```bash
uv lock && uv sync --group training --group review
```

In `.github/workflows/ci.yml`, change the install step to `run: uv sync --locked --group training`.

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_training_tracking.py`:

```python
from pathlib import Path

import pytest

from claimlens.training.manifest import ModelReport, read_json
from tests.training_helpers import make_run_dir

mlflow = pytest.importorskip("mlflow")

from claimlens.training.tracking import (  # noqa: E402
    REGISTERED_MODEL,
    import_run,
    set_champion_alias,
)


def _uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _import(tmp_path: Path, run_dir: Path, **kwargs: bool) -> ModelReport:
    return import_run(
        run_dir,
        run_name=run_dir.name,
        tracking_uri=_uri(tmp_path),
        artifact_root=tmp_path / "artifacts",
        models_dir=tmp_path / "models" / "damage",
        reports_dir=tmp_path / "reports",
        **kwargs,
    )


def test_import_logs_registers_and_copies_weights(tmp_path: Path) -> None:
    report = _import(tmp_path, make_run_dir(tmp_path))
    assert report.model_version == "1"
    assert report.val.mask_map50 == 0.55
    assert report.base_model == "yolo11n-seg.pt"
    assert (tmp_path / "models" / "damage" / "r1" / "best.pt").read_bytes() == b"weights"
    assert read_json(tmp_path / "reports" / "r1.json", ModelReport) == report
    client = mlflow.MlflowClient(_uri(tmp_path))
    run = client.get_run(report.mlflow_run_id)
    assert run.data.params["model"] == "yolo11n-seg.pt"
    assert run.data.tags["claimlens_commit"] == "0123abc"
    assert run.data.metrics["val_mask_map50"] == pytest.approx(0.55)
    assert run.data.metrics["test_mask_map50_dent"] == pytest.approx(0.5)
    history = client.get_metric_history(report.mlflow_run_id, "metrics/mAP50_M")
    assert [m.step for m in history] == [1, 2, 3]


def test_import_is_idempotent(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    first = _import(tmp_path, run_dir)
    second = _import(tmp_path, run_dir)
    assert second == first
    client = mlflow.MlflowClient(_uri(tmp_path))
    assert len(client.search_model_versions(f"name='{REGISTERED_MODEL}'")) == 1


def test_incomplete_run_needs_explicit_permission(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, fail=True)
    with pytest.raises(ValueError, match="incomplete"):
        _import(tmp_path, run_dir)
    assert _import(tmp_path, run_dir, allow_incomplete=True).status == "incomplete"


def test_run_name_must_match_the_manifest(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    with pytest.raises(ValueError, match="manifest is for run 'r1'"):
        import_run(
            run_dir,
            run_name="other",
            tracking_uri=_uri(tmp_path),
            artifact_root=tmp_path / "artifacts",
            models_dir=tmp_path / "models",
            reports_dir=tmp_path / "reports",
        )


def test_missing_manifest_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        _import(tmp_path, tmp_path / "empty")


def test_champion_alias_points_at_a_version(tmp_path: Path) -> None:
    report = _import(tmp_path, make_run_dir(tmp_path))
    set_champion_alias(_uri(tmp_path), report.model_version)
    client = mlflow.MlflowClient(_uri(tmp_path))
    assert client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version == "1"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_training_tracking.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.training.tracking'`

- [ ] **Step 4: Implement**

`src/claimlens/training/tracking.py`:

```python
"""Import finished training runs into a local MLflow logbook and model registry.

Requires `uv sync --group training`. The logbook lives in `var/mlflow/` (git-ignored).
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path
from typing import Any

from claimlens.training.manifest import ModelReport, RunManifest, RunMetrics, read_json, write_json

REGISTERED_MODEL = "claimlens-damage"
EXPERIMENT = "claimlens-damage"


class TrainingUnavailableError(RuntimeError):
    pass


def _mlflow() -> Any:
    try:
        import mlflow
    except ImportError:
        raise TrainingUnavailableError(
            "MLflow is not installed: run `uv sync --group training`"
        ) from None
    return mlflow


def default_tracking(repo_root: Path) -> tuple[str, Path]:
    root = repo_root / "var" / "mlflow"
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}", root / "artifacts"


def _metric_key(column: str) -> str:
    """MLflow-safe metric key: `metrics/mAP50(M)` becomes `metrics/mAP50_M`."""
    return re.sub(r"[^\w\-./ ]", "_", column).strip("_")


def _epoch_metrics(results_csv: Path) -> list[tuple[int, dict[str, float]]]:
    with results_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]
    epochs: list[tuple[int, dict[str, float]]] = []
    for row in rows:
        values: dict[str, float] = {}
        for column, value in row.items():
            if column == "epoch":
                continue
            try:
                values[_metric_key(column)] = float(value)
            except ValueError:
                continue
        epochs.append((int(float(row["epoch"])), values))
    return epochs


def _final_metrics(metrics: RunMetrics) -> dict[str, float]:
    final: dict[str, float] = {}
    for split, values in (("val", metrics.val), ("test", metrics.test)):
        for name in ("box_map50", "box_map50_95", "mask_map50", "mask_map50_95"):
            final[f"{split}_{name}"] = getattr(values, name)
        for class_name, ap in values.per_class_mask_map50.items():
            final[f"{split}_mask_map50_{class_name}"] = ap
    return final


def _experiment_id(client: Any, artifact_root: Path) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is not None:
        return str(experiment.experiment_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    return str(
        client.create_experiment(EXPERIMENT, artifact_location=artifact_root.resolve().as_uri())
    )


def import_run(
    run_dir: Path,
    *,
    run_name: str,
    tracking_uri: str,
    artifact_root: Path,
    models_dir: Path,
    reports_dir: Path,
    allow_incomplete: bool = False,
) -> ModelReport:
    manifest = read_json(run_dir / "manifest.json", RunManifest)
    metrics = read_json(run_dir / "metrics.json", RunMetrics)
    if manifest.run != run_name or metrics.run != run_name:
        raise ValueError(f"manifest is for run {manifest.run!r}, not {run_name!r}")
    if manifest.status == "incomplete" and not allow_incomplete:
        raise ValueError(f"run {run_name} is incomplete ({manifest.note}); pass --allow-incomplete")
    weights = run_dir / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"run has no weights: {weights}")

    report_path = reports_dir / f"{run_name}.json"
    mlflow = _mlflow()
    client = mlflow.MlflowClient(tracking_uri)
    experiment_id = _experiment_id(client, artifact_root)
    existing = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.claimlens_run = '{run_name}' and tags.claimlens_commit = '{manifest.commit}'"
        ),
    )
    if existing and report_path.is_file():
        return read_json(report_path, ModelReport)

    run = client.create_run(
        experiment_id,
        run_name=run_name,
        tags={
            "claimlens_run": run_name,
            "claimlens_commit": manifest.commit,
            "claimlens_dataset_md5": manifest.dataset_md5,
            "status": manifest.status,
            "trainer": manifest.trainer_version,
        },
    )
    run_id: str = run.info.run_id
    for key, value in manifest.config.model_dump().items():
        client.log_param(run_id, key, str(value))
    for step, values in _epoch_metrics(run_dir / "train" / "results.csv"):
        for key, value in values.items():
            client.log_metric(run_id, key, value, step=step)
    for key, value in _final_metrics(metrics).items():
        client.log_metric(run_id, key, value)
    client.log_artifact(run_id, str(run_dir / "manifest.json"))
    client.log_artifact(run_id, str(run_dir / "metrics.json"))
    for plot in sorted((run_dir / "train").glob("*.png")):
        client.log_artifact(run_id, str(plot), "plots")
    client.log_artifact(run_id, str(weights), "weights")
    client.set_terminated(run_id)

    if not client.search_registered_models(f"name='{REGISTERED_MODEL}'"):
        client.create_registered_model(REGISTERED_MODEL)
    version = client.create_model_version(
        REGISTERED_MODEL,
        source=f"{run.info.artifact_uri}/weights",
        run_id=run_id,
        tags={"claimlens_run": run_name},
    )

    target = models_dir / run_name / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, target)
    report = ModelReport(
        run=run_name,
        base_model=manifest.config.model,
        commit=manifest.commit,
        dataset=manifest.dataset,
        dataset_md5=manifest.dataset_md5,
        status=manifest.status,
        epochs_run=manifest.epochs_run,
        best_epoch=manifest.best_epoch,
        val=metrics.val,
        test=metrics.test,
        mlflow_run_id=run_id,
        model_version=str(version.version),
    )
    write_json(report_path, report)
    return report


def set_champion_alias(tracking_uri: str, version: str) -> None:
    client = _mlflow().MlflowClient(tracking_uri)
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
```

If `create_model_version` rejects the `file:` source on the installed MLflow version, use
`source=f"runs:/{run_id}/weights"` instead and keep the test unchanged.

- [ ] **Step 5: Add `train import` to the CLI**

In `src/claimlens/training/commands.py`, add these lines to `add_train_parser`:

```python
    imp = train_sub.add_parser("import", help="log a finished run into the local MLflow logbook")
    imp.add_argument("run")
    imp.add_argument("--from", dest="source", type=Path, required=True, help="downloaded run folder")
    imp.add_argument("--allow-incomplete", action="store_true")
```

Add a handler:

```python
def _import(args: argparse.Namespace) -> int:
    from claimlens.training.tracking import default_tracking, import_run

    repo_root = Path.cwd()
    tracking_uri, artifact_root = default_tracking(repo_root)
    report = import_run(
        args.source,
        run_name=args.run,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        models_dir=repo_root / "models" / "damage",
        reports_dir=repo_root / "reports" / "models",
        allow_incomplete=args.allow_incomplete,
    )
    print(
        f"Imported {report.run} as {report.model_version}: "
        f"val mask mAP50 {report.val.mask_map50:.3f}, test {report.test.mask_map50:.3f}"
    )
    return 0
```

In `run_train_command`, dispatch `if args.train_command == "import": return _import(args)`, and add
`TrainingUnavailableError` to the caught exceptions. Import it inside the function so `claimlens`
loads without MLflow, and handle it like this:

```python
    except Exception as exc:
        from claimlens.training.tracking import TrainingUnavailableError

        if isinstance(exc, TrainingUnavailableError):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if isinstance(exc, OSError | ValueError | subprocess.CalledProcessError):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise
```

This replaces the narrower `except` from Task 3. `tracking.py` imports MLflow only inside `_mlflow()`,
so importing the module itself is always safe.

Add a CLI test to `tests/unit/test_training_cli.py`:

```python
def test_train_import_reports_the_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("mlflow")
    from tests.training_helpers import make_run_dir

    run_dir = make_run_dir(tmp_path, name="damage-yolo11n-v1")
    monkeypatch.chdir(tmp_path)
    assert main(["train", "import", "damage-yolo11n-v1", "--from", str(run_dir)]) == 0
    assert "as 1" in capsys.readouterr().out
    assert (tmp_path / "models" / "damage" / "damage-yolo11n-v1" / "best.pt").is_file()
    assert (tmp_path / "reports" / "models" / "damage-yolo11n-v1.json").is_file()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_training_tracking.py tests/unit/test_training_cli.py -q`
Expected: all pass (MLflow creates its SQLite schema on first use; allow a few seconds)

- [ ] **Step 7: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add pyproject.toml uv.lock .github/workflows/ci.yml src/claimlens/training tests/unit/test_training_tracking.py tests/unit/test_training_cli.py
git commit -m "feat: import training runs into a local MLflow logbook and registry"
```

---

### Task 5: Choose the champion and write the model report

**Files:**
- Create: `src/claimlens/training/select.py`
- Modify: `src/claimlens/training/commands.py` (`train select`, `train report`)
- Test: `tests/unit/test_training_select.py`

**Interfaces:**
- Consumes: `ModelReport`, `SplitMetrics`, `read_json` (Task 1); `set_champion_alias`, `default_tracking` (Task 4).
- Produces:
  - `ChampionModel(run: str, weights: str, mlflow_version: str)` and `ModelsConfig(damage: ChampionModel)`.
  - `load_model_reports(reports_dir: Path) -> list[ModelReport]`.
  - `select_champion(reports: Sequence[ModelReport]) -> ModelReport`.
  - `write_models_config(path: Path, champion: ModelReport) -> ModelsConfig`.
  - `load_models_config(path: Path) -> ModelsConfig | None`.
  - `render_model_report(reports: Sequence[ModelReport], champion: ModelReport, generated_on: date) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_training_select.py`:

```python
from datetime import date
from pathlib import Path

import pytest

from claimlens.training.manifest import ModelReport, SplitMetrics, write_json
from claimlens.training.select import (
    load_model_reports,
    load_models_config,
    render_model_report,
    select_champion,
    write_models_config,
)


def _split(mask_map50: float) -> SplitMetrics:
    return SplitMetrics(
        box_map50=0.6,
        box_map50_95=0.4,
        mask_map50=mask_map50,
        mask_map50_95=0.3,
        per_class_mask_map50={"dent": 0.5, "scratch": 0.4},
    )


def _report(run: str, val: float, test: float, status: str = "complete") -> ModelReport:
    return ModelReport(
        run=run,
        base_model=f"{run}.pt",
        commit="c",
        dataset="damage-v1",
        dataset_md5="abc.dir",
        status=status,  # type: ignore[arg-type]
        epochs_run=50,
        best_epoch=40,
        val=_split(val),
        test=_split(test),
        mlflow_run_id="id",
        model_version="1",
        cpu_ms_per_image=120.0,
    )


def test_champion_is_chosen_on_validation_not_test() -> None:
    chosen = select_champion([_report("n", val=0.50, test=0.70), _report("s", val=0.60, test=0.40)])
    assert chosen.run == "s"


def test_incomplete_runs_are_never_chosen() -> None:
    reports = [_report("n", 0.5, 0.5), _report("s", 0.9, 0.9, status="incomplete")]
    assert select_champion(reports).run == "n"


def test_no_complete_run_is_an_error() -> None:
    with pytest.raises(ValueError, match="no complete run"):
        select_champion([_report("s", 0.9, 0.9, status="incomplete")])


def test_models_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    assert load_models_config(path) is None
    written = write_models_config(path, _report("s", 0.6, 0.5))
    assert written.damage.weights == "models/damage/s/best.pt"
    assert load_models_config(path) == written


def test_reports_are_loaded_in_name_order(tmp_path: Path) -> None:
    write_json(tmp_path / "s.json", _report("s", 0.6, 0.5))
    write_json(tmp_path / "n.json", _report("n", 0.5, 0.5))
    assert [r.run for r in load_model_reports(tmp_path)] == ["n", "s"]


def test_report_names_the_champion_and_both_runs() -> None:
    reports = [_report("n", 0.50, 0.48), _report("s", 0.60, 0.55)]
    text = render_model_report(reports, reports[1], date(2026, 10, 2))
    assert "Champion: `s`" in text
    assert "| n |" in text
    assert "| s |" in text
    assert "0.550" in text
    assert "chosen on validation" in text.lower()
    assert "| dent |" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_training_select.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.training.select'`

- [ ] **Step 3: Implement**

`src/claimlens/training/select.py`:

```python
"""Choose the champion damage model on validation and describe every candidate."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from claimlens.domain import Frozen
from claimlens.training.manifest import ModelReport, read_json


class ChampionModel(Frozen):
    run: str
    weights: str
    mlflow_version: str


class ModelsConfig(Frozen):
    damage: ChampionModel


def load_model_reports(reports_dir: Path) -> list[ModelReport]:
    return [read_json(path, ModelReport) for path in sorted(reports_dir.glob("*.json"))]


def select_champion(reports: Sequence[ModelReport]) -> ModelReport:
    """Highest validation mask mAP50 among complete runs. Test scores are never used here."""
    complete = [r for r in reports if r.status == "complete"]
    if not complete:
        raise ValueError("no complete run to choose from; import a finished run first")
    return max(complete, key=lambda r: (r.val.mask_map50, r.val.mask_map50_95, r.run))


def write_models_config(path: Path, champion: ModelReport) -> ModelsConfig:
    config = ModelsConfig(
        damage=ChampionModel(
            run=champion.run,
            weights=f"models/damage/{champion.run}/best.pt",
            mlflow_version=champion.model_version,
        )
    )
    path.write_text(
        "# Written by `claimlens train select`. The pipeline uses this damage model by default.\n"
        "[damage]\n"
        f'run = "{config.damage.run}"\n'
        f'weights = "{config.damage.weights}"\n'
        f'mlflow_version = "{config.damage.mlflow_version}"\n',
        encoding="utf-8",
        newline="\n",
    )
    return config


def load_models_config(path: Path) -> ModelsConfig | None:
    if not path.is_file():
        return None
    return ModelsConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def _cpu(report: ModelReport) -> str:
    return "-" if report.cpu_ms_per_image is None else f"{report.cpu_ms_per_image:.0f}"


def render_model_report(
    reports: Sequence[ModelReport], champion: ModelReport, generated_on: date
) -> str:
    lines = [
        "# Damage model v1: YOLO11-seg on damage-v1",
        "",
        f"- Date: {generated_on.isoformat()}",
        f"- Dataset: `{champion.dataset}` (md5 `{champion.dataset_md5}`)",
        f"- Champion: `{champion.run}` (MLflow version {champion.model_version})",
        "- Rule: chosen on validation mask mAP50; test scores are reported, never used to choose.",
        "",
        "| Run | Base model | Status | Epochs (best) | Val mask mAP50 | Test mask mAP50 "
        "| Test mask mAP50-95 | Test box mAP50 | CPU ms/image |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.run} | {r.base_model} | {r.status} | {r.epochs_run} ({r.best_epoch}) "
            f"| {r.val.mask_map50:.3f} | {r.test.mask_map50:.3f} | {r.test.mask_map50_95:.3f} "
            f"| {r.test.box_map50:.3f} | {_cpu(r)} |"
        )
    target = "met" if champion.test.mask_map50 >= 0.50 else "not met"
    lines += [
        "",
        f"Target test mask mAP50 >= 0.50: **{target}** ({champion.test.mask_map50:.3f}).",
        "",
        f"## Per-class test mask mAP50 (`{champion.run}`)",
        "",
        "| Class | Mask mAP50 |",
        "|---|---|",
    ]
    for name, ap in sorted(champion.test.per_class_mask_map50.items()):
        lines.append(f"| {name} | {ap:.3f} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Add `train select` and `train report` to the CLI**

In `add_train_parser`:

```python
    train_sub.add_parser("select", help="choose the champion on validation mask mAP50")
    report = train_sub.add_parser("report", help="write the damage model report")
    report.add_argument("--out", type=Path, required=True)
```

Handlers in `commands.py`:

```python
def _select(args: argparse.Namespace) -> int:
    from claimlens.training.select import load_model_reports, select_champion, write_models_config
    from claimlens.training.tracking import default_tracking, set_champion_alias

    repo_root = Path.cwd()
    champion = select_champion(load_model_reports(repo_root / "reports" / "models"))
    write_models_config(args.config / "models.toml", champion)
    tracking_uri, _ = default_tracking(repo_root)
    set_champion_alias(tracking_uri, champion.model_version)
    print(f"Champion: {champion.run} (val mask mAP50 {champion.val.mask_map50:.3f})")
    return 0


def _report(args: argparse.Namespace) -> int:
    from datetime import date

    from claimlens.training.select import (
        load_model_reports,
        load_models_config,
        render_model_report,
    )

    reports = load_model_reports(Path.cwd() / "reports" / "models")
    models = load_models_config(args.config / "models.toml")
    if models is None:
        raise ValueError("no champion yet: run `claimlens train select` first")
    champion = next(r for r in reports if r.run == models.damage.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_model_report(reports, champion, date.today()), encoding="utf-8")
    print(f"Report written to {args.out}")
    return 0
```

Dispatch both commands in `run_train_command`. Add a CLI test:

```python
def test_train_select_writes_models_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mlflow")
    import shutil

    from tests.training_helpers import make_run_dir

    config = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config)
    run_dir = make_run_dir(tmp_path, name="damage-yolo11n-v1")
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "--config",
                str(config),
                "train",
                "import",
                "damage-yolo11n-v1",
                "--from",
                str(run_dir),
            ]
        )
        == 0
    )
    assert main(["--config", str(config), "train", "select"]) == 0
    assert 'run = "damage-yolo11n-v1"' in (config / "models.toml").read_text(encoding="utf-8")
    out = tmp_path / "report.md"
    assert main(["--config", str(config), "train", "report", "--out", str(out)]) == 0
    assert "Champion: `damage-yolo11n-v1`" in out.read_text(encoding="utf-8")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_training_select.py tests/unit/test_training_cli.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src/claimlens/training tests/unit/test_training_select.py tests/unit/test_training_cli.py
git commit -m "feat: choose the champion model on validation and render the model report"
```

---

### Task 6: `YoloSegDetector`, mask-based size, `--detector` and CPU benchmark

**Files:**
- Create: `src/claimlens/vision/yolo_seg.py`, `src/claimlens/training/benchmark.py`, `tests/unit/test_yolo_seg_math.py`, `tests/integration/test_yolo_seg.py`
- Modify: `src/claimlens/vision/base.py`, `src/claimlens/cli.py`, `src/claimlens/training/commands.py` (`train benchmark`), `tests/unit/test_cli.py` and `tests/unit/test_eval_triage.py` (factory now takes two arguments), `pyproject.toml` (coverage omit)

**Interfaces:**
- Consumes: `load_models_config` (Task 5); `findings_from_predictions` (existing).
- Produces:
  - `polygon_area_fraction(points_xyn: Sequence[float]) -> float`.
  - `findings_from_predictions(..., area_fractions: Sequence[float] | None = None)`.
  - `YoloSegDetector(weights: Path, run: str, conf_floor: float = 0.10)`.
  - `DetectorFactory = Callable[[str, Path], Detector]` in `cli.py`.
  - `resolve_detector(detector: str | None, weights: Path | None, config_dir: Path) -> tuple[str, Path]`.
  - `benchmark_detector(detector: Detector, images: Sequence[Path]) -> float`.

- [ ] **Step 1: Write the failing unit tests**

`tests/unit/test_yolo_seg_math.py`:

```python
from pathlib import Path

import pytest

from claimlens.cli import resolve_detector
from claimlens.domain import DamageType
from claimlens.training.benchmark import benchmark_detector
from claimlens.vision.base import findings_from_predictions, polygon_area_fraction
from tests.fakes import FakeDetector


def test_unit_square_quarter_has_quarter_area() -> None:
    assert polygon_area_fraction([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]) == pytest.approx(0.25)


def test_orientation_does_not_matter() -> None:
    clockwise = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0]
    assert polygon_area_fraction(clockwise) == pytest.approx(0.25)


def test_degenerate_polygon_has_zero_area() -> None:
    assert polygon_area_fraction([]) == 0.0
    assert polygon_area_fraction([0.1, 0.1, 0.2, 0.2]) == 0.0


def test_area_fraction_is_clamped() -> None:
    assert polygon_area_fraction([-0.1, -0.1, 1.1, -0.1, 1.1, 1.1, -0.1, 1.1]) == 1.0


def test_odd_number_of_coordinates_is_rejected() -> None:
    with pytest.raises(ValueError, match="pairs"):
        polygon_area_fraction([0.1, 0.2, 0.3])


def _findings(area_fractions: list[float] | None) -> list[float]:
    findings = findings_from_predictions(
        photo_id="p1",
        boxes_xyxy=[[0, 0, 100, 100]],
        confidences=[0.9],
        class_ids=[1],
        class_names={1: "dent"},
        image_width=200,
        image_height=200,
        area_fractions=area_fractions,
    )
    assert findings[0].damage_type is DamageType.DENT
    return [f.image_area_fraction for f in findings]


def test_mask_area_replaces_box_area() -> None:
    assert _findings(None) == [pytest.approx(0.25)]
    assert _findings([0.04]) == [pytest.approx(0.04)]


def test_mismatched_area_fractions_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        _findings([0.1, 0.2])


def test_no_predictions_give_no_findings() -> None:
    assert (
        findings_from_predictions(
            photo_id="p1",
            boxes_xyxy=[],
            confidences=[],
            class_ids=[],
            class_names={0: "dent"},
            image_width=10,
            image_height=10,
            area_fractions=[],
        )
        == []
    )


def test_resolve_prefers_explicit_arguments(tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    assert resolve_detector("legacy", weights, tmp_path) == ("legacy", weights)


def test_resolve_defaults_to_legacy_without_a_champion(tmp_path: Path) -> None:
    kind, weights = resolve_detector(None, None, tmp_path)
    assert kind == "legacy"
    assert weights.name == "yolov8n-cardamage-v6.pt"


def test_resolve_uses_the_champion(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(
        '[damage]\nrun = "s"\nweights = "models/damage/s/best.pt"\nmlflow_version = "2"\n',
        encoding="utf-8",
    )
    assert resolve_detector(None, None, tmp_path) == ("yolo-seg", Path("models/damage/s/best.pt"))


def test_resolve_yolo_without_champion_or_weights_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train select"):
        resolve_detector("yolo-seg", None, tmp_path)


def test_benchmark_returns_median_milliseconds(tmp_path: Path) -> None:
    images = [tmp_path / f"{i}.jpg" for i in range(3)]
    detector = FakeDetector()
    assert benchmark_detector(detector, images) >= 0.0
    assert len(detector.calls) == 4  # one warm-up call plus one per image


def test_benchmark_needs_images() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        benchmark_detector(FakeDetector(), [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_yolo_seg_math.py -q`
Expected: FAIL with `ImportError: cannot import name 'resolve_detector'` (or `polygon_area_fraction`)

- [ ] **Step 3: Implement the area maths in `vision/base.py`**

Add to `src/claimlens/vision/base.py`:

```python
def polygon_area_fraction(points_xyn: Sequence[float]) -> float:
    """Area of a polygon in normalised [0, 1] coordinates (shoelace formula), clamped to [0, 1]."""
    if len(points_xyn) % 2:
        raise ValueError("polygon coordinates must come in x, y pairs")
    xs, ys = points_xyn[0::2], points_xyn[1::2]
    if len(xs) < 3:
        return 0.0
    twice_area = sum(
        xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs))
    )
    return min(abs(twice_area) / 2.0, 1.0)
```

Change `findings_from_predictions` to accept `area_fractions: Sequence[float] | None = None` as its
last keyword argument. Extend the length check:

```python
    if area_fractions is not None and len(area_fractions) != len(boxes_xyxy):
        raise ValueError("area fractions must have the same length as the boxes")
```

and inside the loop use the mask area when given. Iterate with `enumerate`, so the loop header
becomes `for index, (box, confidence, class_id) in enumerate(zip(boxes_xyxy, confidences, class_ids, strict=True)):`.
Then:

```python
        box_fraction = min(area / (width * height), 1.0)
        fraction = box_fraction if area_fractions is None else area_fractions[index]
```

Pass `image_area_fraction=min(max(fraction, 0.0), 1.0)`.

- [ ] **Step 4: Implement the detector, the benchmark and the resolver**

`src/claimlens/vision/yolo_seg.py`:

```python
"""Our YOLO11-seg damage model. Size comes from the mask outline, not the box.

Requires `uv sync --group vision`. Excluded from coverage; see tests/integration/test_yolo_seg.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claimlens.domain import DamageFinding
from claimlens.vision.base import findings_from_predictions, polygon_area_fraction


class YoloSegDetector:
    def __init__(self, weights: Path, run: str, conf_floor: float = 0.10) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        from ultralytics import YOLO

        self._model: Any = YOLO(str(weights))
        self._conf_floor = conf_floor
        self.model_version = f"yolo11-seg:{run}"

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        result = self._model.predict(source=str(image_path), conf=self._conf_floor, verbose=False)[
            0
        ]
        height, width = result.orig_shape
        boxes = result.boxes
        fractions = (
            None
            if result.masks is None
            else [polygon_area_fraction(p.reshape(-1).tolist()) for p in result.masks.xyn]
        )
        return findings_from_predictions(
            photo_id=photo_id,
            boxes_xyxy=boxes.xyxy.tolist(),
            confidences=boxes.conf.tolist(),
            class_ids=[int(c) for c in boxes.cls.tolist()],
            class_names=result.names,
            image_width=int(width),
            image_height=int(height),
            area_fractions=fractions,
        )
```

`src/claimlens/training/benchmark.py`:

```python
"""Median CPU time per image for a detector, after one warm-up call."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from claimlens.vision.base import Detector


def benchmark_detector(detector: Detector, images: Sequence[Path]) -> float:
    if not images:
        raise ValueError("benchmark needs at least one image")
    detector.detect(images[0], "warm-up")
    timings: list[float] = []
    for index, image in enumerate(images):
        start = time.perf_counter()
        detector.detect(image, f"bench-{index}")
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)
```

In `src/claimlens/cli.py`:

1. Replace `DetectorFactory = Callable[[Path], Detector]` with
   `DetectorFactory = Callable[[str, Path], Detector]`, and add `from claimlens.training.select import load_models_config`.
2. Replace `_legacy_detector` with:

```python
def _default_detector(kind: str, weights: Path) -> Detector:
    if kind == "yolo-seg":
        from claimlens.vision.yolo_seg import YoloSegDetector

        return YoloSegDetector(weights, run=weights.parent.name)
    from claimlens.vision.legacy_yolo import LegacyYoloDetector

    return LegacyYoloDetector(weights)


def resolve_detector(
    detector: str | None, weights: Path | None, config_dir: Path
) -> tuple[str, Path]:
    """Explicit flags win; otherwise the champion in config/models.toml; otherwise legacy."""
    models = load_models_config(config_dir / "models.toml")
    kind = detector or ("yolo-seg" if models is not None else "legacy")
    if weights is not None:
        return kind, weights
    if kind == "legacy":
        return kind, DEFAULT_WEIGHTS
    if models is None:
        raise ValueError("no champion model: run `claimlens train select` or pass --weights")
    return kind, Path(models.damage.weights)
```

3. In `build_parser`, change `--weights` to `parser.add_argument("--weights", type=Path, default=None)`.
   Then add `parser.add_argument("--detector", choices=["legacy", "yolo-seg"], default=None)`.
4. Set the `main` default to `detector_factory: DetectorFactory = _default_detector`. Pass
   `detector_factory=detector_factory` directly to `run_train_command`, which replaces the Task 3 lambda.
5. Add a helper and use it in `_run`, `_resume` and `_eval_triage` in place of `factory(args.weights)`:

```python
def _make_detector(args: argparse.Namespace, factory: DetectorFactory) -> Detector:
    kind, weights = resolve_detector(args.detector, args.weights, args.config)
    return factory(kind, weights)
```

In `tests/unit/test_cli.py` and `tests/unit/test_eval_triage.py`, change every
`detector_factory=lambda _: chosen` (or similar one-argument lambda) to `lambda *_: chosen`.

In `commands.py`, add `train benchmark`:

```python
    bench = train_sub.add_parser("benchmark", help="median CPU ms per image on test photos")
    bench.add_argument("run")
    bench.add_argument("--images", type=int, default=20)
```

```python
def _benchmark(args: argparse.Namespace, detector_factory: DetectorFactory) -> int:
    from claimlens.training.benchmark import benchmark_detector
    from claimlens.training.manifest import ModelReport, read_json, write_json

    repo_root = Path.cwd()
    report_path = repo_root / "reports" / "models" / f"{args.run}.json"
    report = read_json(report_path, ModelReport)
    test_dir = repo_root / "data" / "processed" / report.dataset / "images" / "test"
    images = sorted(test_dir.glob("*.jpg"))[: args.images]
    detector = detector_factory("yolo-seg", repo_root / "models" / "damage" / args.run / "best.pt")
    ms = benchmark_detector(detector, images)
    write_json(report_path, report.model_copy(update={"cpu_ms_per_image": round(ms, 1)}))
    print(f"{args.run}: median {ms:.0f} ms per image on CPU over {len(images)} images")
    return 0
```

and dispatch it with `detector_factory`.

In `pyproject.toml`, add `"*/claimlens/vision/yolo_seg.py",` to the coverage omit list.

`tests/integration/test_yolo_seg.py`:

```python
import importlib.util
from pathlib import Path

import pytest

from claimlens.cli import resolve_detector
from claimlens.domain import DamageType

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"


def _champion() -> Path | None:
    try:
        kind, weights = resolve_detector("yolo-seg", None, ROOT / "config")
    except ValueError:
        return None
    path = ROOT / weights
    return path if path.is_file() else None


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _champion() is None or importlib.util.find_spec("ultralytics") is None,
        reason="needs a champion in config/models.toml, its weights and `uv sync --group vision`",
    ),
]


def test_yolo_seg_returns_findings_with_mask_areas() -> None:
    from claimlens.vision.yolo_seg import YoloSegDetector

    weights = _champion()
    assert weights is not None
    detector = YoloSegDetector(weights, run=weights.parent.name)
    findings = detector.detect(SAMPLE, "p1")
    assert detector.model_version.startswith("yolo11-seg:")
    assert all(isinstance(f.damage_type, DamageType) for f in findings)
    assert all(0.0 <= f.image_area_fraction <= 1.0 for f in findings)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass (the integration test is skipped)

- [ ] **Step 6: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src tests pyproject.toml
git commit -m "feat: add YOLO11-seg detector with mask-based size and --detector switch"
```

---

### Task 7: What-if thresholds in the golden evaluation

**Files:**
- Modify: `src/claimlens/evals/triage.py`, `src/claimlens/cli.py` (`--what-if`)
- Test: `tests/unit/test_eval_triage.py`

**Interfaces:**
- Consumes: `decide`, `DecisionConfig` (existing); `compute_triage_metrics` (existing).
- Produces:
  - `CaseResult.state: ClaimState | None = None`.
  - `what_if_thresholds(results: Sequence[CaseResult], config: DecisionConfig, thresholds: Sequence[float]) -> list[tuple[float, TriageMetrics]]`.
  - `render_report(..., what_if: Sequence[tuple[float, TriageMetrics]] = ())`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_eval_triage.py` (reuse that file's existing helpers for building cases
and deps; the new tests below build `CaseResult`s directly):

```python
from claimlens.decision import DecisionConfig
from claimlens.domain import Route
from claimlens.evals.metrics import compute_triage_metrics
from claimlens.evals.triage import CaseResult, what_if_thresholds


def _config(threshold: float = 0.40) -> DecisionConfig:
    return DecisionConfig(
        version="t",
        fraud_score_threshold=0.8,
        max_fast_track_cost_usd=3000,
        min_finding_confidence=threshold,
    )


def test_what_if_counts_errored_cases_as_errors() -> None:
    errored = CaseResult(
        case_id="g1", scenario="s", expected=Route.FAST_TRACK, predicted=None, error="boom"
    )
    ((threshold, metrics),) = what_if_thresholds([errored], _config(), [0.25])
    assert threshold == 0.25
    assert metrics == compute_triage_metrics([(Route.FAST_TRACK, None)])
```

Add one more test that runs a real fake-detector case through `run_triage_eval`. Use the file's
existing fixture that builds a golden case with `FakeDetector(findings=…)` and a `dent(confidence=0.30)`.
Then assert:

```python
    results = run_triage_eval(...)  # as in the existing tests, with a single 0.30-confidence dent
    assert results[0].state is not None
    (low, strict) = what_if_thresholds(results, _config(), [0.25, 0.55])
    assert low[1].confusion.get((Route.FAST_TRACK, Route.FAST_TRACK), 0) == 1
    assert strict[1].confusion.get((Route.FAST_TRACK, Route.ADJUSTER_REVIEW), 0) == 1
```

Also assert that `render_report(..., what_if=[(0.25, low[1])])` contains `"## What if"` and `"| 0.25 |"`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_eval_triage.py -q`
Expected: FAIL with `ImportError: cannot import name 'what_if_thresholds'`

- [ ] **Step 3: Implement**

In `src/claimlens/evals/triage.py`:

- Import `from claimlens.decision import DecisionConfig, decide` and
  `from claimlens.events.projection import ClaimState, fold`.
- Add `state: ClaimState | None = None` as the last field of `CaseResult`.
- In `run_case`, pass `state=fold(deps.store.load(claim_id))` to the returned `CaseResult`.
- Add:

```python
def what_if_thresholds(
    results: Sequence[CaseResult], config: DecisionConfig, thresholds: Sequence[float]
) -> list[tuple[float, TriageMetrics]]:
    """Re-decide each finished claim at other confidence thresholds. Agent output is held fixed."""
    table: list[tuple[float, TriageMetrics]] = []
    for threshold in thresholds:
        variant = config.model_copy(update={"min_finding_confidence": threshold})
        pairs = [
            (r.expected, decide(r.state, variant).route if r.state is not None else None)
            for r in results
        ]
        table.append((threshold, compute_triage_metrics(pairs)))
    return table
```

  Import `compute_triage_metrics` from `claimlens.evals.metrics`.
- Give `render_report` a final parameter `what_if: Sequence[tuple[float, TriageMetrics]] = ()`.
  Before the final `return`, add:

```python
    if what_if:
        lines += [
            "",
            "## What if: other confidence thresholds (R6)",
            "",
            "Policy unchanged; each claim is re-decided from its final state.",
            "",
            "| Min confidence | Route accuracy | Escalation recall | Correct fast-tracks |",
            "|---|---|---|---|",
        ]
        for threshold, m in what_if:
            rec = "n/a" if m.escalation_recall is None else f"{m.escalation_recall:.2f}"
            fast = m.confusion.get((Route.FAST_TRACK, Route.FAST_TRACK), 0)
            expected_fast = sum(v for (e, _), v in m.confusion.items() if e is Route.FAST_TRACK)
            lines.append(
                f"| {threshold:.2f} | {m.route_accuracy:.2f} | {rec} | {fast} of {expected_fast} |"
            )
```

In `cli.py`, add `evaluate.add_argument("--what-if", default="", help="comma-separated thresholds, e.g. 0.25,0.40,0.55")`.
Then in `_eval_triage`:

```python
    thresholds = [float(t) for t in args.what_if.split(",") if t.strip()]
    decision_config = load_decision_config(args.config / "decision_policy.toml")
    what_if = what_if_thresholds(results, decision_config, thresholds)
```

and pass `what_if=what_if` to `render_report`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_triage.py tests/unit/test_cli.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy
git add src/claimlens/evals/triage.py src/claimlens/cli.py tests/unit/test_eval_triage.py
git commit -m "feat: add what-if confidence thresholds to the golden evaluation"
```

---

### Task 8: Kaggle dataset bundle and training job

**Files:**
- Create: `scripts/make_train_bundle.py`, `training/kaggle/train/run_train.py`, `training/kaggle/train/kernel-metadata.json`
- Modify: `.gitignore` (allow `models/*.dvc`), `models/README.md`, `docs/specs/2026-10-02-m3a-damage-model-design.md` (flag name `--what-if`)

**Interfaces:**
- Consumes: `dvc_out_md5` (Task 1); `claimlens train run` (Task 3).
- Produces:
  - `var/kaggle/train-bundle/claimlens-damage-v1.zip`, containing `dataset.json` and `data/processed/damage-v1/**`.
  - `var/kaggle/train-bundle/dataset-metadata.json`.
  - A Kaggle kernel that writes `/kaggle/working/<run>/` for each run, plus `commit.txt`.

- [ ] **Step 1: Write the bundle script**

`scripts/make_train_bundle.py`:

```python
"""Pack damage-v1 for the Kaggle training job (private dataset `claimlens-damage-v1`).

The zip holds the YOLO export at the path the job expects, plus `dataset.json` with the md5 DVC
recorded for it, so the job can refuse a stale bundle. Run from the repository root:

uv run python scripts/make_train_bundle.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from claimlens.training.config import dvc_out_md5

DATASET = "damage-v1"
SOURCE = Path("data/processed") / DATASET
OUT_DIR = Path("var/kaggle/train-bundle")
KAGGLE_ID = "karthickbalaje/claimlens-damage-v1"


def main() -> int:
    md5 = dvc_out_md5(Path("dvc.lock"), SOURCE.as_posix())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUT_DIR / f"claimlens-{DATASET}.zip"
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file())
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("dataset.json", json.dumps({"dataset": DATASET, "md5": md5}))
        for path in files:
            archive.write(path, path.as_posix())
    metadata = {
        "title": f"claimlens-{DATASET}",
        "id": KAGGLE_ID,
        "licenses": [{"name": "other"}],
    }
    (OUT_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    size_mb = archive_path.stat().st_size / 1_000_000
    print(f"Wrote {archive_path} with {len(files)} files ({size_mb:.0f} MB), md5 {md5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the Kaggle job**

`training/kaggle/train/run_train.py`:

```python
"""Kaggle job: train the M3a damage models on a free Kaggle GPU.

Pushed with `kaggle kernels push -p .` from this folder. It reads the private `claimlens-damage-v1`
dataset, checks out the pinned commit, installs the locked `vision` group with uv, and runs
`claimlens train run` for each run. Results go straight to /kaggle/working/<run>/ so they survive
a time-out, and are downloaded with `kaggle kernels output`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = "https://github.com/kb2326/claimlens.git"
# The exact commit this job runs; set in Task 9 to the pushed commit that contains the training code.
COMMIT = "SET-IN-TASK-9"
RUNS = ("damage-yolo11n-v1", "damage-yolo11s-v1")
WORK = Path("/tmp/claimlens")
OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def bundle_root() -> Path:
    """Kaggle may keep the uploaded zip or extract it; return the folder holding dataset.json."""
    archives = sorted(INPUT.rglob("claimlens-damage-v1.zip"))
    if archives:
        target = Path("/tmp/bundle")
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(target)
        return target
    manifests = sorted(INPUT.rglob("dataset.json"))
    if not manifests:
        raise SystemExit("dataset.json not found under /kaggle/input")
    return manifests[0].parent


def main() -> None:
    if COMMIT == "SET-IN-TASK-9":
        raise SystemExit("set COMMIT to the pushed commit before pushing this kernel")
    run("nvidia-smi")
    run("git", "clone", "--filter=blob:none", REPO, str(WORK))
    run("git", "checkout", "--detach", COMMIT, cwd=WORK)
    root = bundle_root()
    shutil.copytree(root / "data", WORK / "data", dirs_exist_ok=True)
    run(sys.executable, "-m", "pip", "install", "-q", "uv")
    run("uv", "sync", "--locked", "--no-dev", "--group", "vision", cwd=WORK)
    (OUT / "commit.txt").write_text(f"{COMMIT}\n", encoding="utf-8")
    failed = []
    for name in RUNS:
        try:
            run(
                "uv",
                "run",
                "--no-sync",
                "claimlens",
                "train",
                "run",
                name,
                "--dataset-dir",
                "data/processed/damage-v1",
                "--bundle",
                str(root / "dataset.json"),
                "--out",
                str(OUT),
                "--commit",
                COMMIT,
                cwd=WORK,
            )
        except subprocess.CalledProcessError as exc:
            print(f"run {name} failed: {exc}", flush=True)
            failed.append(name)
    print(f"Done. Failed runs: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
```

(Ruff format may re-wrap the argument list one item per line. Accept its output.)

`training/kaggle/train/kernel-metadata.json`:

```json
{
  "id": "karthickbalaje/claimlens-train-damage-v1",
  "title": "claimlens-train-damage-v1",
  "code_file": "run_train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["karthickbalaje/claimlens-damage-v1"],
  "competition_sources": [],
  "kernel_sources": []
}
```

- [ ] **Step 3: Allow DVC pointers for models; update docs**

In `.gitignore`, under `/models/*` and `!/models/README.md`, add `!/models/*.dvc`.

Replace the first paragraph of `models/README.md` with:

```markdown
Model weights are stored here locally and are **not committed to git**. Trained weights are
tracked with DVC (`models/damage.dvc`, local remote) and registered in a local MLflow logbook
(`var/mlflow`). Because they are trained on CarDD (non-commercial terms, ADR 0004), they are
**private**: never published to Hugging Face, a public Kaggle item, or git.
```

Then add a row to its table:
`| damage/<run>/best.pt | M3a YOLO11-seg damage models; the champion is named in config/models.toml. |`.

In the spec, §5.5, replace "`eval-triage` gains `--min-confidence` for the what-if table only" with
"`eval-triage` gains `--what-if 0.25,0.40,0.55` (thresholds for the what-if table only)".

- [ ] **Step 4: Check the bundle builds**

Run: `uv run python scripts/make_train_bundle.py`
Expected: `Wrote var/kaggle/train-bundle/claimlens-damage-v1.zip with 8001 files (… MB), md5 cc56ac5162bfc4fdad163862267fedbd.dir`

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q`
Expected: all clean, all pass

- [ ] **Step 5: Commit**

```bash
git add scripts/make_train_bundle.py training/kaggle/train .gitignore models/README.md docs/specs/2026-10-02-m3a-damage-model-design.md
git commit -m "feat: add Kaggle training bundle and job"
```

---

### Task 9: Train on Kaggle, import, select, benchmark

This task runs real infrastructure. Everything heavy runs on Kaggle; the laptop only uploads,
waits, downloads and benchmarks 20 images.

**Files:**
- Modify: `training/kaggle/train/run_train.py` (`COMMIT`), `config/models.toml` (created by `train select`), `reports/models/*.json`, `models/damage.dvc`

- [ ] **Step 1: Push the branch and pin the commit**

```bash
git -c credential.helper= -c credential.helper='!env -u GITHUB_TOKEN gh auth git-credential' push -u origin feat/m3a-damage-model
git rev-parse HEAD
```

Set `COMMIT` in `training/kaggle/train/run_train.py` to that full SHA. The kernel script does not
need to be in the pinned commit, because Kaggle uploads it directly.

- [ ] **Step 2: Upload the private dataset**

```bash
export KAGGLE_API_TOKEN="$(grep -E '^KAGGLE_API_TOKEN=' .env | cut -d= -f2-)"
cd var/kaggle/train-bundle && uvx --from kaggle kaggle datasets create -p . && cd -
uvx --from kaggle kaggle datasets status karthickbalaje/claimlens-damage-v1
```

Expected: the status reaches `ready`. Do not pass `--public`; Kaggle datasets are private by default.

- [ ] **Step 3: Push the kernel and watch it**

```bash
cd training/kaggle/train && uvx --from kaggle kaggle kernels push -p . && cd -
uvx --from kaggle kaggle kernels status karthickbalaje/claimlens-train-damage-v1
```

Poll every few minutes in the background until it reports `COMPLETE` or `ERROR`. Expect about
1 hour for n and 2 hours for s on a T4. On `ERROR`, download the log
(`kaggle kernels output … -p var/kaggle/output/train`) and read the last lines before changing
anything.

- [ ] **Step 4: Download, import, select, benchmark, report**

```bash
mkdir -p var/kaggle/output/train && cd var/kaggle/output/train && uvx --from kaggle kaggle kernels output karthickbalaje/claimlens-train-damage-v1 -p . && cd -
cat var/kaggle/output/train/commit.txt
uv sync --group training --group vision --group review
uv run claimlens train import damage-yolo11n-v1 --from var/kaggle/output/train/damage-yolo11n-v1
uv run claimlens train import damage-yolo11s-v1 --from var/kaggle/output/train/damage-yolo11s-v1
uv run claimlens train select
uv run claimlens train benchmark damage-yolo11n-v1
uv run claimlens train benchmark damage-yolo11s-v1
uv run claimlens train report --out evals/reports/<today>-damage-model-v1.md
```

Expected: two imports (versions 1 and 2), a champion printed, two CPU timings, and a report written.
If a run is `incomplete`, import it only with `--allow-incomplete`, and only if its numbers are
usable; say so in the report notes.

- [ ] **Step 5: Version the weights with DVC**

```bash
uv run dvc add models/damage
uv run dvc push
uv run pytest tests/integration/test_yolo_seg.py -q
```

Expected: `models/damage.dvc` is created; the integration test passes (it is no longer skipped).

- [ ] **Step 6: Commit**

```bash
git add training/kaggle/train/run_train.py config/models.toml reports/models models/damage.dvc models/.gitignore evals/reports
git commit -m "feat: train YOLO11n/s-seg damage models on Kaggle and select the champion"
```

(`dvc add` may write `models/.gitignore`. Commit it if so.)

---

### Task 10: Golden baseline with the champion, ADR, retro and docs

**Files:**
- Create: `evals/reports/<today>-triage-baseline-v1-yolo11.md`, `docs/adr/0008-training-and-experiment-tracking.md`, `docs/retros/m3a-damage-model.md`
- Modify: `docs/roadmap.md`, `README.md` (current status line, if it has one)

- [ ] **Step 1: Run the golden evaluation with the champion**

```bash
uv run claimlens eval-triage --golden evals/golden/v1/claims.jsonl --report evals/reports/<today>-triage-baseline-v1-yolo11.md --what-if 0.25,0.40,0.55
```

Expected: a line `Cases: 97  Route accuracy: …  Escalation recall: …`, using detector
`yolo11-seg:<champion>`.

**Gate:** if escalation recall is below 1.00, stop here. List the cases that were fast-tracked but
needed a person (from the report's mismatch table) and report them to the human partner. Do not
change the policy or the rate card to make the number pass.

- [ ] **Step 2: Write ADR 0008**

`docs/adr/0008-training-and-experiment-tracking.md`: status Accepted, today's date. Cover:
- **Context:** there is no local GPU and the laptop overheats.
- **Decision:**
  - Train on private Kaggle kernels pinned to a commit, with a bundle md5 check.
  - Use a local MLflow logbook (SQLite) instead of a hosted service.
  - Choose the champion on validation only.
  - Keep weights private in DVC and MLflow.
  - Measure size from the mask polygon.
- **Consequences:**
  - Runs appear in the logbook only after import.
  - Weights have one local copy plus the Kaggle output, until an off-site DVC remote exists.
  - The test split is used once per model.

- [ ] **Step 3: Write the retro**

`docs/retros/m3a-damage-model.md`, in the same format as `docs/retros/m2-data-engine.md`:
- what shipped;
- a numbers table with n vs s (val and test mask mAP50, CPU ms) and the golden comparison (legacy
  v1 0.73 / 1.00 / 4 of 30 vs the champion), taken from the two reports;
- what went well; what was harder; what changes in M3b.

Use only numbers that appear in the reports.

- [ ] **Step 4: Update the roadmap**

In `docs/roadmap.md`, under M3, tick "Damage segmentation model (YOLO11-seg) trained on Colab" and
reword it to "on Kaggle", tick "MLflow experiment tracking and model registry", and add M3a story
material to the LinkedIn table.

- [ ] **Step 5: Final checks and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
git add evals/reports docs README.md
git commit -m "docs: add M3a baseline with the champion model, ADR 0008 and retro"
```

---

## Self-review notes

- **Spec coverage:**
  - §2 targets and gate: Tasks 9–10.
  - §3 decisions: Global Constraints, plus Tasks 4, 5 and 8.
  - §5.1: Task 1. §5.2: Tasks 1, 2, 4, 5 and 6 (benchmark). §5.3: Task 8. §5.4: Task 6. §5.5: Tasks 3–7. §5.6: Task 9.
  - §6 reports: Tasks 9–10. §7 errors: Tasks 2–6. §8 tests: every task.
- **Deviations from the spec:**
  - The what-if flag is named `--what-if`; the spec is updated in Task 8.
  - The Kaggle job writes runs straight to `/kaggle/working`, so they survive a time-out.
  - The Ultralytics `time` argument is not used, because it overrides `epochs`.
