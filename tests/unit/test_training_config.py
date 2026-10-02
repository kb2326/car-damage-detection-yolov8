from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.training.config import TrainingRun, dvc_out_md5, load_training_runs
from claimlens.training.manifest import RunManifest, SplitMetrics, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


def test_repo_training_config_loads_both_runs() -> None:
    runs = load_training_runs(ROOT / "config" / "training.toml")
    assert {"damage-yolo11n-v1", "damage-yolo11s-v1"} <= set(runs)
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


def test_parts_run_is_a_parts_task() -> None:
    runs = load_training_runs(ROOT / "config" / "training.toml")
    parts = runs["parts-yolo11n-v1"]
    assert parts.task == "parts"
    assert parts.dataset == "parts-v1"
    assert runs["damage-yolo11n-v1"].task == "damage"
