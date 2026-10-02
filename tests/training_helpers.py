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


def make_run(name: str = "r1", model: str = "yolo11n-seg.pt", task: str = "damage") -> TrainingRun:
    return TrainingRun(
        name=name,
        task=task,
        model=model,
        dataset="damage-v1",
        epochs=3,
        patience=0,
        imgsz=64,
        batch=1,
        seed=0,
        workers=0,
    )


def make_run_dir(
    tmp_path: Path, *, name: str = "r1", fail: bool = False, task: str = "damage"
) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "data.yaml").write_text("names: {0: dent}\n", encoding="utf-8")
    run_training(
        make_run(name, task=task),
        dataset_dir=dataset,
        bundle_md5="abc.dir",
        expected_md5="abc.dir",
        commit="0123abc",
        out_dir=tmp_path / "runs",
        trainer=FakeTrainer(fail_after_weights=fail),
    )
    return tmp_path / "runs" / name
