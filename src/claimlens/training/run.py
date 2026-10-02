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
