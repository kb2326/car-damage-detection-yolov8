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
    task: Literal["damage", "parts"] = "damage"
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
    cpu_ms_per_image_onnx: float | None = None


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")


def read_json[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text(encoding="utf-8"))
