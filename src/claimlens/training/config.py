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
