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
