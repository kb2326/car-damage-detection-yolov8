"""Choose the champion model per task on validation and describe every candidate."""

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


def select_champion(reports: Sequence[ModelReport]) -> ModelReport:
    """Highest validation mask mAP50 among complete runs. Test scores are never used here."""
    complete = [r for r in reports if r.status == "complete"]
    if not complete:
        raise ValueError("no complete run to choose from; import a finished run first")
    return max(complete, key=lambda r: (r.val.mask_map50, r.val.mask_map50_95, r.run))


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
    """Set one task's champion; the other task's section is kept."""
    current = load_models_config(path) or ModelsConfig()
    config = current.model_copy(update={task: champion})
    lines = ["# Written by `claimlens train select` and `train calibrate`. The pipeline defaults."]
    for name in ("damage", "parts"):
        section: ChampionModel | None = getattr(config, name)
        if section is not None:
            lines += ["", *_section(name, section)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
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
        f"# {'Damage' if champion.task == 'damage' else 'Part'} model v1: "
        f"YOLO11-seg on {champion.dataset}",
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
