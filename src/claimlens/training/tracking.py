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


def registered_model(task: str) -> str:
    return f"claimlens-{task}"


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


def _log_run(
    client: Any,
    run_id: str,
    run_dir: Path,
    manifest: RunManifest,
    metrics: RunMetrics,
    weights: Path,
) -> None:
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
    if existing:
        run = existing[0]
        run_id: str = run.info.run_id
    else:
        run = client.create_run(
            experiment_id,
            run_name=run_name,
            tags={
                "claimlens_run": run_name,
                "claimlens_commit": manifest.commit,
                "claimlens_dataset_md5": manifest.dataset_md5,
                "status": manifest.status,
                "trainer": manifest.trainer_version,
                "claimlens_task": manifest.config.task,
            },
        )
        run_id = run.info.run_id
        _log_run(client, run_id, run_dir, manifest, metrics, weights)

    name = registered_model(manifest.config.task)
    if not client.search_registered_models(f"name='{name}'"):
        client.create_registered_model(name)
    versions = client.search_model_versions(f"run_id = '{run_id}'")
    version = (
        versions[0]
        if versions
        else client.create_model_version(
            name,
            source=f"{run.info.artifact_uri}/weights",
            run_id=run_id,
            tags={"claimlens_run": run_name},
        )
    )

    target = models_dir / run_name / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, target)
    report = ModelReport(
        run=run_name,
        task=manifest.config.task,
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


def set_champion_alias(tracking_uri: str, version: str, task: str = "damage") -> None:
    mlflow = _mlflow()
    client = mlflow.MlflowClient(tracking_uri)
    try:
        client.set_registered_model_alias(registered_model(task), "champion", version)
    except mlflow.exceptions.MlflowException as exc:
        raise ValueError(
            f"model version {version} is not in this MLflow logbook; "
            "run `claimlens train import` for the downloaded runs first"
        ) from exc
