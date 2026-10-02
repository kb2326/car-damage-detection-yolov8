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
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL, "champion").version) == "1"


def test_parts_runs_register_a_parts_model(tmp_path: Path) -> None:
    from claimlens.training.tracking import registered_model

    run_dir = make_run_dir(tmp_path, name="p1", task="parts")
    report = _import(tmp_path, run_dir)
    assert report.task == "parts"
    client = mlflow.MlflowClient(_uri(tmp_path))
    assert client.search_model_versions(f"name='{registered_model('parts')}'")


def test_reimport_after_the_report_is_lost_reuses_the_run(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    first = _import(tmp_path, run_dir)
    (tmp_path / "reports" / "r1.json").unlink()
    second = _import(tmp_path, run_dir)
    assert second.model_version == first.model_version
    assert second.mlflow_run_id == first.mlflow_run_id
    client = mlflow.MlflowClient(_uri(tmp_path))
    assert len(client.search_model_versions(f"name='{REGISTERED_MODEL}'")) == 1


def test_reimport_keeps_benchmark_timings(tmp_path: Path) -> None:
    from claimlens.training.manifest import write_json

    run_dir = make_run_dir(tmp_path)
    first = _import(tmp_path, run_dir)
    write_json(
        tmp_path / "reports" / "r1.json", first.model_copy(update={"cpu_ms_per_image": 99.0})
    )
    assert _import(tmp_path, run_dir).cpu_ms_per_image == 99.0


def test_alias_on_an_empty_logbook_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train import"):
        set_champion_alias(_uri(tmp_path), "1")
