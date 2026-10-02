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


def test_train_select_writes_models_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mlflow")
    import shutil

    from tests.training_helpers import make_run_dir

    config = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config)
    (config / "models.toml").unlink(missing_ok=True)
    run_dir = make_run_dir(tmp_path, name="damage-yolo11n-v1")
    monkeypatch.chdir(tmp_path)
    base = ["--config", str(config), "train"]
    assert main([*base, "import", "damage-yolo11n-v1", "--from", str(run_dir)]) == 0
    assert main([*base, "select"]) == 0
    assert 'run = "damage-yolo11n-v1"' in (config / "models.toml").read_text(encoding="utf-8")
    out = tmp_path / "report.md"
    assert main([*base, "report", "--out", str(out)]) == 0
    assert "Champion: `damage-yolo11n-v1`" in out.read_text(encoding="utf-8")


def test_train_report_without_a_champion_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    code = main(["--config", str(tmp_path / "config"), "train", "report", "--out", "r.md"])
    assert code == 1
    assert "train select" in capsys.readouterr().err
