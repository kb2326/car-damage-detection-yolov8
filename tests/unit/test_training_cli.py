import json
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.training.manifest import RunManifest, read_json
from claimlens.vision.instances import SegInstance, Segmentation
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
    text = (config / "models.toml").read_text(encoding="utf-8")
    assert "[damage]" in text
    assert 'run = "damage-yolo11n-v1"' in text
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


SQUARE = (0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4)


class _CalibrationSegmenter:
    model_version = "fake"

    def segment(self, image_path: Path) -> Segmentation:
        hit = SegInstance(label="dent", confidence=0.9, box_xyxy=(0, 0, 1, 1), polygon_xyn=SQUARE)
        miss = SegInstance(
            label="dent",
            confidence=0.6,
            box_xyxy=(0, 0, 1, 1),
            polygon_xyn=(0.6, 0.6, 0.9, 0.6, 0.9, 0.9),
        )
        return Segmentation(instances=(hit, miss), width=10, height=10)


def test_train_calibrate_writes_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from claimlens.training.manifest import write_json
    from claimlens.training.select import (
        champion_from_report,
        load_models_config,
        update_models_config,
    )
    from tests.unit.test_training_select import _report

    config = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config)
    (config / "models.toml").unlink(missing_ok=True)
    write_json(tmp_path / "reports" / "models" / "d1.json", _report("d1", 0.6, 0.5))
    dataset = tmp_path / "data" / "processed" / "damage-v1"
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "data.yaml").write_text("names:\n  0: dent\n", encoding="utf-8")
    for i in range(4):
        (dataset / "images" / "val" / f"{i}.jpg").write_bytes(b"x")
        (dataset / "labels" / "val" / f"{i}.txt").write_text(
            "0 " + " ".join(map(str, SQUARE)) + "\n", encoding="utf-8"
        )
    update_models_config(
        config / "models.toml", "damage", champion_from_report(_report("d1", 0.6, 0.5))
    )
    monkeypatch.chdir(tmp_path)
    code = main(
        ["--config", str(config), "train", "calibrate", "d1"],
        segmenter_factory=lambda weights, name: _CalibrationSegmenter(),
    )
    assert code == 0
    models = load_models_config(config / "models.toml")
    assert models is not None
    assert models.damage is not None
    assert models.damage.temperature is not None
    assert (tmp_path / "reports" / "models" / "calibration" / "d1.json").is_file()


def test_calibrate_with_no_predictions_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.training.manifest import write_json
    from tests.unit.test_training_select import _report

    write_json(tmp_path / "reports" / "models" / "d1.json", _report("d1", 0.6, 0.5))
    dataset = tmp_path / "data" / "processed" / "damage-v1"
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "data.yaml").write_text("names:\n  0: dent\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = main(
        ["train", "calibrate", "d1"], segmenter_factory=lambda w, n: _CalibrationSegmenter()
    )
    assert code == 1
    assert "no matched predictions" in capsys.readouterr().err


class _EmptySegmenter:
    model_version = "fake"

    def segment(self, image_path: Path) -> Segmentation:
        return Segmentation(instances=(), width=1, height=1)


def test_train_export_and_onnx_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from claimlens.training.manifest import ModelReport, write_json
    from tests.unit.test_training_select import _report

    write_json(
        tmp_path / "reports" / "models" / "p1.json",
        _report("p1", 0.6, 0.5).model_copy(update={"task": "parts", "dataset": "parts-v1"}),
    )
    weights = tmp_path / "models" / "parts" / "p1" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"pt")
    test_dir = tmp_path / "data" / "processed" / "parts-v1" / "images" / "test"
    test_dir.mkdir(parents=True)
    for i in range(3):
        (test_dir / f"{i}.jpg").write_bytes(b"x")

    def fake_export(path: Path) -> Path:
        out = path.parent / "exported.onnx"
        out.write_bytes(b"onnx")
        return out

    seen: list[Path] = []

    def factory(path: Path, name: str) -> _EmptySegmenter:
        seen.append(path)
        return _EmptySegmenter()

    monkeypatch.chdir(tmp_path)
    assert main(["train", "export", "p1"], exporter=fake_export) == 0
    assert (tmp_path / "models" / "parts" / "p1" / "best.onnx").read_bytes() == b"onnx"
    assert main(["train", "benchmark", "p1", "--format", "onnx"], segmenter_factory=factory) == 0
    assert seen[-1].name == "best.onnx"
    report = read_json(tmp_path / "reports" / "models" / "p1.json", ModelReport)
    assert report.cpu_ms_per_image_onnx is not None
    assert report.cpu_ms_per_image == 120.0  # the .pt timing is left unchanged


class _PartsSegmenter:
    model_version = "fake-parts"

    def segment(self, image_path: Path) -> Segmentation:
        door = SegInstance(
            label="front_left_door", confidence=0.9, box_xyxy=(0, 0, 1, 1), polygon_xyn=SQUARE
        )
        return Segmentation(instances=(door,), width=10, height=10)


def test_train_fusion_eval_reports_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.data.records import Annotation, ImageRecord, write_records

    record = ImageRecord(
        image_id="s:a",
        source="s",
        source_split="test",
        path="a.jpg",
        width=10,
        height=10,
        annotations=(Annotation(label="door", polygon=SQUARE),),
    )
    write_records(tmp_path / "data" / "processed" / "fusion-eval-v1" / "parts.jsonl", [record])
    monkeypatch.chdir(tmp_path)
    code = main(
        ["--config", str(CONFIG_DIR), "train", "fusion-eval", "p1"],
        segmenter_factory=lambda w, n: _PartsSegmenter(),
    )
    assert code == 0
    data = json.loads(
        (tmp_path / "reports" / "models" / "fusion-eval" / "p1.json").read_text(encoding="utf-8")
    )
    assert data == {"door": {"agreed": 1, "total": 1}}
    assert "1 of 1" in capsys.readouterr().out


def test_select_on_a_fresh_logbook_fails_cleanly_and_keeps_models_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("mlflow")
    import shutil

    from claimlens.training.manifest import write_json
    from tests.unit.test_training_select import _report

    config = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, config)
    before = (config / "models.toml").read_text(encoding="utf-8")
    write_json(tmp_path / "reports" / "models" / "d1.json", _report("d1", 0.9, 0.9))
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config), "train", "select"]) == 1
    assert "train import" in capsys.readouterr().err
    assert (config / "models.toml").read_text(encoding="utf-8") == before
