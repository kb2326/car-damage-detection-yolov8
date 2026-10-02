from pathlib import Path

import pytest

from claimlens.training.manifest import RunManifest, RunMetrics, read_json
from claimlens.training.run import DatasetMismatchError, epochs_from_results, run_training
from tests.training_helpers import RESULTS_CSV, FakeTrainer, make_run, make_run_dir


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: {0: dent}\n", encoding="utf-8")
    return dataset


def test_a_complete_run_writes_manifest_and_metrics(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    manifest = read_json(run_dir / "manifest.json", RunManifest)
    metrics = read_json(run_dir / "metrics.json", RunMetrics)
    assert manifest.status == "complete"
    assert manifest.commit == "0123abc"
    assert manifest.dataset_md5 == "abc.dir"
    assert manifest.trainer_version == "fake-trainer-1"
    assert (manifest.epochs_run, manifest.best_epoch) == (3, 2)
    assert metrics.val.mask_map50 == 0.55
    assert metrics.test.mask_map50 == 0.5
    assert (run_dir / "train" / "weights" / "best.pt").read_bytes() == b"weights"


def test_validation_and_test_are_each_evaluated_once(tmp_path: Path) -> None:
    trainer = FakeTrainer()
    run_training(
        make_run(),
        dataset_dir=_dataset(tmp_path),
        bundle_md5="a",
        expected_md5="a",
        commit="c",
        out_dir=tmp_path / "runs",
        trainer=trainer,
    )
    assert trainer.evaluated == ["val", "test"]


def test_a_stale_bundle_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetMismatchError, match=r"old.dir"):
        run_training(
            make_run(),
            dataset_dir=_dataset(tmp_path),
            bundle_md5="old.dir",
            expected_md5="new.dir",
            commit="c",
            out_dir=tmp_path / "runs",
            trainer=FakeTrainer(),
        )


def test_missing_data_yaml_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"data\.yaml"):
        run_training(
            make_run(),
            dataset_dir=tmp_path,
            bundle_md5="a",
            expected_md5="a",
            commit="c",
            out_dir=tmp_path / "runs",
            trainer=FakeTrainer(),
        )


def test_existing_run_folder_is_refused(tmp_path: Path) -> None:
    make_run_dir(tmp_path)
    with pytest.raises(FileExistsError, match="r1"):
        make_run_dir(tmp_path)


def test_a_crash_after_a_checkpoint_gives_an_incomplete_run(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, fail=True)
    manifest = read_json(run_dir / "manifest.json", RunManifest)
    assert manifest.status == "incomplete"
    assert "CUDA out of memory" in manifest.note


def test_a_crash_without_a_checkpoint_is_raised(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        run_training(
            make_run(),
            dataset_dir=_dataset(tmp_path),
            bundle_md5="a",
            expected_md5="a",
            commit="c",
            out_dir=tmp_path / "runs",
            trainer=FakeTrainer(fail_after_weights=True, write_weights=False),
        )


def test_results_with_padded_headers_are_parsed(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(RESULTS_CSV, encoding="utf-8")
    assert epochs_from_results(path) == (3, 2)


def test_empty_results_give_zero_epochs(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("epoch,metrics/mAP50(B)\n", encoding="utf-8")
    assert epochs_from_results(path) == (0, 0)
