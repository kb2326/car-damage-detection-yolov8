import json
from pathlib import Path

import pytest
from PIL import Image

from claimlens.cli import main
from claimlens.data.pipeline import DataContractError, build_dataset
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, write_golden
from tests.data_helpers import make_pattern_image

ROOT = Path(__file__).resolve().parents[2]
SQUARE = "0 0.1 0.1 0.5 0.1 0.5 0.5\n"
DATASETS_TOML = """
[source.toy]
kind = "local"
format = "yolo-seg"
class_names_file = "data.yaml"
terms = "test"
[source.toy.splits]
train = { images = "train/images", labels = "train/labels" }
test = { images = "test/images", labels = "test/labels" }

[dataset.toy-v1]
taxonomy = "damage"
sources = ["toy"]
protect_golden = "golden.jsonl"
"""


def _repo(tmp_path: Path) -> Path:
    """A tiny repo: one YOLO source with a train/test duplicate and a golden lookalike."""
    raw = tmp_path / "data" / "raw" / "toy"
    for split, names in {"train": ["a", "b", "g"], "test": ["c"]}.items():
        for name in names:
            seed = {"a": 1, "b": 2, "c": 1, "g": 3}[name]
            make_pattern_image(raw / split / "images" / f"{name}.jpg", seed=seed)
            (raw / split / "labels").mkdir(parents=True, exist_ok=True)
            (raw / split / "labels" / f"{name}.txt").write_text(SQUARE, encoding="utf-8")
    (raw / "data.yaml").write_text("names: ['dent']\n", encoding="utf-8")

    golden_photo = tmp_path / "golden" / "g.jpg"
    golden_photo.parent.mkdir()
    with Image.open(raw / "train" / "images" / "g.jpg") as image:
        image.resize((300, 225)).save(golden_photo, quality=85)
    write_golden(
        tmp_path / "golden.jsonl",
        [
            GoldenClaim(
                case_id="g001",
                scenario="s",
                policy_id="P-1001",
                description="",
                photos=("golden/g.jpg",),
                expected_route=Route.FAST_TRACK,
                label_source="scenario",
            )
        ],
    )

    config = tmp_path / "config"
    config.mkdir()
    (config / "taxonomy.toml").write_text(
        'version = "t"\n[damage]\nclasses = ["dent"]\n', encoding="utf-8"
    )
    (config / "datasets.toml").write_text(DATASETS_TOML, encoding="utf-8")
    return config


def _splits(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "data" / "interim" / "toy-v1" / "splits.json"
    result: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return result


def test_build_moves_cross_split_duplicates_to_test(tmp_path: Path) -> None:
    config = _repo(tmp_path)
    result = build_dataset("toy-v1", repo_root=tmp_path, config_dir=config)

    splits = _splits(tmp_path)
    assert splits["toy:a"] == "test"
    assert splits["toy:c"] == "test"
    assert splits["toy:b"] == "train"
    assert result.stats["leaks_prevented"] == 1
    processed = tmp_path / "data" / "processed" / "toy-v1"
    assert (processed / "images" / "train" / "toy__b.jpg").is_file()
    assert (tmp_path / "reports" / "data" / "toy-v1-stats.md").is_file()


def test_build_excludes_images_near_golden_photos(tmp_path: Path) -> None:
    config = _repo(tmp_path)
    build_dataset("toy-v1", repo_root=tmp_path, config_dir=config)

    assert _splits(tmp_path)["toy:g"] == "excluded"
    processed = tmp_path / "data" / "processed" / "toy-v1" / "images"
    assert list(processed.rglob("toy__g.*")) == []


def test_unreadable_golden_photo_is_skipped(tmp_path: Path) -> None:
    config = _repo(tmp_path)
    (tmp_path / "golden" / "not_an_image.jpg").write_text("text", encoding="utf-8")
    write_golden(
        tmp_path / "golden.jsonl",
        [
            GoldenClaim(
                case_id="g001",
                scenario="unusable_photo",
                policy_id="P-1001",
                description="",
                photos=("golden/not_an_image.jpg", "golden/g.jpg"),
                expected_route=Route.ADJUSTER_REVIEW,
                label_source="scenario",
            )
        ],
    )
    build_dataset("toy-v1", repo_root=tmp_path, config_dir=config)
    assert _splits(tmp_path)["toy:g"] == "excluded"


def test_build_stops_on_contract_violation(tmp_path: Path) -> None:
    config = _repo(tmp_path)
    labels = tmp_path / "data" / "raw" / "toy" / "train" / "labels" / "b.txt"
    labels.write_text("0 0 0 1.8 0 1.8 1\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="1 error"):
        build_dataset("toy-v1", repo_root=tmp_path, config_dir=config)


def test_data_build_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config), "data", "build", "toy-v1"]) == 0
    assert "Built toy-v1" in capsys.readouterr().out


def test_data_build_command_reports_contract_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _repo(tmp_path)
    labels = tmp_path / "data" / "raw" / "toy" / "train" / "labels" / "b.txt"
    labels.write_text("0 0 0 1.8 0 1.8 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config), "data", "build", "toy-v1"]) == 1
    assert "out_of_bounds" in capsys.readouterr().err


def test_data_fetch_refuses_existing_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "data" / "raw" / "carparts-seg").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "carparts-seg" / "x").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(ROOT / "config"), "data", "fetch", "carparts-seg"]) == 2
    assert "immutable" in capsys.readouterr().err
