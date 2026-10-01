import io
import json
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.config import SourceConfig, load_data_config, read_secret
from claimlens.data.fetch import extract_zip, fetch_source, roboflow_download_url

ROOT = Path(__file__).resolve().parents[2]


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _url_source() -> SourceConfig:
    return SourceConfig.model_validate(
        {
            "id": "parts",
            "kind": "url",
            "format": "yolo-seg",
            "url": "https://example.test/parts.zip",
            "terms": "test",
            "splits": {"train": {"images": "images/train", "labels": "labels/train"}},
        }
    )


def test_repo_config_loads() -> None:
    config = load_data_config(ROOT / "config" / "datasets.toml")
    cardd = config.sources["cardd-roboflow-v6"]
    assert cardd.splits["test"].annotations == "test/_annotations.coco.json"
    assert config.datasets["damage-v1"].sources == ("cardd-roboflow-v6",)
    assert config.datasets["damage-v1"].phash_max_distance == 6


def test_yolo_source_requires_images_and_labels() -> None:
    with pytest.raises(ValidationError, match="images and labels"):
        SourceConfig.model_validate(
            {
                "id": "x",
                "kind": "local",
                "format": "yolo-seg",
                "terms": "t",
                "splits": {"train": {}},
            }
        )


def test_read_secret_prefers_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nROBOFLOW_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    assert read_secret("ROBOFLOW_API_KEY", env_file) == "from-file"
    monkeypatch.setenv("ROBOFLOW_API_KEY", "from-env")
    assert read_secret("ROBOFLOW_API_KEY", env_file) == "from-env"
    assert read_secret("MISSING_KEY", env_file) is None


def test_roboflow_url_comes_from_the_export_endpoint() -> None:
    config = load_data_config(ROOT / "config" / "datasets.toml")
    requested: list[str] = []

    def fake_get(url: str) -> bytes:
        requested.append(url)
        return json.dumps({"export": {"link": "https://download.test/x.zip"}}).encode()

    link = roboflow_download_url(config.sources["cardd-roboflow-v6"], "KEY", fake_get)
    assert link == "https://download.test/x.zip"
    assert requested[0].startswith(
        "https://api.roboflow.com/auto-industry/car-damage-detection-vyhvw/6/coco-segmentation"
    )


def test_roboflow_errors_do_not_leak_the_key() -> None:
    config = load_data_config(ROOT / "config" / "datasets.toml")

    def failing_get(url: str) -> bytes:
        raise OSError(f"could not reach {url}")

    with pytest.raises(RuntimeError) as excinfo:
        roboflow_download_url(config.sources["cardd-roboflow-v6"], "SECRET-KEY", failing_get)
    assert "SECRET-KEY" not in str(excinfo.value)


def test_fetch_extracts_and_writes_manifest(tmp_path: Path) -> None:
    archive = _zip({"images/train/a.jpg": b"jpeg-bytes", "labels/train/a.txt": b"0 0 0 1 0 1 1"})

    dest = fetch_source(_url_source(), tmp_path, http_get=lambda _url: archive)

    assert (dest / "images" / "train" / "a.jpg").read_bytes() == b"jpeg-bytes"
    manifest = json.loads((dest / "MANIFEST.json").read_text(encoding="utf-8"))
    assert [entry["path"] for entry in manifest] == ["images/train/a.jpg", "labels/train/a.txt"]
    assert manifest[0]["bytes"] == 10


def test_fetch_refuses_to_overwrite_raw_data(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        fetch_source(_url_source(), tmp_path, http_get=lambda _url: b"")


def test_extract_refuses_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        extract_zip(_zip({"../../evil.txt": b"x"}), tmp_path / "dest")
    assert not (tmp_path / "evil.txt").exists()


def test_roboflow_fetch_needs_a_key(tmp_path: Path) -> None:
    config = load_data_config(ROOT / "config" / "datasets.toml")
    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        fetch_source(config.sources["cardd-roboflow-v6"], tmp_path, http_get=lambda _url: b"")
