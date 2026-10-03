from pathlib import Path

import pytest

from claimlens.web.settings import WebSettings, load_web_settings

ROOT = Path(__file__).resolve().parents[3]


def test_the_repo_settings_load() -> None:
    settings = load_web_settings(ROOT / "config" / "web.toml")
    assert settings == WebSettings(host="127.0.0.1", port=8000, max_upload_mb=10, poll_seconds=1.0)


def test_a_bad_upload_limit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "web.toml"
    path.write_text("max_upload_mb = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_upload_mb"):
        load_web_settings(path)
