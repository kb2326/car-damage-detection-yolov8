from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from claimlens.quality import QualityConfig, check_quality


def test_accepts_a_normal_photo(make_image: Callable[..., Path]) -> None:
    result = check_quality(make_image("ok.jpg"), QualityConfig())
    assert result.ok
    assert (result.width, result.height) == (640, 480)


def test_rejects_a_small_photo(make_image: Callable[..., Path]) -> None:
    result = check_quality(make_image("tiny.png", size=(100, 100)), QualityConfig())
    assert not result.ok
    assert "too small" in result.reason


def test_rejects_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    path = tmp_path / "note.jpg"
    path.write_text("not an image", encoding="utf-8")
    result = check_quality(path, QualityConfig())
    assert not result.ok
    assert result.reason == "not a readable image"


def test_rejects_an_oversized_file(make_image: Callable[..., Path]) -> None:
    result = check_quality(make_image("big.jpg"), QualityConfig(max_bytes=100))
    assert not result.ok
    assert "limit is 100 bytes" in result.reason


def test_rejects_an_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "scan.bmp"
    Image.new("RGB", (640, 480)).save(path, format="BMP")
    result = check_quality(path, QualityConfig())
    assert not result.ok
    assert result.reason == "unsupported format BMP"


def test_rejects_a_decompression_bomb_instead_of_crashing(
    make_image: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    result = check_quality(make_image("bomb.png"), QualityConfig())
    assert not result.ok
    assert result.reason == "image has too many pixels to decode safely"
