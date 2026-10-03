import io
from pathlib import Path

import pytest
from PIL import Image

from claimlens.web.uploads import UploadError, check_upload


def _png(size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def test_a_png_is_kept_under_its_hash(tmp_path: Path) -> None:
    kept = check_upload(_png(), 1_000_000, tmp_path)
    assert kept.parent == tmp_path
    assert kept.suffix == ".png"
    assert check_upload(_png(), 1_000_000, tmp_path) == kept


def test_a_fake_jpeg_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="JPEG, PNG or WebP"):
        check_upload(b"not an image at all", 1_000_000, tmp_path)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_a_gif_is_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, "GIF")
    with pytest.raises(UploadError, match="JPEG, PNG or WebP"):
        check_upload(buffer.getvalue(), 1_000_000, tmp_path)


def test_a_large_photo_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="larger than 1 MB"):
        check_upload(_png(), 100, tmp_path, limit_mb=1)
