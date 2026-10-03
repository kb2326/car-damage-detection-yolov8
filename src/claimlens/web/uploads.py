"""Photo uploads: checked by content, size-limited, saved under their content hash."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_FORMATS = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp"}  # MPO: phone JPEGs


class UploadError(ValueError):
    """A plain message for the customer."""


def check_upload(data: bytes, max_bytes: int, folder: Path, *, limit_mb: int | None = None) -> Path:
    """Keep the photo if it really is a JPEG, PNG or WebP within the size limit. The name the
    browser sent is ignored: the file is named by its content hash."""
    if len(data) > max_bytes:
        shown = limit_mb if limit_mb is not None else max(1, max_bytes // 1_000_000)
        raise UploadError(f"That photo is larger than {shown} MB. Please send a smaller one.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            kind = image.format or ""
    except (UnidentifiedImageError, OSError, SyntaxError):
        kind = ""
    if kind not in _FORMATS:
        raise UploadError("Please send a photo as JPEG, PNG or WebP.")
    folder.mkdir(parents=True, exist_ok=True)
    kept = folder / f"{hashlib.sha256(data).hexdigest()}{_FORMATS[kind]}"
    if not kept.exists():
        kept.write_bytes(data)
    return kept
