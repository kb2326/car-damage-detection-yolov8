"""Settings for the web prototype, from config/web.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from claimlens.domain import Frozen


class WebSettings(Frozen):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_upload_mb: int = Field(default=10, ge=1, le=50)
    poll_seconds: float = Field(default=1.0, gt=0, le=10)


def load_web_settings(path: Path) -> WebSettings:
    return WebSettings.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
