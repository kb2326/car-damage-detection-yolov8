"""Settings for the web prototype, from config/web.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import Field

from claimlens.domain import Frozen


class WebSettings(Frozen):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_upload_mb: int = Field(default=10, ge=1, le=50)
    poll_seconds: float = Field(default=1.0, gt=0, le=10)
    # Host names the app answers to; anything else (DNS rebinding) is refused.
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    showcase: bool = False  # the public read-only demo: no writes, recorded samples


def load_web_settings(path: Path) -> WebSettings:
    """From config/web.toml; CLAIMLENS_ALLOWED_HOSTS (comma-separated) replaces the host list,
    for the public showcase's address."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    hosts = os.environ.get("CLAIMLENS_ALLOWED_HOSTS", "")
    if hosts.strip():
        data["allowed_hosts"] = tuple(h.strip() for h in hosts.split(",") if h.strip())
    return WebSettings.model_validate(data)
