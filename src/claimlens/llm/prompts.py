"""Versioned prompt files: prompts/<name>/<version>.md."""

from __future__ import annotations

import hashlib
from pathlib import Path

from claimlens.domain import Frozen


class Prompt(Frozen):
    name: str
    version: str
    text: str
    sha256: str

    @property
    def id(self) -> str:
        return f"{self.name}/{self.version}"


def load_prompt(root: Path, name: str, version: str) -> Prompt:
    path = root / name / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    return Prompt(
        name=name, version=version, text=text, sha256=hashlib.sha256(text.encode()).hexdigest()
    )
