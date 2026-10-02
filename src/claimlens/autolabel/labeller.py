"""Interface for anything that proposes part-group masks for an image."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Protocol

from claimlens.data.records import Annotation


class PartLabeller(Protocol):
    @property
    def model_version(self) -> str: ...

    def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]: ...
