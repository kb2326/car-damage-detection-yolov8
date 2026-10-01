"""Content-addressed storage for uploaded photos."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredBlob:
    sha256: str
    name: str


class BlobStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(self, src: Path) -> StoredBlob:
        data = src.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        name = f"{sha256}{src.suffix.lower()}"
        destination = self._root / name
        if not destination.exists():
            destination.write_bytes(data)
        return StoredBlob(sha256=sha256, name=name)

    def path(self, name: str) -> Path:
        return self._root / name
