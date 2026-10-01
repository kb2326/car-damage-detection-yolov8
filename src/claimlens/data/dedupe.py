"""Exact and perceptual duplicate detection, clustered with union-find."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageHash:
    image_id: str
    sha256: str
    phash: int


def hash_file(image_id: str, path: Path) -> ImageHash:
    data = path.read_bytes()
    with Image.open(path) as image:
        perceptual = imagehash.phash(image)
    return ImageHash(
        image_id=image_id,
        sha256=hashlib.sha256(data).hexdigest(),
        phash=int(str(perceptual), 16),
    )


def find_clusters(
    hashes: Sequence[ImageHash], max_distance: int, *, block: int = 512
) -> dict[str, int]:
    """Group images whose bytes match or whose pHash differs by at most `max_distance` bits."""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    first_with_sha: dict[str, int] = {}
    for i, item in enumerate(hashes):
        if item.sha256 in first_with_sha:
            union(first_with_sha[item.sha256], i)
        else:
            first_with_sha[item.sha256] = i

    values = np.array([item.phash for item in hashes], dtype=np.uint64)
    for start in range(0, len(hashes), block):
        chunk = values[start : start + block]
        distances = np.bitwise_count(chunk[:, None] ^ values[None, :])
        rows, cols = np.nonzero(distances <= max_distance)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            if start + row < col:
                union(start + row, col)

    cluster_of_root: dict[int, int] = {}
    return {
        item.image_id: cluster_of_root.setdefault(find(i), len(cluster_of_root))
        for i, item in enumerate(hashes)
    }
