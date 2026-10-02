"""Assign whole near-duplicate clusters to one split so no image leaks across splits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence, Set

from claimlens.data.records import ImageRecord


def assign_splits(
    records: Sequence[ImageRecord],
    clusters: Mapping[str, int],
    protected_clusters: Set[int],
) -> dict[str, str]:
    """Test wins over everything; protected clusters leave train and valid; valid wins over train."""
    source_splits: dict[int, set[str]] = defaultdict(set)
    for record in records:
        source_splits[clusters[record.image_id]].add(record.source_split)

    def target(cluster: int) -> str:
        splits = source_splits[cluster]
        if "test" in splits:
            # The frozen benchmark split is never altered, even next to a golden photo.
            return "test"
        if cluster in protected_clusters:
            return "excluded"
        if "valid" in splits:
            return "valid"
        return "train"

    return {record.image_id: target(clusters[record.image_id]) for record in records}
