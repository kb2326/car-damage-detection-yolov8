"""Dataset statistics, written as DVC metrics and a Markdown summary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from claimlens.data.records import ImageRecord
from claimlens.data.taxonomy import Taxonomy

_ORDER = ("train", "valid", "test", "excluded")


def dataset_stats(
    dataset_id: str,
    records: Sequence[ImageRecord],
    splits: Mapping[str, str],
    taxonomy: Taxonomy,
    *,
    clusters: Mapping[str, int],
    ignored_labels: Mapping[str, int],
    warning_counts: Mapping[str, int],
) -> dict[str, Any]:
    images: Counter[str] = Counter(splits[r.image_id] for r in records)
    instances: dict[str, dict[str, int]] = {}
    for split in _ORDER:
        if images[split]:
            counts = Counter(
                a.label for r in records if splits[r.image_id] == split for a in r.annotations
            )
            instances[split] = {label: counts[label] for label in taxonomy.classes}
    return {
        "dataset": dataset_id,
        "taxonomy": taxonomy.name,
        "images": {split: images[split] for split in _ORDER if images[split]},
        "instances": instances,
        "clusters": len({clusters[r.image_id] for r in records}),
        "leaks_prevented": sum(
            1 for r in records if splits[r.image_id] not in ("excluded", r.source_split)
        ),
        "ignored_labels": dict(ignored_labels),
        "validation_warnings": dict(warning_counts),
    }


def stats_markdown(stats: Mapping[str, Any]) -> str:
    classes = list(next(iter(stats["instances"].values()), {}).keys())
    lines = [
        f"# Dataset {stats['dataset']}",
        "",
        f"- Taxonomy: `{stats['taxonomy']}`",
        f"- Near-duplicate clusters: {stats['clusters']}",
        f"- Images moved to another split to prevent leakage: {stats['leaks_prevented']}",
        f"- Ignored labels: {stats['ignored_labels'] or 'none'}",
        f"- Validation warnings: {stats['validation_warnings'] or 'none'}",
        "",
        "| Split | Images | " + " | ".join(classes) + " |",
        "|---" * (len(classes) + 2) + "|",
    ]
    for split, count in stats["images"].items():
        per_class = stats["instances"].get(split, {})
        cells = " | ".join(str(per_class.get(c, 0)) for c in classes)
        lines.append(f"| {split} | {count} | {cells} |")
    return "\n".join(lines) + "\n"
