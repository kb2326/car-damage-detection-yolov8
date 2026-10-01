"""Build a dataset end to end: convert, validate, deduplicate, split, export, report."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from claimlens.data.config import SourceConfig, load_data_config
from claimlens.data.convert_coco import convert_coco_split
from claimlens.data.convert_yolo import convert_yolo_split, read_class_names
from claimlens.data.dedupe import find_clusters, hash_file
from claimlens.data.export import export_yolo_seg
from claimlens.data.records import ConversionResult, ImageRecord, relative_posix, write_records
from claimlens.data.split import assign_splits
from claimlens.data.stats import dataset_stats, stats_markdown
from claimlens.data.taxonomy import Taxonomy, load_taxonomies
from claimlens.data.validate import ValidationReport, validate_records
from claimlens.evals.golden import load_golden

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class DataContractError(Exception):
    def __init__(self, report: ValidationReport) -> None:
        super().__init__(f"data contract failed with {len(report.errors)} error(s)")
        self.report = report


@dataclass(frozen=True)
class BuildResult:
    dataset_id: str
    stats: dict[str, Any]
    validation: ValidationReport


def convert_source(
    source: SourceConfig, taxonomy: Taxonomy, *, repo_root: Path
) -> ConversionResult:
    root = repo_root / "data" / "raw" / source.id
    combined = ConversionResult(records=[])
    for split, layout in source.splits.items():
        if source.format == "coco-seg":
            result = convert_coco_split(
                root / str(layout.annotations),
                source=source.id,
                split=split,
                taxonomy=taxonomy,
                repo_root=repo_root,
            )
        else:
            names = read_class_names(root / str(source.class_names_file))
            result = convert_yolo_split(
                root / str(layout.images),
                root / str(layout.labels),
                source=source.id,
                split=split,
                class_names=names,
                taxonomy=taxonomy,
                repo_root=repo_root,
            )
        combined.records.extend(result.records)
        combined.ignored_labels.update(result.ignored_labels)
    return combined


def _readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def golden_image_paths(golden_file: Path, repo_root: Path) -> list[Path]:
    """Golden photos that are real images. Unreadable test files cannot leak into training."""
    paths: dict[str, Path] = {}
    for case in load_golden(golden_file):
        photos = [*case.photos, *(p for prior in case.prior_claims for p in prior.photos)]
        for photo in photos:
            path = repo_root / photo
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES and _readable_image(path):
                paths.setdefault(path.as_posix(), path)
    return list(paths.values())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def build_dataset(dataset_id: str, *, repo_root: Path, config_dir: Path) -> BuildResult:
    config = load_data_config(config_dir / "datasets.toml")
    dataset = config.datasets[dataset_id]
    taxonomy = load_taxonomies(config_dir / "taxonomy.toml")[dataset.taxonomy]
    reports = repo_root / "reports" / "data"

    records: list[ImageRecord] = []
    ignored: Counter[str] = Counter()
    for source_id in dataset.sources:
        converted = convert_source(config.sources[source_id], taxonomy, repo_root=repo_root)
        records.extend(converted.records)
        ignored.update(converted.ignored_labels)
    records.sort(key=lambda r: r.image_id)

    validation = validate_records(records, repo_root, taxonomy)
    _write_json(reports / f"{dataset_id}-validation.json", validation.model_dump(mode="json"))
    if not validation.ok:
        raise DataContractError(validation)

    hashes = [hash_file(r.image_id, repo_root / r.path) for r in records]
    golden_ids: list[str] = []
    if dataset.protect_golden:
        for path in golden_image_paths(repo_root / dataset.protect_golden, repo_root):
            golden_id = f"golden:{relative_posix(path, repo_root)}"
            hashes.append(hash_file(golden_id, path))
            golden_ids.append(golden_id)
    clusters = find_clusters(hashes, dataset.phash_max_distance)
    protected = {clusters[g] for g in golden_ids}
    splits = assign_splits(records, clusters, protected)

    interim = repo_root / "data" / "interim" / dataset_id
    write_records(interim / "records.jsonl", records)
    _write_json(
        interim / "clusters.json",
        {
            h.image_id: {
                "cluster": clusters[h.image_id],
                "sha256": h.sha256,
                "phash": f"{h.phash:016x}",
            }
            for h in hashes
        },
    )
    _write_json(interim / "splits.json", splits)

    export_yolo_seg(
        records,
        splits,
        taxonomy,
        repo_root=repo_root,
        out_dir=repo_root / "data" / "processed" / dataset_id,
    )
    stats = dataset_stats(
        dataset_id,
        records,
        splits,
        taxonomy,
        clusters=clusters,
        ignored_labels=ignored,
        warning_counts=validation.warning_counts,
    )
    _write_json(reports / f"{dataset_id}-stats.json", stats)
    (reports / f"{dataset_id}-stats.md").write_text(
        stats_markdown(stats), encoding="utf-8", newline="\n"
    )
    return BuildResult(dataset_id=dataset_id, stats=stats, validation=validation)
