"""Run a verified auto-labelling job over a deterministic sample of a built dataset."""

from __future__ import annotations

import json
import random
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from claimlens.autolabel.labeller import PartLabeller
from claimlens.data.records import ImageRecord, read_records, write_records
from claimlens.data.taxonomy import PartGroups
from claimlens.domain import Frozen


class AutolabelJob(Frozen):
    id: str
    dataset: str
    split: str
    sample_size: int
    seed: int
    box_threshold: float
    text_threshold: float
    min_score: float
    max_per_group: int
    detector: str
    segmenter: str
    prompts: dict[str, str]


def load_autolabel_jobs(path: Path) -> dict[str, AutolabelJob]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {key: AutolabelJob.model_validate({"id": key, **body}) for key, body in data.items()}


def select_sample(
    records: Sequence[ImageRecord],
    splits: Mapping[str, str],
    *,
    split: str,
    size: int,
    seed: int,
) -> list[ImageRecord]:
    candidates = sorted(
        (r for r in records if splits.get(r.image_id) == split), key=lambda r: r.image_id
    )
    chosen = random.Random(seed).sample(candidates, min(size, len(candidates)))
    return sorted(chosen, key=lambda r: r.image_id)


def run_autolabel(
    job: AutolabelJob,
    *,
    repo_root: Path,
    labeller: PartLabeller,
    part_groups: PartGroups,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    interim = repo_root / "data" / "interim" / job.dataset
    records = read_records(interim / "records.jsonl")
    splits: dict[str, str] = json.loads((interim / "splits.json").read_text(encoding="utf-8"))
    sample = select_sample(records, splits, split=job.split, size=job.sample_size, seed=job.seed)

    proposals: list[ImageRecord] = []
    counts: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    for number, record in enumerate(sample, start=1):
        annotations, reasons = labeller.label(repo_root / record.path)
        for annotation in annotations:
            if annotation.label not in part_groups.classes:
                raise ValueError(f"labeller produced unknown part group {annotation.label!r}")
            counts[annotation.label] += 1
        dropped.update(reasons)
        proposals.append(record.model_copy(update={"annotations": tuple(annotations)}))
        if progress:
            progress(number, len(sample))

    write_records(repo_root / "data" / "interim" / job.id / "autolabels.jsonl", proposals)
    report: dict[str, Any] = {
        "job": job.id,
        "model_version": labeller.model_version,
        "images": len(proposals),
        "images_with_proposals": sum(1 for p in proposals if p.annotations),
        "proposals": dict(sorted(counts.items())),
        "dropped": dict(sorted(dropped.items())),
    }
    report_path = repo_root / "reports" / "data" / f"{job.id}-autolabel.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return report
