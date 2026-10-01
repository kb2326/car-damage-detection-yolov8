"""FiftyOne as the review UI. Requires `uv sync --group review`. Excluded from coverage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from claimlens.data.records import Annotation, ImageRecord
from claimlens.evals.golden import GoldenClaim
from claimlens.review.decisions import GoldenReview, PartReview, annotation_key


class ReviewUnavailableError(RuntimeError):
    pass


def _fo() -> Any:
    try:
        import fiftyone as fo
    except ImportError:
        raise ReviewUnavailableError(
            "FiftyOne is not installed: run `uv sync --group review`"
        ) from None
    return fo


def _polyline(fo: Any, annotation: Annotation, **attributes: object) -> Any:
    points = list(zip(annotation.polygon[0::2], annotation.polygon[1::2], strict=True))
    return fo.Polyline(
        label=annotation.label, points=[points], closed=True, filled=True, **attributes
    )


def launch_parts_review(
    records: Sequence[ImageRecord],
    *,
    repo_root: Path,
    name: str,
    damage: Mapping[str, ImageRecord],
) -> None:
    fo = _fo()
    dataset = fo.Dataset(name, overwrite=True, persistent=True)
    samples = []
    for record in records:
        sample = fo.Sample(
            filepath=str((repo_root / record.path).resolve()), image_id=record.image_id
        )
        sample["parts"] = fo.Polylines(
            polylines=[
                _polyline(fo, a, key=annotation_key(record.image_id, i), confidence=a.score)
                for i, a in enumerate(record.annotations)
            ]
        )
        context = damage.get(record.image_id)
        if context is not None:
            sample["damage"] = fo.Polylines(
                polylines=[_polyline(fo, a) for a in context.annotations]
            )
        samples.append(sample)
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    session.wait()


def export_parts_review(name: str, *, job_id: str, reviewer: str) -> PartReview:
    fo = _fo()
    decisions: dict[str, str] = {}
    for sample in fo.load_dataset(name):
        for polyline in sample["parts"].polylines:
            if "approved" in polyline.tags:
                decisions[polyline.key] = "approved"
            elif "rejected" in polyline.tags:
                decisions[polyline.key] = "rejected"
    return PartReview.model_validate(
        {"job_id": job_id, "reviewer": reviewer, "decisions": decisions}
    )


def launch_golden_review(cases: Sequence[GoldenClaim], *, repo_root: Path, name: str) -> None:
    fo = _fo()
    dataset = fo.Dataset(name, overwrite=True, persistent=True)
    samples = [
        fo.Sample(
            filepath=str((repo_root / case.photos[0]).resolve()),
            case_id=case.case_id,
            scenario=case.scenario,
            policy_id=case.policy_id,
            expected_route=case.expected_route.value,
        )
        for case in cases
    ]
    dataset.add_samples(samples)
    session = fo.launch_app(dataset)
    session.wait()


def export_golden_review(name: str, *, reviewer: str) -> GoldenReview:
    fo = _fo()
    decisions: dict[str, str] = {}
    for sample in fo.load_dataset(name):
        routes = [tag.removeprefix("route:") for tag in sample.tags if tag.startswith("route:")]
        if routes:
            decisions[sample["case_id"]] = routes[0]
        elif "agree" in sample.tags:
            decisions[sample["case_id"]] = "agree"
    return GoldenReview(reviewer=reviewer, decisions=decisions)
