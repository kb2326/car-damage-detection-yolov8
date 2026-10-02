"""Review decisions are data: stored as JSON in git and applied by pure functions."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from claimlens.data.records import ImageRecord
from claimlens.domain import Frozen, Route
from claimlens.evals.golden import GoldenClaim

PartDecision = Literal["approved", "rejected"]


def annotation_key(image_id: str, index: int) -> str:
    return f"{image_id}#{index}"


class PartReview(Frozen):
    job_id: str
    reviewer: str
    decisions: dict[str, PartDecision]
    # SHA-256 of the autolabels.jsonl the decisions were made against; keys are positional.
    proposals_sha256: str | None = None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_part_review(review: PartReview, *, job_id: str, proposals_sha256: str) -> None:
    """Refuse to apply decisions to a different job or to proposals that have changed."""
    if review.job_id != job_id:
        raise ValueError(f"review is for job {review.job_id!r}, not {job_id!r}")
    if review.proposals_sha256 is None:
        raise ValueError("review has no proposals_sha256; re-export it against the proposals")
    if review.proposals_sha256 != proposals_sha256:
        raise ValueError(
            "proposals changed since the review was made (proposals_sha256 mismatch); "
            "review the new proposals instead of reusing old decisions"
        )


def merge_part_reviews(old: PartReview, new: PartReview) -> PartReview:
    """Later decisions win; earlier ones are kept, so a spot-check never erases a full review."""
    reviewers = old.reviewer if new.reviewer == old.reviewer else f"{old.reviewer}; {new.reviewer}"
    return new.model_copy(
        update={"reviewer": reviewers, "decisions": {**old.decisions, **new.decisions}}
    )


class GoldenReview(Frozen):
    reviewer: str
    decisions: dict[str, str]


def apply_part_review(
    records: Sequence[ImageRecord], review: PartReview
) -> tuple[list[ImageRecord], dict[str, int]]:
    """Keep approved annotations; an image is used only if every proposal on it was decided."""
    kept: list[ImageRecord] = []
    counts: Counter[str] = Counter()
    for record in records:
        keys = [annotation_key(record.image_id, i) for i in range(len(record.annotations))]
        if not keys:
            counts["images_without_proposals"] += 1
            continue
        if any(key not in review.decisions for key in keys):
            counts["unreviewed_images"] += 1
            continue
        approved = tuple(
            annotation
            for key, annotation in zip(keys, record.annotations, strict=True)
            if review.decisions[key] == "approved"
        )
        counts["approved"] += len(approved)
        counts["rejected"] += len(keys) - len(approved)
        if approved:
            kept.append(record.model_copy(update={"annotations": approved}))
        else:
            counts["images_all_rejected"] += 1
    return kept, {key: value for key, value in counts.items() if value}


def apply_golden_review(cases: Sequence[GoldenClaim], review: GoldenReview) -> list[GoldenClaim]:
    updated: list[GoldenClaim] = []
    for case in cases:
        decision = review.decisions.get(case.case_id)
        if decision is None:
            updated.append(case)
        elif decision == "agree":
            updated.append(case.model_copy(update={"reviewed": True}))
        else:
            try:
                route = Route(decision)
            except ValueError:
                raise ValueError(
                    f"{case.case_id}: decision {decision!r} is neither 'agree' nor a route"
                ) from None
            if route is case.expected_route:
                updated.append(case.model_copy(update={"reviewed": True}))
                continue
            old = case.expected_route.value
            change = f"Reviewer changed expected route from {old} to {route.value}."
            note = f"{case.notes} {change}".strip()
            updated.append(
                case.model_copy(update={"reviewed": True, "expected_route": route, "notes": note})
            )
    return updated


def read_review[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_review(path: Path, review: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(review.model_dump_json(indent=1) + "\n", encoding="utf-8", newline="\n")
