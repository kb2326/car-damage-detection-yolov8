"""Golden claims: labelled end-to-end test cases stored as JSON Lines."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from claimlens.domain import Frozen, Route


class PriorClaim(Frozen):
    """A claim submitted before the case under test, e.g. to set up photo reuse."""

    policy_id: str
    photos: tuple[str, ...]


class GoldenClaim(Frozen):
    case_id: str
    scenario: str
    policy_id: str
    description: str
    photos: tuple[str, ...]
    expected_route: Route
    label_source: str
    prior_claims: tuple[PriorClaim, ...] = ()
    reviewed: bool = False
    notes: str = ""


def load_golden(path: Path) -> list[GoldenClaim]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cases = [GoldenClaim.model_validate_json(line) for line in lines if line.strip()]
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate case id {case.case_id}")
        seen.add(case.case_id)
    return cases


def write_golden(path: Path, cases: Sequence[GoldenClaim]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8")
