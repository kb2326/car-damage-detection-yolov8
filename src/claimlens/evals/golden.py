"""Golden claims: labelled end-to-end test cases stored as JSON Lines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Self

from pydantic import model_validator

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
    # Narrative cases (golden v2): the story, not the photo, decides the route.
    narrative: bool = False
    expected_citations: tuple[str, ...] = ()  # any one of these clause ids is a hit
    must_not_fast_track_reason: str = ""

    @model_validator(mode="after")
    def _reason_only_for_escalations(self) -> Self:
        if self.must_not_fast_track_reason and self.expected_route is Route.FAST_TRACK:
            raise ValueError("must_not_fast_track_reason is only for cases that must escalate")
        return self


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
    path.write_text("".join(_line(case) + "\n" for case in cases), encoding="utf-8")


_NARRATIVE_FIELDS = ("narrative", "expected_citations", "must_not_fast_track_reason")


def _line(case: GoldenClaim) -> str:
    """One JSON line; the narrative fields are left out when unset, so v1 files stay unchanged."""
    unset = {f for f in _NARRATIVE_FIELDS if f not in case.model_fields_set}
    return case.model_dump_json(exclude=unset)


def check_golden_citations(
    cases: Sequence[GoldenClaim],
    wording_of_policy: Mapping[str, str],
    wording_of_clause: Callable[[str], str | None],
) -> list[str]:
    """Problems with expected citations: unknown clauses, or clauses from another wording."""
    problems: list[str] = []
    for case in cases:
        wording = wording_of_policy.get(case.policy_id)
        for clause in case.expected_citations:
            actual = wording_of_clause(clause)
            if actual is None:
                problems.append(f"{case.case_id}: clause {clause} does not exist")
            elif actual != wording:
                problems.append(
                    f"{case.case_id}: clause {clause} is from the {actual} wording, "
                    f"but {case.policy_id} uses {wording}"
                )
    return problems
