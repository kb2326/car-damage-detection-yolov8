from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, load_golden

ROOT = Path(__file__).resolve().parents[2]


def test_v1_cases_still_load_with_the_new_optional_fields() -> None:
    cases = load_golden(ROOT / "evals" / "golden" / "v1" / "claims.jsonl")
    assert len(cases) == 97
    assert all(not c.narrative and c.expected_citations == () for c in cases)


def test_expected_citations_must_exist_in_the_policys_wording() -> None:
    from claimlens.evals.golden import check_golden_citations

    case = GoldenClaim(
        case_id="n001",
        scenario="exclusion_commercial",
        policy_id="P-1001",
        description="Delivering pizza.",
        photos=("a.jpg",),
        expected_route=Route.ADJUSTER_REVIEW,
        label_source="ai_authored_narrative",
        narrative=True,
        expected_citations=("STD-8.5", "PRM-8.5", "STD-99.9"),
        must_not_fast_track_reason="commercial use",
    )
    clause_wordings = {"STD-8.5": "standard", "PRM-8.5": "premium"}
    problems = check_golden_citations([case], {"P-1001": "standard"}, clause_wordings.get)
    assert problems == [
        "n001: clause PRM-8.5 is from the premium wording, but P-1001 uses standard",
        "n001: clause STD-99.9 does not exist",
    ]


def test_a_fast_track_case_cannot_have_a_must_not_fast_track_reason() -> None:
    with pytest.raises(ValidationError, match="must_not_fast_track_reason"):
        GoldenClaim(
            case_id="n002",
            scenario="benign",
            policy_id="P-1001",
            description="x",
            photos=("a.jpg",),
            expected_route=Route.FAST_TRACK,
            label_source="ai_authored_narrative",
            must_not_fast_track_reason="racing",
        )


def test_rewriting_v1_leaves_its_lines_unchanged(tmp_path: Path) -> None:
    from claimlens.evals.golden import write_golden

    source = ROOT / "evals" / "golden" / "v1" / "claims.jsonl"
    out = tmp_path / "claims.jsonl"
    write_golden(out, load_golden(source))
    assert out.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
