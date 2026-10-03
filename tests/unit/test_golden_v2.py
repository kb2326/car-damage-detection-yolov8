import tomllib
from collections import Counter
from pathlib import Path

from claimlens.domain import Route
from claimlens.evals.golden import check_golden_citations, load_golden
from claimlens.knowledge.clauses import load_wordings
from claimlens.policy import load_policies

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "evals" / "golden" / "v1" / "claims.jsonl"
V2 = ROOT / "evals" / "golden" / "v2" / "claims.jsonl"
EXPECTED = {
    "exclusion_commercial": 7,
    "exclusion_racing": 5,
    "exclusion_driver": 5,
    "exclusion_intentional_or_wear": 5,
    "late_report": 5,
    "story_mismatch": 6,
    "cover_missing_basic": 5,
    "prompt_injection": 5,
    "benign_distractor": 10,
}


def test_v2_is_v1_plus_53_narrative_cases() -> None:
    v1_lines = V1.read_text(encoding="utf-8").splitlines()
    v2_lines = V2.read_text(encoding="utf-8").splitlines()
    assert v2_lines[:97] == v1_lines
    cases = load_golden(V2)
    assert len(cases) == 150
    new = cases[97:]
    assert [c.case_id for c in new] == [f"n{i:03d}" for i in range(1, 54)]
    assert all(c.narrative for c in new)
    assert all(c.label_source == "ai_authored_narrative" for c in new)
    assert not any(c.reviewed for c in new)
    assert Counter(c.scenario for c in new) == EXPECTED


def test_narrative_cases_reuse_photos_of_v1_fast_track_cases() -> None:
    cases = load_golden(V2)
    fast_track_photos = {
        p for c in cases[:97] if c.expected_route is Route.FAST_TRACK for p in c.photos
    }
    for case in cases[97:]:
        assert len(case.photos) == 1
        assert set(case.photos) <= fast_track_photos, case.case_id


def test_routes_and_reasons_match_the_scenario() -> None:
    for case in load_golden(V2)[97:]:
        if case.scenario == "benign_distractor":
            assert case.expected_route is Route.FAST_TRACK
            assert not case.expected_citations
        else:
            assert case.expected_route is Route.ADJUSTER_REVIEW
            assert case.must_not_fast_track_reason


def test_narrative_policies_have_active_collision_cover() -> None:
    policies = load_policies(ROOT / "config" / "policies.toml")
    for case in load_golden(V2)[97:]:
        assert policies.get_coverage(case.policy_id).confirmed, case.case_id


def test_every_expected_citation_exists_in_the_right_wording() -> None:
    policies = load_policies(ROOT / "config" / "policies.toml")
    wordings = {c.clause_id: c.wording for c in load_wordings(ROOT / "knowledge" / "policies")}
    cases = load_golden(V2)
    by_policy: dict[str, str] = {}
    for case in cases[97:]:
        record = policies.get_record(case.policy_id)
        assert record is not None
        by_policy[case.policy_id] = record.wording
    assert check_golden_citations(cases[97:], by_policy, wordings.get) == []


def test_the_narratives_file_matches_the_generated_cases() -> None:
    narratives = tomllib.loads(
        (ROOT / "evals" / "golden" / "v2" / "narratives.toml").read_text(encoding="utf-8")
    )["case"]
    new = load_golden(V2)[97:]
    assert [n["description"] for n in narratives] == [c.description for c in new]
    assert len({n["description"] for n in narratives}) == 53
