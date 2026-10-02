from pathlib import Path

import pytest

from claimlens.knowledge.clauses import load_wordings, parse_policy
from claimlens.policy import load_policies

ROOT = Path(__file__).resolve().parents[2]


def test_repo_wordings_parse_with_unique_ids() -> None:
    clauses = load_wordings(ROOT / "knowledge" / "policies")
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids))
    assert {c.wording for c in clauses} == {"basic", "standard", "premium"}
    assert all(c.text for c in clauses)
    rental = next(c for c in clauses if c.clause_id == "STD-6.1")
    assert "10 days" in rental.text


def test_clause_id_prefix_must_match_the_wording(tmp_path: Path) -> None:
    path = tmp_path / "basic.md"
    path.write_text("# Basic\n\n### STD-1.1 Wrong prefix\nText.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prefix"):
        parse_policy(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "basic.md"
    path.write_text("### BAS-1.1 A\nx\n### BAS-1.1 B\ny\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        parse_policy(path)


def test_every_policy_has_a_known_wording() -> None:
    wordings = {c.wording for c in load_wordings(ROOT / "knowledge" / "policies")}
    policies = load_policies(ROOT / "config" / "policies.toml")
    for policy_id in ("P-1001", "P-1003", "P-1004", "P-2002"):
        record = policies.get_record(policy_id)
        assert record is not None
        assert record.wording in wordings


@pytest.mark.parametrize(
    "heading",
    [
        "### BAS-1.2",
        "#### BAS-1.3 Too deep",
        "### BAS-1.4.1 Three levels",
        "### bas-1.5 lower case",
    ],
)
def test_malformed_clause_headings_are_errors(tmp_path: Path, heading: str) -> None:
    path = tmp_path / "basic.md"
    path.write_text(f"### BAS-1.1 Fine\nText.\n{heading}\nMore text.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed clause heading"):
        parse_policy(path)


def test_a_clause_without_text_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "basic.md"
    path.write_text("### BAS-1.1 Empty\n\n### BAS-1.2 Fine\nText.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no text"):
        parse_policy(path)
