from pathlib import Path

import pytest

from claimlens.policy import PolicyRecord, PolicyRepository, PolicyStatus, load_policies

ROOT = Path(__file__).resolve().parents[2]


def _record(policy_id: str = "P-1") -> PolicyRecord:
    return PolicyRecord(
        policy_id=policy_id, status=PolicyStatus.ACTIVE, collision=True, deductible=500
    )


def test_active_collision_policy_is_confirmed() -> None:
    coverage = PolicyRepository([_record()]).get_coverage("P-1")
    assert coverage.confirmed
    assert coverage.deductible == 500


def test_unknown_policy_is_not_confirmed() -> None:
    coverage = PolicyRepository([]).get_coverage("P-404")
    assert not coverage.found
    assert not coverage.confirmed


def test_duplicate_policy_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate policy id P-1"):
        PolicyRepository([_record(), _record()])


def test_repo_config_covers_each_scenario() -> None:
    repo = load_policies(ROOT / "config" / "policies.toml")
    assert repo.get_coverage("P-1001").confirmed
    assert not repo.get_coverage("P-2001").active
    assert not repo.get_coverage("P-2002").collision
