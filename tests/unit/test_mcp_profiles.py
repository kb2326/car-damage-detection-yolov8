from pathlib import Path

import pytest

from claimlens.mcp.profiles import OPERATOR, SERVERS, WRITE_TOOLS, load_profiles

ROOT = Path(__file__).resolve().parents[2]
PROFILES = load_profiles(ROOT / "config" / "agents.toml")


def test_repo_profiles_load() -> None:
    assert set(PROFILES) == {"intake", "triage", "demo"}
    assert PROFILES["intake"].allows("vision", "assess_quality")
    assert not PROFILES["intake"].allows("claims-system", "get_claim_history")


def test_no_agent_profile_can_pay() -> None:
    assert all(not p.allows("payments", "issue_payment") for p in PROFILES.values())
    assert OPERATOR.allows("payments", "issue_payment")


def test_demo_profile_is_read_only() -> None:
    demo = PROFILES["demo"]
    assert not any(
        demo.allows(s, t) for s, tools in SERVERS.items() for t in tools if t in WRITE_TOOLS
    )


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "agents.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_server_or_tool_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown server"):
        load_profiles(_write(tmp_path, '[profiles.x]\nbilling = ["refund"]\n'))
    with pytest.raises(ValueError, match="unknown tool"):
        load_profiles(_write(tmp_path, '[profiles.x]\nvision = ["delete_photo"]\n'))


def test_a_profile_listing_payments_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="issue_payment"):
        load_profiles(_write(tmp_path, '[profiles.x]\npayments = ["issue_payment"]\n'))


def test_operator_name_is_reserved(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        load_profiles(_write(tmp_path, '[profiles.operator]\nvision = ["assess_quality"]\n'))
