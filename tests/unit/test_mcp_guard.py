from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from claimlens.mcp.guard import (
    ApprovalInvalid,
    PathRejected,
    ScopeDenied,
    check_scope,
    issue_approval,
    safe_photo_path,
    verify_approval,
)
from claimlens.mcp.profiles import SERVERS, load_profiles

ROOT = Path(__file__).resolve().parents[2]
PROFILES = load_profiles(ROOT / "config" / "agents.toml")
NOW = datetime(2026, 10, 2, 12, tzinfo=UTC)
SECRET = "test-secret"


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_scope_matrix_matches_the_config(profile: str) -> None:
    p = PROFILES[profile]
    for server, tools in SERVERS.items():
        for tool in tools:
            if p.allows(server, tool):
                check_scope(p, server, tool)
            else:
                with pytest.raises(ScopeDenied, match=f"{server}.{tool}"):
                    check_scope(p, server, tool)


def test_photo_inside_an_allowed_root_is_accepted(tmp_path: Path) -> None:
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    assert safe_photo_path(str(photo), [tmp_path]) == photo.resolve()


def test_dotenv_and_traversal_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    outside = tmp_path / "x.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(PathRejected, match="not an image"):
        safe_photo_path(str(tmp_path / ".env"), [root])
    with pytest.raises(PathRejected, match="outside"):
        safe_photo_path(str(root / ".." / "x.jpg"), [root])


def test_missing_photo_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathRejected, match="not found"):
        safe_photo_path(str(tmp_path / "nope.png"), [tmp_path])


def test_valid_token_verifies() -> None:
    token = issue_approval("c1", 850, SECRET, now=NOW)
    verify_approval(token, "c1", 850, SECRET, now=NOW + timedelta(hours=1))


def test_token_for_another_amount_is_invalid() -> None:
    token = issue_approval("c1", 850, SECRET, now=NOW)
    with pytest.raises(ApprovalInvalid, match="different claim or amount"):
        verify_approval(token, "c1", 9000, SECRET, now=NOW)
    with pytest.raises(ApprovalInvalid, match="different claim or amount"):
        verify_approval(token, "c2", 850, SECRET, now=NOW)


def test_tampered_expired_and_wrong_secret_tokens_are_invalid() -> None:
    token = issue_approval("c1", 850, SECRET, now=NOW)
    with pytest.raises(ApprovalInvalid, match="expired"):
        verify_approval(token, "c1", 850, SECRET, now=NOW + timedelta(hours=25))
    with pytest.raises(ApprovalInvalid, match="signature"):
        verify_approval(token, "c1", 850, "other-secret", now=NOW)
    with pytest.raises(ApprovalInvalid, match="malformed"):
        verify_approval("not-a-token", "c1", 850, SECRET, now=NOW)


def test_missing_secret_disables_payments() -> None:
    with pytest.raises(ApprovalInvalid, match="CLAIMLENS_APPROVAL_SECRET"):
        verify_approval("x", "c1", 850, "", now=NOW)
