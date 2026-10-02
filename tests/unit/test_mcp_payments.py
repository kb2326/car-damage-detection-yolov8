from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from mcp.types import CallToolResult

from claimlens.domain import Decision, Route
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import ClaimReported, RouteDecided
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.mcp.base import ScopedServer
from claimlens.mcp.guard import issue_approval
from claimlens.mcp.payments import build_payments
from claimlens.mcp.profiles import OPERATOR, load_profiles
from tests.mcp_helpers import MemoryAudit, call, error_text, tool_names

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 2, 12, tzinfo=UTC)
SECRET = "s3cret"
SYSTEM = Actor(kind=ActorKind.SYSTEM, name="test")


def _claim(db: Path, route: Route = Route.FAST_TRACK) -> UUID:
    store = SQLiteEventStore(db)
    claim = uuid4()
    store.append(claim, ClaimReported(policy_id="P-1001", description=""), SYSTEM)
    decision = Decision(route=route, rule_id="R9", reason="ok", policy_version="v")
    store.append(claim, RouteDecided(decision=decision), SYSTEM)
    store.close()
    return claim


def _server(db: Path, secret: str = SECRET) -> ScopedServer:
    return build_payments(OPERATOR, db, secret, MemoryAudit(), now=lambda: NOW)


def _pay(
    server: ScopedServer, claim: UUID, amount: int, token: str, key: str = "p1"
) -> CallToolResult:
    return call(
        server,
        "issue_payment",
        {
            "claim_id": str(claim),
            "amount_usd": amount,
            "approval_token": token,
            "idempotency_key": key,
        },
    )


def test_valid_token_pays_once(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    claim = _claim(db)
    token = issue_approval(str(claim), 850, SECRET, now=NOW)
    server = _server(db)
    first = _pay(server, claim, 850, token).structured_content
    again = _pay(server, claim, 850, token).structured_content
    assert first is not None
    assert first["duplicate"] is False
    assert again is not None
    assert again["duplicate"] is True
    assert again["payment_id"] == first["payment_id"]
    assert len(fold(SQLiteEventStore(db).load(claim)).payments) == 1


@pytest.mark.parametrize("amount", [851, 85000])
def test_amount_must_match_the_token(tmp_path: Path, amount: int) -> None:
    db = tmp_path / "c.db"
    claim = _claim(db)
    token = issue_approval(str(claim), 850, SECRET, now=NOW)
    result = _pay(_server(db), claim, amount, token)
    assert result.is_error
    assert "different claim or amount" in error_text(result)
    assert fold(SQLiteEventStore(db).load(claim)).payments == []


def test_fraud_routed_claims_are_never_paid(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    claim = _claim(db, Route.FRAUD_REVIEW)
    token = issue_approval(str(claim), 850, SECRET, now=NOW)
    result = _pay(_server(db), claim, 850, token)
    assert result.is_error
    assert "FRAUD_REVIEW" in error_text(result)


def test_missing_secret_refuses(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    claim = _claim(db)
    result = _pay(_server(db, secret=""), claim, 850, "anything")
    assert result.is_error
    assert "CLAIMLENS_APPROVAL_SECRET" in error_text(result)


def test_agent_profiles_see_no_payment_tool(tmp_path: Path) -> None:
    for profile in load_profiles(ROOT / "config" / "agents.toml").values():
        server = build_payments(profile, tmp_path / "c.db", SECRET, MemoryAudit())
        assert tool_names(server) == []


def test_one_approval_pays_only_once(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    claim = _claim(db)
    token = issue_approval(str(claim), 850, SECRET, now=NOW)
    server = _server(db)
    assert not _pay(server, claim, 850, token, key="a").is_error
    replay = _pay(server, claim, 850, token, key="b")
    assert replay.is_error
    assert "already used" in error_text(replay)
    assert len(fold(SQLiteEventStore(db).load(claim)).payments) == 1


def test_claims_without_a_decision_are_not_paid(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    store = SQLiteEventStore(db)
    claim = uuid4()
    store.append(claim, ClaimReported(policy_id="P-1001", description=""), SYSTEM)
    store.close()
    token = issue_approval(str(claim), 850, SECRET, now=NOW)
    result = _pay(_server(db), claim, 850, token)
    assert result.is_error
    assert "no decision" in error_text(result)


def test_a_note_key_does_not_block_a_payment(tmp_path: Path) -> None:
    from claimlens.events.payloads import NoteAdded

    db = tmp_path / "c.db"
    claim = _claim(db)
    store = SQLiteEventStore(db)
    store.append(claim, NoteAdded(text="x", author="a", idempotency_key="p1"), SYSTEM)
    store.close()
    token = issue_approval(str(claim), 850, SECRET, now=NOW)
    result = _pay(_server(db), claim, 850, token, key="p1")
    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["duplicate"] is False


def test_payment_audits_name_a_human_actor(tmp_path: Path) -> None:
    from claimlens.mcp.claims_system import event_audit

    db = tmp_path / "c.db"
    claim = _claim(db)
    token = issue_approval(str(claim), 850, SECRET, now=NOW)
    server = build_payments(OPERATOR, db, SECRET, event_audit(db, MemoryAudit()), now=lambda: NOW)
    _pay(server, claim, 850, token)
    audit = SQLiteEventStore(db).load(claim)[-1]
    assert audit.type == "ToolCalled"
    assert audit.actor.kind.value == "human"
