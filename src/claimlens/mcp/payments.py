"""MCP `payments` server (mock). Only the human operator profile; every payment needs a token."""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import Field

from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import PaymentIssued
from claimlens.events.projection import find_by_idempotency_key, fold
from claimlens.events.store import SQLiteEventStore
from claimlens.mcp.base import AuditSink, ScopedServer, tool_errors
from claimlens.mcp.guard import verify_approval
from claimlens.mcp.profiles import Profile
from claimlens.mcp.schemas import PaymentResult
from claimlens.review_queue import payable


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_payments(
    profile: Profile,
    store_path: Path,
    secret: str,
    audit: AuditSink,
    *,
    now: Callable[[], datetime] = _utc_now,
) -> ScopedServer:
    server = ScopedServer("payments", profile, audit)

    def issue_payment(
        claim_id: str,
        amount_usd: Annotated[int, Field(gt=0, le=100_000)],
        approval_token: str,
        idempotency_key: Annotated[str, Field(min_length=1, max_length=128)],
    ) -> PaymentResult:
        with tool_errors():
            verify_approval(approval_token, claim_id, amount_usd, secret, now=now())
            store = SQLiteEventStore(store_path)
            try:
                uuid = UUID(claim_id)
                events = store.load(uuid)
                state = fold(events)
                refusal = payable(state)
                if refusal is not None:
                    raise ValueError(f"this claim cannot be paid: {refusal}")
                existing = find_by_idempotency_key(events, idempotency_key, "PaymentIssued")
                if existing is not None:
                    return PaymentResult(
                        payment_id=str(existing.payload["payment_id"]),
                        event_seq=existing.seq,
                        duplicate=True,
                    )
                approval = hashlib.sha256(approval_token.encode()).hexdigest()
                if any(
                    e.type == "PaymentIssued" and e.payload.get("approval_sha256") == approval
                    for e in events
                ):
                    raise ValueError("ApprovalInvalid: this approval token was already used")
                payment_id = (
                    "pay-"
                    + hashlib.sha256(f"{claim_id}|{idempotency_key}".encode()).hexdigest()[:12]
                )
                event = store.append(
                    uuid,
                    PaymentIssued(
                        payment_id=payment_id,
                        amount_usd=amount_usd,
                        idempotency_key=idempotency_key,
                        approval_sha256=approval,
                    ),
                    Actor(kind=ActorKind.HUMAN, name="payments-operator"),
                )
                return PaymentResult(payment_id=payment_id, event_seq=event.seq, duplicate=False)
            finally:
                store.close()

    server.register(issue_payment, "issue_payment", "Issue a mock payment (needs approval token).")
    return server
