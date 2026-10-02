from uuid import uuid4

from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import (
    ClaimReported,
    NoteAdded,
    PaymentIssued,
    QueueAssigned,
    ToolCalled,
)
from claimlens.events.projection import find_by_idempotency_key, fold
from claimlens.events.store import SQLiteEventStore

AGENT = Actor(kind=ActorKind.AGENT, name="mcp:triage")


def test_new_events_fold_into_state() -> None:
    store = SQLiteEventStore(":memory:")
    claim = uuid4()
    store.append(claim, ClaimReported(policy_id="P-1001", description="scrape"), AGENT)
    store.append(
        claim, NoteAdded(text="Photo is blurry", author="mcp:triage", idempotency_key="k1"), AGENT
    )
    store.append(claim, QueueAssigned(queue="adjuster", idempotency_key="k2"), AGENT)
    store.append(
        claim, PaymentIssued(payment_id="pay-1", amount_usd=850, idempotency_key="k3"), AGENT
    )
    store.append(
        claim,
        ToolCalled(
            server="claims-system",
            tool="add_note",
            profile="triage",
            input_sha256="ab",
            outcome="ok",
        ),
        AGENT,
    )
    state = fold(store.load(claim))
    assert state.notes == ["Photo is blurry"]
    assert state.queue == "adjuster"
    assert state.payments == ["pay-1"]
    assert state.last_seq == 5


def test_idempotency_lookup_finds_the_first_event() -> None:
    store = SQLiteEventStore(":memory:")
    claim = uuid4()
    store.append(claim, ClaimReported(policy_id="P-1001", description=""), AGENT)
    first = store.append(claim, NoteAdded(text="a", author="x", idempotency_key="k"), AGENT)
    events = store.load(claim)
    assert find_by_idempotency_key(events, "k", "NoteAdded") == first
    assert find_by_idempotency_key(events, "k", "QueueAssigned") is None
    assert find_by_idempotency_key(events, "other", "NoteAdded") is None
