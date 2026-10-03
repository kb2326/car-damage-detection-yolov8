"""MCP `claims-system` server (mock claims system): history, similar claims, notes, queues."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import NoteAdded, QueueAssigned, ToolCalled
from claimlens.events.projection import ClaimState, find_by_idempotency_key, fold
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore
from claimlens.mcp.base import AuditRecord, AuditSink, ScopedServer, tool_errors
from claimlens.mcp.profiles import Profile
from claimlens.mcp.schemas import ClaimHistory, EventRef, SimilarClaim, SimilarClaims, WriteResult
from claimlens.memory.index import ClaimMemory
from claimlens.memory.records import build_record


def _parse(claim_id: str) -> UUID | None:
    try:
        return UUID(claim_id)
    except ValueError:
        return None


def _load(store: SQLiteEventStore, claim_id: str) -> tuple[UUID, ClaimState] | None:
    uuid = _parse(claim_id)
    if uuid is None:
        return None
    try:
        return uuid, fold(store.load(uuid))
    except ClaimNotFoundError:
        return None


def event_audit(store_path: Path, fallback: AuditSink) -> AuditSink:
    def sink(record: AuditRecord) -> None:
        uuid = _parse(record.claim_id) if record.claim_id else None
        store = SQLiteEventStore(store_path)
        try:
            if uuid is not None and uuid in store.claim_ids():
                store.append(
                    uuid,
                    ToolCalled(
                        server=record.server,
                        tool=record.tool,
                        profile=record.profile,
                        input_sha256=record.input_sha256,
                        outcome=record.outcome,
                    ),
                    Actor(
                        kind=ActorKind.HUMAN if record.profile == "operator" else ActorKind.AGENT,
                        name=f"mcp:{record.profile}",
                    ),
                )
                return
        finally:
            store.close()
        fallback(record)

    return sink


def _parts(state: ClaimState) -> set[str]:
    return {f.part for f in state.findings if f.part}


def build_claims_system(
    profile: Profile,
    store_path: Path,
    audit: AuditSink,
    *,
    memory: ClaimMemory | None = None,
    photo_path: Callable[[str], Path] | None = None,
) -> ScopedServer:
    server = ScopedServer("claims-system", profile, audit)
    actor = Actor(kind=ActorKind.AGENT, name=f"mcp:{profile.name}")

    def get_claim_history(claim_id: str) -> ClaimHistory:
        store = SQLiteEventStore(store_path)
        try:
            loaded = _load(store, claim_id)
            if loaded is None:
                return ClaimHistory(found=False, claim_id=claim_id)
            uuid, state = loaded
            return ClaimHistory(
                found=True,
                claim_id=claim_id,
                policy_id=state.policy_id,
                route=state.decision.route.value if state.decision else None,
                findings=[
                    f"{f.damage_type.value} on {f.part or 'unknown part'}" for f in state.findings
                ],
                notes=list(state.notes),
                queue=state.queue,
                events=[EventRef(seq=e.seq, type=e.type) for e in store.load(uuid)],
            )
        finally:
            store.close()

    def find_similar_claims(
        claim_id: str, limit: Annotated[int, Field(ge=1, le=20)] = 5
    ) -> SimilarClaims:
        store = SQLiteEventStore(store_path)
        try:
            loaded = _load(store, claim_id)
            if loaded is None:
                return SimilarClaims(items=[])
            uuid, state = loaded
            if memory is not None and photo_path is not None:
                events = store.load(uuid)
                mine = build_record(state, events, photo_path, provisional=True)
                return SimilarClaims(
                    items=[SimilarClaim(**s.model_dump()) for s in memory.similar(mine, limit)]
                )
            items: list[SimilarClaim] = []
            for other in sorted(store.claim_ids(), key=str):
                if other == uuid:
                    continue
                other_state = fold(store.load(other))
                shared = sorted(_parts(state) & _parts(other_state))
                if other_state.policy_id == state.policy_id:
                    items.append(SimilarClaim(claim_id=str(other), reason="same policy"))
                elif shared:
                    items.append(
                        SimilarClaim(claim_id=str(other), reason=f"same part: {shared[0]}")
                    )
            return SimilarClaims(items=items[:limit])
        finally:
            store.close()

    def _write(claim_id: str, key: str, payload: NoteAdded | QueueAssigned) -> WriteResult:
        store = SQLiteEventStore(store_path)
        try:
            uuid = _parse(claim_id)
            if uuid is None:
                raise ValueError(f"not a claim id: {claim_id!r}")
            events = store.load(uuid)
            existing = find_by_idempotency_key(events, key, payload.event_type.value)
            if existing is not None:
                return WriteResult(event_seq=existing.seq, duplicate=True)
            return WriteResult(event_seq=store.append(uuid, payload, actor).seq, duplicate=False)
        finally:
            store.close()

    def add_note(
        claim_id: str,
        text: Annotated[str, Field(min_length=1, max_length=2000)],
        idempotency_key: Annotated[str, Field(min_length=1, max_length=128)],
    ) -> WriteResult:
        with tool_errors():
            note = NoteAdded(text=text, author=actor.name, idempotency_key=idempotency_key)
            return _write(claim_id, idempotency_key, note)

    def assign_queue(
        claim_id: str,
        queue: Literal["adjuster", "fraud", "desk"],
        idempotency_key: Annotated[str, Field(min_length=1, max_length=128)],
    ) -> WriteResult:
        with tool_errors():
            assignment = QueueAssigned(queue=queue, idempotency_key=idempotency_key)
            return _write(claim_id, idempotency_key, assignment)

    server.register(get_claim_history, "get_claim_history", "Read a claim's state and events.")
    server.register(
        find_similar_claims, "find_similar_claims", "Claims with the same policy or part."
    )
    server.register(add_note, "add_note", "Add a note to a claim (idempotent).")
    server.register(assign_queue, "assign_queue", "Send a claim to a work queue (idempotent).")
    return server
