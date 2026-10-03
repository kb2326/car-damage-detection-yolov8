"""Rebuild claim state by folding its events. State is a projection, never stored as truth."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from claimlens.domain import (
    AgentRecommendation,
    CostEstimate,
    Coverage,
    DamageFinding,
    Decision,
    FraudSignal,
)
from claimlens.events.envelope import ClaimEvent
from claimlens.events.payloads import (
    AgentRecommended,
    ClaimReported,
    CostEstimated,
    DamageDetected,
    HumanReviewed,
    IntegrityChecked,
    LLMCalled,
    NoteAdded,
    Payload,
    PaymentIssued,
    PhotoAccepted,
    PhotoRejected,
    PhotoUploaded,
    PolicyRetrieved,
    QueueAssigned,
    RouteDecided,
    StageFailed,
    ToolCalled,
    parse_payload,
)


class PhotoStatus(StrEnum):
    UPLOADED = "uploaded"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class PhotoRecord:
    photo_id: str
    sha256: str
    filename: str
    blob_name: str
    status: PhotoStatus = PhotoStatus.UPLOADED
    reject_reason: str | None = None


@dataclass
class ClaimState:
    claim_id: UUID
    policy_id: str
    description: str
    photos: dict[str, PhotoRecord] = field(default_factory=dict)
    findings: list[DamageFinding] = field(default_factory=list)
    detection_event_ids: dict[str, str] = field(default_factory=dict)
    integrity_checked: bool = False
    fraud_signals: list[FraudSignal] = field(default_factory=list)
    cost_estimate: CostEstimate | None = None
    coverage: Coverage | None = None
    recommendation: AgentRecommendation | None = None
    decision: Decision | None = None
    failures: list[StageFailed] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    queue: str | None = None
    payments: list[str] = field(default_factory=list)
    review: HumanReviewed | None = None  # the latest human review
    last_seq: int = 0

    @property
    def accepted_photos(self) -> list[PhotoRecord]:
        return [p for p in self.photos.values() if p.status is PhotoStatus.ACCEPTED]

    def failed(self, stage: str, photo_id: str | None = None) -> bool:
        return any(f.stage == stage and f.photo_id == photo_id for f in self.failures)


def fold(events: Sequence[ClaimEvent]) -> ClaimState:
    if not events:
        raise ValueError("cannot fold an empty event list")
    first = parse_payload(events[0])
    if not isinstance(first, ClaimReported):
        raise ValueError("the first event of a claim must be ClaimReported")
    state = ClaimState(
        claim_id=events[0].claim_id, policy_id=first.policy_id, description=first.description
    )
    for event in events[1:]:
        _apply(state, event, parse_payload(event))
    state.last_seq = events[-1].seq
    return state


def _apply(state: ClaimState, event: ClaimEvent, payload: Payload) -> None:
    match payload:
        case PhotoUploaded():
            state.photos[payload.photo_id] = PhotoRecord(
                photo_id=payload.photo_id,
                sha256=payload.sha256,
                filename=payload.filename,
                blob_name=payload.blob_name,
            )
        case PhotoAccepted():
            state.photos[payload.photo_id].status = PhotoStatus.ACCEPTED
        case PhotoRejected():
            photo = state.photos[payload.photo_id]
            photo.status = PhotoStatus.REJECTED
            photo.reject_reason = payload.reason
        case DamageDetected():
            state.findings.extend(payload.findings)
            state.detection_event_ids[payload.photo_id] = str(event.event_id)
        case IntegrityChecked():
            state.integrity_checked = True
            state.fraud_signals.extend(payload.signals)
        case CostEstimated():
            state.cost_estimate = payload.estimate
        case PolicyRetrieved():
            state.coverage = payload.coverage
        case AgentRecommended():
            state.recommendation = payload.recommendation
        case RouteDecided():
            state.decision = payload.decision
        case StageFailed():
            state.failures.append(payload)
        case NoteAdded():
            state.notes.append(payload.text)
        case QueueAssigned():
            state.queue = payload.queue
        case PaymentIssued():
            state.payments.append(payload.payment_id)
        case HumanReviewed():
            state.review = payload
        case ToolCalled() | LLMCalled():
            pass
        case ClaimReported():
            raise ValueError("ClaimReported may only be the first event of a claim")
        case _:
            raise ValueError(f"no fold rule for event type {event.type}")


def find_by_idempotency_key(
    events: Sequence[ClaimEvent], key: str, event_type: str
) -> ClaimEvent | None:
    """The first event of this type recorded with this idempotency key, if any.

    Keys are scoped to the event type, so a note key can never swallow a queue or payment write.
    """
    for event in events:
        if event.type == event_type and event.payload.get("idempotency_key") == key:
            return event
    return None
