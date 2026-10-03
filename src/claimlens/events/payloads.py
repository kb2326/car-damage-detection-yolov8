"""One Pydantic model per event type. The payload is the event's business content."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from pydantic import model_validator

from claimlens.domain import (
    AgentRecommendation,
    CostEstimate,
    Coverage,
    DamageFinding,
    Decision,
    FraudSignal,
    Frozen,
    Route,
)
from claimlens.events.envelope import ClaimEvent


class EventType(StrEnum):
    CLAIM_REPORTED = "ClaimReported"
    PHOTO_UPLOADED = "PhotoUploaded"
    PHOTO_ACCEPTED = "PhotoAccepted"
    PHOTO_REJECTED = "PhotoRejected"
    DAMAGE_DETECTED = "DamageDetected"
    INTEGRITY_CHECKED = "IntegrityChecked"
    COST_ESTIMATED = "CostEstimated"
    POLICY_RETRIEVED = "PolicyRetrieved"
    AGENT_RECOMMENDED = "AgentRecommended"
    ROUTE_DECIDED = "RouteDecided"
    STAGE_FAILED = "StageFailed"
    NOTE_ADDED = "NoteAdded"
    QUEUE_ASSIGNED = "QueueAssigned"
    PAYMENT_ISSUED = "PaymentIssued"
    TOOL_CALLED = "ToolCalled"
    LLM_CALLED = "LLMCalled"
    HUMAN_REVIEWED = "HumanReviewed"
    INTAKE_COMPLETED = "IntakeCompleted"


class Payload(Frozen):
    event_type: ClassVar[EventType]


class ClaimReported(Payload):
    event_type: ClassVar[EventType] = EventType.CLAIM_REPORTED
    policy_id: str
    description: str


class PhotoUploaded(Payload):
    event_type: ClassVar[EventType] = EventType.PHOTO_UPLOADED
    photo_id: str
    sha256: str
    filename: str
    blob_name: str
    size_bytes: int


class PhotoAccepted(Payload):
    event_type: ClassVar[EventType] = EventType.PHOTO_ACCEPTED
    photo_id: str
    width: int
    height: int


class PhotoRejected(Payload):
    event_type: ClassVar[EventType] = EventType.PHOTO_REJECTED
    photo_id: str
    reason: str


class DamageDetected(Payload):
    event_type: ClassVar[EventType] = EventType.DAMAGE_DETECTED
    photo_id: str
    model_version: str
    findings: tuple[DamageFinding, ...]


class IntegrityChecked(Payload):
    event_type: ClassVar[EventType] = EventType.INTEGRITY_CHECKED
    signals: tuple[FraudSignal, ...]


class CostEstimated(Payload):
    event_type: ClassVar[EventType] = EventType.COST_ESTIMATED
    estimate: CostEstimate


class PolicyRetrieved(Payload):
    event_type: ClassVar[EventType] = EventType.POLICY_RETRIEVED
    coverage: Coverage


class AgentRecommended(Payload):
    event_type: ClassVar[EventType] = EventType.AGENT_RECOMMENDED
    agent_version: str
    recommendation: AgentRecommendation


class ReviewAction(StrEnum):
    APPROVE = "approve"
    OVERRIDE = "override"
    DENY = "deny"  # only a person can deny; no model, agent or rule produces this
    REQUEST_INFO = "request_info"


class IntakeCompleted(Payload):
    """What the intake conversation collected. The transcript itself stays out of the log."""

    event_type: ClassVar[EventType] = EventType.INTAKE_COMPLETED
    session_id: str
    facts: dict[str, str]
    photo_kinds: dict[str, str]  # kind -> photo id
    photo_gaps: dict[str, str]  # kind -> why it is missing
    turns: int
    retakes: int
    transcript_sha256: str


class HumanReviewed(Payload):
    """What a person decided about a claim on a review route (or any decided claim)."""

    event_type: ClassVar[EventType] = EventType.HUMAN_REVIEWED
    reviewer: str
    action: ReviewAction
    final_route: Route | None = None
    note: str = ""

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if not self.reviewer.strip():
            raise ValueError("a review needs the reviewer's name")
        if self.action is ReviewAction.OVERRIDE and self.final_route is None:
            raise ValueError("an override needs a final_route")
        if self.action is not ReviewAction.OVERRIDE and self.final_route is not None:
            raise ValueError("final_route is only for an override")
        if self.action in (ReviewAction.OVERRIDE, ReviewAction.DENY) and not self.note.strip():
            raise ValueError("an override or a denial needs a note")
        return self


class RouteDecided(Payload):
    event_type: ClassVar[EventType] = EventType.ROUTE_DECIDED
    decision: Decision


class StageFailed(Payload):
    event_type: ClassVar[EventType] = EventType.STAGE_FAILED
    stage: str
    error: str
    photo_id: str | None = None


class NoteAdded(Payload):
    event_type: ClassVar[EventType] = EventType.NOTE_ADDED
    text: str
    author: str
    idempotency_key: str


class QueueAssigned(Payload):
    event_type: ClassVar[EventType] = EventType.QUEUE_ASSIGNED
    queue: str
    idempotency_key: str


class PaymentIssued(Payload):
    event_type: ClassVar[EventType] = EventType.PAYMENT_ISSUED
    payment_id: str
    amount_usd: int
    idempotency_key: str
    approval_sha256: str = ""


class ToolCalled(Payload):
    event_type: ClassVar[EventType] = EventType.TOOL_CALLED
    server: str
    tool: str
    profile: str
    input_sha256: str
    outcome: str


class LLMCalled(Payload):
    event_type: ClassVar[EventType] = EventType.LLM_CALLED
    request_id: str
    model: str
    prompt_id: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    outcome: str


PAYLOAD_TYPES: dict[EventType, type[Payload]] = {
    cls.event_type: cls
    for cls in (
        ClaimReported,
        PhotoUploaded,
        PhotoAccepted,
        PhotoRejected,
        DamageDetected,
        IntegrityChecked,
        CostEstimated,
        PolicyRetrieved,
        AgentRecommended,
        RouteDecided,
        StageFailed,
        NoteAdded,
        QueueAssigned,
        PaymentIssued,
        ToolCalled,
        LLMCalled,
        HumanReviewed,
        IntakeCompleted,
    )
}


def parse_payload(event: ClaimEvent) -> Payload:
    """Validate an event's payload against the model for its type."""
    return PAYLOAD_TYPES[EventType(event.type)].model_validate(event.payload)
