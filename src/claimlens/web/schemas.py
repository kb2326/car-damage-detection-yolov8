"""The web API's request and response models (they also feed the page templates)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from claimlens.domain import Route
from claimlens.events.payloads import ReviewAction

ClaimStatus = Literal["running", "stopped", "decided", "reviewed", "log_failed"]
StageState = Literal["done", "running", "failed", "waiting", "skipped"]


class ChatTurn(BaseModel):
    session_id: str
    message: str
    photo_kind: str | None = None  # the photo the assistant is waiting for
    claim_id: str | None = None  # set once the claim is filed


class StageStatus(BaseModel):
    key: str
    label: str
    state: StageState
    detail: str = ""


class Box(BaseModel):
    """A finding's box as percentages of the photo, so it scales with the image."""

    label: str
    left: float
    top: float
    width: float
    height: float


class PhotoView(BaseModel):
    photo_id: str
    kind: str | None
    status: str
    reject_reason: str | None = None
    boxes: list[Box] = Field(default_factory=list)


class AgentView(BaseModel):
    route_suggestion: str
    confidence: str
    rationale: str
    citations: list[str]
    policy_citations: list[str]
    open_questions: list[str]
    skills_used: list[str]
    tools_used: list[str]
    llm_calls: int
    llm_cost_usd: float


class SimilarView(BaseModel):
    claim_id: str
    reason: str
    distance: int | None
    route: str
    damage: str


class EventItem(BaseModel):
    seq: int
    type: str
    actor: str
    occurred_at: str


class ClaimSummary(BaseModel):
    claim_id: str
    policy_id: str
    filed_at: str
    status: ClaimStatus
    route: str | None = None  # what the rules decided
    rule_id: str | None = None
    review_action: str | None = None
    final_route: str | None = None  # the route that counts now (a reviewer's override wins)
    needs_review: bool = False  # waits for a person, as in `claimlens queue`


class ClaimDetail(ClaimSummary):
    description: str
    facts: dict[str, str]
    stages: list[StageStatus]
    photos: list[PhotoView]
    damage: list[str]
    cost_low: int | None
    cost_high: int | None
    coverage: str | None
    fraud_signals: list[str]
    agent: AgentView | None
    decision_reason: str | None
    similar: list[SimilarView]
    chain_ok: bool
    event_count: int
    events: list[EventItem]
    error: str | None = None  # a plain message; details stay in the server log
    stopped_at: str | None = None  # the stage a stopped claim did not finish


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    action: ReviewAction
    final_route: Route | None = None
    note: str = Field(default="", max_length=2000)
