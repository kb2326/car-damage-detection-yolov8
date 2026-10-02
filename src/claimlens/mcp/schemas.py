"""Typed tool outputs: what each MCP tool returns to the client."""

from __future__ import annotations

from pydantic import BaseModel


class FindingOut(BaseModel):
    type: str
    confidence: float
    part: str | None
    part_area_ratio: float | None
    severity: str


class DamageReport(BaseModel):
    model_version: str
    findings: list[FindingOut]


class PartOut(BaseModel):
    label: str
    group: str
    confidence: float


class PartsReport(BaseModel):
    model_version: str
    parts: list[PartOut]


class QualityReport(BaseModel):
    accepted: bool
    reason: str


class PolicySummary(BaseModel):
    found: bool
    policy_id: str
    status: str | None = None
    collision: bool | None = None
    deductible: int | None = None


class CoverageResult(BaseModel):
    found: bool
    active: bool
    collision: bool
    deductible: int


class EventRef(BaseModel):
    seq: int
    type: str


class ClaimHistory(BaseModel):
    found: bool
    claim_id: str
    policy_id: str | None = None
    route: str | None = None
    findings: list[str] = []
    notes: list[str] = []
    queue: str | None = None
    events: list[EventRef] = []


class SimilarClaim(BaseModel):
    claim_id: str
    reason: str


class SimilarClaims(BaseModel):
    items: list[SimilarClaim]


class WriteResult(BaseModel):
    event_seq: int
    duplicate: bool


class PaymentResult(BaseModel):
    payment_id: str
    event_seq: int
    duplicate: bool


class ClauseOut(BaseModel):
    clause_id: str
    title: str
    text: str
    score: float


class ClauseResults(BaseModel):
    wording: str | None
    clauses: list[ClauseOut]
