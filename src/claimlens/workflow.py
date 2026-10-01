"""Deterministic claim workflow. Each stage reads folded state and appends events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from uuid import UUID

from claimlens.agent import TriageAgent
from claimlens.blobs import BlobStore
from claimlens.decision import DecisionConfig, decide
from claimlens.domain import Decision
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import (
    AgentRecommended,
    CostEstimated,
    DamageDetected,
    IntegrityChecked,
    Payload,
    PhotoAccepted,
    PhotoRejected,
    PolicyRetrieved,
    RouteDecided,
    StageFailed,
)
from claimlens.events.projection import ClaimState, PhotoStatus, fold
from claimlens.events.store import SQLiteEventStore
from claimlens.integrity import check_integrity
from claimlens.policy import PolicyRepository
from claimlens.pricing import RateCard, estimate_cost
from claimlens.quality import QualityConfig, check_quality
from claimlens.vision.base import Detector

WORKFLOW = Actor(kind=ActorKind.SYSTEM, name="workflow")
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class PipelineDeps:
    store: SQLiteEventStore
    blobs: BlobStore
    detector: Detector
    policies: PolicyRepository
    rate_card: RateCard
    decision_config: DecisionConfig
    agent: TriageAgent
    quality: QualityConfig = field(default_factory=QualityConfig)


def process_claim(claim_id: UUID, deps: PipelineDeps) -> Decision:
    """Run every unfinished stage, then decide. Safe to call again after any interruption."""
    state = _load(deps, claim_id)
    if state.decision is not None:
        return state.decision
    stages: tuple[Callable[[ClaimState, PipelineDeps], None], ...] = (
        _quality_stage,
        _perception_stage,
        _integrity_stage,
        _pricing_stage,
        _coverage_stage,
        _agent_stage,
    )
    for stage in stages:
        stage(_load(deps, claim_id), deps)
    decision = decide(_load(deps, claim_id), deps.decision_config)
    deps.store.append(claim_id, RouteDecided(decision=decision), WORKFLOW)
    return decision


def _load(deps: PipelineDeps, claim_id: UUID) -> ClaimState:
    return fold(deps.store.load(claim_id))


def _attempt[T](call: Callable[[], T]) -> T | Exception:
    last: Exception = RuntimeError("no attempt was made")
    for _ in range(MAX_ATTEMPTS):
        try:
            return call()
        except Exception as exc:
            last = exc
    return last


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _quality_stage(state: ClaimState, deps: PipelineDeps) -> None:
    seen = {p.sha256: p.photo_id for p in state.photos.values() if p.status is PhotoStatus.ACCEPTED}
    for photo in state.photos.values():
        if photo.status is not PhotoStatus.UPLOADED:
            continue
        payload: Payload
        if photo.sha256 in seen:
            payload = PhotoRejected(
                photo_id=photo.photo_id, reason=f"duplicate of {seen[photo.sha256]}"
            )
        else:
            result = check_quality(deps.blobs.path(photo.blob_name), deps.quality)
            if result.ok:
                payload = PhotoAccepted(
                    photo_id=photo.photo_id, width=result.width, height=result.height
                )
                seen[photo.sha256] = photo.photo_id
            else:
                payload = PhotoRejected(photo_id=photo.photo_id, reason=result.reason)
        deps.store.append(state.claim_id, payload, WORKFLOW)


def _perception_stage(state: ClaimState, deps: PipelineDeps) -> None:
    for photo in state.accepted_photos:
        done = photo.photo_id in state.detection_event_ids
        if done or state.failed("perception", photo.photo_id):
            continue
        image = deps.blobs.path(photo.blob_name)
        outcome = _attempt(partial(deps.detector.detect, image, photo.photo_id))
        payload: Payload
        if isinstance(outcome, Exception):
            payload = StageFailed(
                stage="perception", photo_id=photo.photo_id, error=_describe(outcome)
            )
        else:
            payload = DamageDetected(
                photo_id=photo.photo_id,
                model_version=deps.detector.model_version,
                findings=tuple(outcome),
            )
        deps.store.append(state.claim_id, payload, WORKFLOW)


def _integrity_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if not state.integrity_checked:
        signals = check_integrity(state, deps.store)
        deps.store.append(state.claim_id, IntegrityChecked(signals=signals), WORKFLOW)


def _pricing_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if state.cost_estimate is None:
        estimate = estimate_cost(state.findings, deps.rate_card)
        deps.store.append(state.claim_id, CostEstimated(estimate=estimate), WORKFLOW)


def _coverage_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if state.coverage is None:
        coverage = deps.policies.get_coverage(state.policy_id)
        deps.store.append(state.claim_id, PolicyRetrieved(coverage=coverage), WORKFLOW)


def _agent_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if state.recommendation is not None or state.failed("agent"):
        return
    outcome = _attempt(partial(deps.agent.recommend, state))
    if isinstance(outcome, Exception):
        deps.store.append(
            state.claim_id, StageFailed(stage="agent", error=_describe(outcome)), WORKFLOW
        )
        return
    actor = Actor(kind=ActorKind.AGENT, name=deps.agent.agent_version)
    deps.store.append(
        state.claim_id,
        AgentRecommended(agent_version=deps.agent.agent_version, recommendation=outcome),
        actor,
    )
