"""Pure functions from a claim's events to what the pages show. No web, no I/O."""

from __future__ import annotations

from collections.abc import Sequence

from claimlens.events.envelope import ClaimEvent
from claimlens.events.projection import ClaimState, PhotoStatus, fold
from claimlens.web.schemas import (
    AgentView,
    Box,
    ClaimDetail,
    ClaimStatus,
    ClaimSummary,
    EventItem,
    PhotoView,
    SimilarView,
    StageState,
    StageStatus,
)

STAGES: tuple[tuple[str, str], ...] = (
    ("intake", "Intake chat"),
    ("photos", "Photo check"),
    ("damage", "Damage and parts"),
    ("integrity", "Integrity"),
    ("pricing", "Pricing"),
    ("coverage", "Coverage"),
    ("agent", "Triage agent"),
    ("decision", "Decision"),
    ("memory", "Memory"),
    ("review", "Human review"),
)
# Stages done by one event type, and the StageFailed stage name that marks them failed.
_ONE_EVENT: dict[str, tuple[str, str | None]] = {
    "integrity": ("IntegrityChecked", "integrity"),
    "pricing": ("CostEstimated", "pricing"),
    "coverage": ("PolicyRetrieved", "coverage"),
    "agent": ("AgentRecommended", "agent"),
    "decision": ("RouteDecided", None),
}
# Stages the pipeline itself runs, in order: the first unfinished one is "running".
_PIPELINE = ("photos", "damage", "integrity", "pricing", "coverage", "agent", "decision")
Stage = tuple[StageState, str]


def claim_status(state: ClaimState, running: bool) -> ClaimStatus:
    if running:
        return "running"
    if state.review is not None:
        return "reviewed"
    if state.decision is not None:
        return "decided"
    return "stopped"


def _failures(events: Sequence[ClaimEvent]) -> dict[str, str]:
    return {
        str(e.payload["stage"]): str(e.payload["error"]) for e in events if e.type == "StageFailed"
    }


def _photos_stage(state: ClaimState) -> Stage:
    if not state.photos:
        return "skipped", "no photos (rule R3 sends the claim to a person)"
    checked = [p for p in state.photos.values() if p.status is not PhotoStatus.UPLOADED]
    if len(checked) < len(state.photos):
        return "waiting", ""
    accepted = len(state.accepted_photos)
    return "done", f"{accepted} accepted, {len(state.photos) - accepted} rejected"


def _damage_stage(state: ClaimState, failures: dict[str, str]) -> Stage:
    if "perception" in failures:
        return "failed", failures["perception"]
    if not state.photos:
        return "skipped", ""
    accepted = state.accepted_photos
    if accepted and all(p.photo_id in state.detection_event_ids for p in accepted):
        return "done", f"{len(state.findings)} finding(s)"
    if all(p.status is PhotoStatus.REJECTED for p in state.photos.values()):
        return "skipped", "no accepted photo"
    return "waiting", ""


def _memory_stage(state: ClaimState, types: set[str]) -> Stage:
    if "MemoryForgotten" in types:
        return "skipped", "forgotten"
    if "MemoryWritten" in types:
        return "done", ""
    if "MemoryWriteFailed" in types:
        return "failed", "the memory write failed; the decision is unchanged"
    if state.decision is not None:
        return "skipped", "memory is off"
    return "waiting", ""


def stage_timeline(events: Sequence[ClaimEvent], running: bool) -> list[StageStatus]:
    state = fold(events)
    types = {e.type for e in events}
    failures = _failures(events)
    intake: Stage = ("skipped", "filed by form")
    if state.intake is not None:
        intake = ("done", f"{state.intake.turns} turns")
    result: dict[str, Stage] = {
        "intake": intake,
        "photos": _photos_stage(state),
        "damage": _damage_stage(state, failures),
        "memory": _memory_stage(state, types),
        "review": ("done", state.review.action.value) if state.review else ("waiting", ""),
    }
    for key, (done_type, failed_stage) in _ONE_EVENT.items():
        if done_type in types:
            detail = ""
            if key == "decision" and state.decision is not None:
                detail = f"{state.decision.route.value} ({state.decision.rule_id})"
            result[key] = ("done", detail)
        elif failed_stage is not None and failed_stage in failures:
            result[key] = ("failed", failures[failed_stage])
        else:
            result[key] = ("waiting", "")
    if running:
        for key in _PIPELINE:
            if result[key][0] == "waiting":
                result[key] = ("running", "")
                break
    return [
        StageStatus(key=key, label=label, state=result[key][0], detail=result[key][1])
        for key, label in STAGES
    ]


def photo_views(state: ClaimState, events: Sequence[ClaimEvent]) -> list[PhotoView]:
    sizes = {
        str(e.payload["photo_id"]): (int(e.payload["width"]), int(e.payload["height"]))
        for e in events
        if e.type == "PhotoAccepted"
    }
    kinds = {pid: kind for kind, pid in state.intake.photo_kinds.items()} if state.intake else {}
    views = []
    for photo in state.photos.values():
        boxes = []
        size = sizes.get(photo.photo_id)
        for f in state.findings:
            if f.photo_id != photo.photo_id or size is None:
                continue
            w, h = size
            label = " · ".join(
                [f.damage_type.value, *([f.part] if f.part else []), f"{f.confidence:.2f}"]
            )
            boxes.append(
                Box(
                    label=label,
                    left=round(100 * f.bbox.x1 / w, 2),
                    top=round(100 * f.bbox.y1 / h, 2),
                    width=round(100 * (f.bbox.x2 - f.bbox.x1) / w, 2),
                    height=round(100 * (f.bbox.y2 - f.bbox.y1) / h, 2),
                )
            )
        views.append(
            PhotoView(
                photo_id=photo.photo_id,
                kind=kinds.get(photo.photo_id),
                status=photo.status.value,
                reject_reason=photo.reject_reason,
                boxes=boxes,
            )
        )
    return views


def agent_view(state: ClaimState, events: Sequence[ClaimEvent]) -> AgentView | None:
    rec = state.recommendation
    if rec is None:
        return None
    tools = {str(e.payload["tool"]) for e in events if e.type == "ToolCalled"}
    llm = [e for e in events if e.type == "LLMCalled"]
    return AgentView(
        route_suggestion=rec.route_suggestion.value,
        confidence=rec.confidence.value,
        rationale=rec.rationale,
        citations=list(rec.citations),
        policy_citations=list(rec.policy_citations),
        open_questions=list(rec.open_questions),
        skills_used=list(rec.skills_used),
        tools_used=sorted(tools),
        llm_calls=len(llm),
        llm_cost_usd=round(sum(float(e.payload["cost_usd"]) for e in llm), 4),
    )


def claim_summary(state: ClaimState, events: Sequence[ClaimEvent], running: bool) -> ClaimSummary:
    return ClaimSummary(
        claim_id=str(state.claim_id),
        policy_id=state.policy_id,
        filed_at=events[0].occurred_at.isoformat(timespec="seconds"),
        status=claim_status(state, running),
        route=state.decision.route.value if state.decision else None,
        rule_id=state.decision.rule_id if state.decision else None,
        review_action=state.review.action.value if state.review else None,
    )


def _coverage_text(state: ClaimState) -> str | None:
    cover = state.coverage
    if cover is None:
        return None
    if not cover.found:
        return "policy not found"
    status = "active" if cover.active else "inactive"
    collision = "collision cover" if cover.collision else "no collision cover"
    return f"{status}, {collision}, ${cover.deductible} deductible"


def claim_detail(
    state: ClaimState,
    events: Sequence[ClaimEvent],
    *,
    running: bool,
    error: str | None,
    similar: list[SimilarView],
) -> ClaimDetail:
    summary = claim_summary(state, events, running)
    return ClaimDetail(
        **summary.model_dump(),
        description=state.description,
        facts=dict(state.intake.facts) if state.intake else {},
        stages=stage_timeline(events, running),
        photos=photo_views(state, events),
        damage=[
            f"{f.damage_type.value}{' on ' + f.part if f.part else ''} ({f.confidence:.2f})"
            for f in state.findings
        ],
        cost_low=state.cost_estimate.low if state.cost_estimate else None,
        cost_high=state.cost_estimate.high if state.cost_estimate else None,
        coverage=_coverage_text(state),
        fraud_signals=[f"{s.kind} ({s.score:.2f}): {s.detail}" for s in state.fraud_signals],
        agent=agent_view(state, events),
        decision_reason=state.decision.reason if state.decision else None,
        similar=similar,
        chain_ok=True,  # the store verified the chain when it loaded these events
        event_count=len(events),
        events=[
            EventItem(
                seq=e.seq,
                type=e.type,
                actor=f"{e.actor.kind.value}:{e.actor.name}",
                occurred_at=e.occurred_at.isoformat(timespec="seconds"),
            )
            for e in events
        ],
        error=error,
    )
