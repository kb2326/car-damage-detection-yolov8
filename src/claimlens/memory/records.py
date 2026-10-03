"""Memory records: what the system remembers about a decided claim (no customer words)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import imagehash
from PIL import Image

from claimlens.domain import Frozen
from claimlens.events.envelope import ClaimEvent
from claimlens.events.projection import ClaimState

# Events a record is built from; their sequence numbers are the record's provenance.
_SOURCES = ("DamageDetected", "CostEstimated", "IntakeCompleted", "RouteDecided", "HumanReviewed")
# Intake facts safe to remember: yes/no answers only, never free text.
_YES_NO_FACTS = ("driving_for_work", "other_party_involved", "injuries")


class MemoryRecord(Frozen):
    claim_id: str
    policy_id: str
    decided_on: str
    route: str
    rule_id: str
    review_action: str = ""
    final_route: str = ""
    damage: str
    cost_low: int
    cost_high: int
    photo_phashes: tuple[str, ...]
    summary: str
    source_seq: tuple[int, ...]


def phash_of(path: Path) -> str:
    """64-bit perceptual hash as 16 hex characters (the data pipeline's dedupe hash)."""
    with Image.open(path) as image:
        return str(imagehash.phash(image))


def build_record(
    state: ClaimState, events: Sequence[ClaimEvent], photo_path: Callable[[str], Path]
) -> MemoryRecord:
    """The memory record for a decided claim. `photo_path` maps a blob name to its file."""
    if state.decision is None:
        raise ValueError("only decided claims are remembered")
    phashes = tuple(phash_of(photo_path(p.blob_name)) for p in state.accepted_photos)
    damage = "; ".join(
        sorted({f.damage_type.value + (f" on {f.part}" if f.part else "") for f in state.findings})
    )
    estimate = state.cost_estimate
    low, high = (estimate.low, estimate.high) if estimate else (0, 0)
    review = state.review
    facts = state.intake.facts if state.intake else {}
    parts = [damage or "no damage found", f"estimate ${low}-${high}"]
    parts += [f"{name}: {facts[name]}" for name in _YES_NO_FACTS if name in facts]
    parts.append(f"routed {state.decision.route.value} ({state.decision.rule_id})")
    if review is not None:
        parts.append(f"reviewed: {review.action.value}")
    decided_on = next(
        (e.occurred_at.date().isoformat() for e in events if e.type == "RouteDecided"), ""
    )
    return MemoryRecord(
        claim_id=str(state.claim_id),
        policy_id=state.policy_id,
        decided_on=decided_on,
        route=state.decision.route.value,
        rule_id=state.decision.rule_id,
        review_action=review.action.value if review else "",
        final_route=review.final_route.value if review and review.final_route else "",
        damage=damage,
        cost_low=low,
        cost_high=high,
        photo_phashes=phashes,
        summary="; ".join(parts),
        source_seq=tuple(e.seq for e in events if e.type in _SOURCES),
    )
