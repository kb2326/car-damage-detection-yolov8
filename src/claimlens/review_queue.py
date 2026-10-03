"""The human review queue: what waits for a person, and what the person decided."""

from __future__ import annotations

from uuid import UUID

from claimlens.domain import Route
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import HumanReviewed, ReviewAction
from claimlens.events.projection import ClaimState, fold
from claimlens.events.store import SQLiteEventStore

_REVIEW_ROUTES = (Route.ADJUSTER_REVIEW, Route.FRAUD_REVIEW)
_CLOSING = (ReviewAction.APPROVE, ReviewAction.OVERRIDE, ReviewAction.DENY)


def pending_reviews(store: SQLiteEventStore, route: Route | None = None) -> list[ClaimState]:
    """Decided claims on a review route with no closing review, oldest first."""
    pending: list[tuple[str, ClaimState]] = []
    for claim_id in store.claim_ids():
        events = store.load(claim_id)
        state = fold(events)
        if state.decision is None or state.decision.route not in _REVIEW_ROUTES:
            continue
        if route is not None and state.decision.route is not route:
            continue
        if state.review is None or state.review.action not in _CLOSING:
            pending.append((events[0].occurred_at.isoformat(), state))
    return [state for _, state in sorted(pending, key=lambda item: item[0])]


def record_review(store: SQLiteEventStore, claim_id: UUID, review: HumanReviewed) -> int:
    if fold(store.load(claim_id)).decision is None:
        raise ValueError(f"claim {claim_id} has not been decided yet")
    event = store.append(claim_id, review, Actor(kind=ActorKind.HUMAN, name=review.reviewer))
    return event.seq


def payable(state: ClaimState) -> str | None:
    """None when a payment may be approved; otherwise the reason it may not."""
    if state.decision is None:
        return "the claim has not been decided"
    if state.decision.route is Route.FRAUD_REVIEW:
        return "fraud-routed claims are never paid here"
    review = state.review
    if review is not None and review.action is ReviewAction.DENY:
        return "denied by a reviewer"
    if state.decision.route is Route.FAST_TRACK:
        return None
    if review is not None and review.action in (ReviewAction.APPROVE, ReviewAction.OVERRIDE):
        return None
    return "waiting for human review"
