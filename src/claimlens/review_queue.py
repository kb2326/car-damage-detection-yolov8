"""The human review queue: what waits for a person, and what the person decided."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from claimlens.domain import Route
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import HumanReviewed, ReviewAction
from claimlens.events.projection import ClaimState, fold
from claimlens.events.store import SQLiteEventStore
from claimlens.memory.index import ClaimMemory
from claimlens.memory.writer import remember

_REVIEW_ROUTES = (Route.ADJUSTER_REVIEW, Route.FRAUD_REVIEW)
_CLOSING = (ReviewAction.APPROVE, ReviewAction.DENY)


def effective_route(state: ClaimState) -> Route | None:
    """The route that counts now: a reviewer's override wins over the rules' decision."""
    if state.decision is None:
        return None
    review = state.review
    if review is not None and review.action is ReviewAction.OVERRIDE:
        return review.final_route
    return state.decision.route


def pending_reviews(store: SQLiteEventStore, route: Route | None = None) -> list[ClaimState]:
    """Claims whose effective route needs a person and that no one has closed, oldest first.

    An override into a review route keeps the claim in the queue; approve or deny closes it.
    """
    pending: list[tuple[str, ClaimState]] = []
    for claim_id in store.claim_ids():
        events = store.load(claim_id)
        state = fold(events)
        current = effective_route(state)
        if current not in _REVIEW_ROUTES:
            continue
        if route is not None and current is not route:
            continue
        if state.review is None or state.review.action not in _CLOSING:
            pending.append((events[0].occurred_at.isoformat(), state))
    return [state for _, state in sorted(pending, key=lambda item: item[0])]


def record_review(
    store: SQLiteEventStore,
    claim_id: UUID,
    review: HumanReviewed,
    *,
    memory: ClaimMemory | None = None,
    photo_path: Callable[[str], Path] | None = None,
) -> int:
    if fold(store.load(claim_id)).decision is None:
        raise ValueError(f"claim {claim_id} has not been decided yet")
    event = store.append(claim_id, review, Actor(kind=ActorKind.HUMAN, name=review.reviewer))
    if memory is not None and photo_path is not None:
        remember(store, claim_id, memory, photo_path)
    return event.seq


def payable(state: ClaimState) -> str | None:
    """None when a payment may be approved; otherwise the reason it may not."""
    current = effective_route(state)
    if current is None:
        return "the claim has not been decided"
    # Fraud on either route blocks payment here: the rules' fraud route cannot be overridden
    # into a payout, and a reviewer's escalation to fraud stops one.
    if Route.FRAUD_REVIEW in (current, state.decision.route if state.decision else None):
        return "fraud-routed claims are never paid here"
    review = state.review
    if review is not None and review.action is ReviewAction.DENY:
        return "denied by a reviewer"
    if review is not None and review.action is ReviewAction.REQUEST_INFO:
        return "waiting for the information a reviewer asked for"
    if current is Route.FAST_TRACK:
        return None
    if review is not None and review.action is ReviewAction.APPROVE:
        return None
    return "waiting for human review"
