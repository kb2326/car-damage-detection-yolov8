from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.domain import Route
from claimlens.events.payloads import HumanReviewed, ReviewAction
from claimlens.review_queue import payable, pending_reviews, record_review
from tests.fakes import decided_claim, undecided_claim


def test_override_needs_a_route_and_a_note() -> None:
    with pytest.raises(ValidationError, match="final_route"):
        HumanReviewed(reviewer="sam", action=ReviewAction.OVERRIDE, note="looks fine")
    with pytest.raises(ValidationError, match="note"):
        HumanReviewed(reviewer="sam", action=ReviewAction.DENY)
    with pytest.raises(ValidationError, match="final_route"):
        HumanReviewed(reviewer="sam", action=ReviewAction.APPROVE, final_route=Route.FAST_TRACK)


def test_queue_lists_review_routes_until_reviewed(tmp_path: Path) -> None:
    store, claim = decided_claim(tmp_path, Route.ADJUSTER_REVIEW)
    assert [s.claim_id for s in pending_reviews(store)] == [claim]
    assert pending_reviews(store, Route.FRAUD_REVIEW) == []
    record_review(store, claim, HumanReviewed(reviewer="sam", action=ReviewAction.APPROVE))
    assert pending_reviews(store) == []


def test_request_info_keeps_the_claim_in_the_queue(tmp_path: Path) -> None:
    store, claim = decided_claim(tmp_path, Route.ADJUSTER_REVIEW)
    record_review(
        store,
        claim,
        HumanReviewed(reviewer="sam", action=ReviewAction.REQUEST_INFO, note="plate photo"),
    )
    assert [s.claim_id for s in pending_reviews(store)] == [claim]


def test_fast_track_claims_are_not_in_the_queue(tmp_path: Path) -> None:
    store, _ = decided_claim(tmp_path, Route.FAST_TRACK)
    assert pending_reviews(store) == []


def test_an_undecided_claim_cannot_be_reviewed(tmp_path: Path) -> None:
    store, claim = undecided_claim(tmp_path)
    with pytest.raises(ValueError, match="not been decided"):
        record_review(store, claim, HumanReviewed(reviewer="sam", action=ReviewAction.APPROVE))


def test_the_review_is_a_human_event_on_the_chain(tmp_path: Path) -> None:
    store, claim = decided_claim(tmp_path, Route.ADJUSTER_REVIEW)
    record_review(
        store,
        claim,
        HumanReviewed(reviewer="sam", action=ReviewAction.DENY, note="rust predates policy"),
    )
    last = store.load(claim)[-1]
    assert last.type == "HumanReviewed"
    assert last.actor.kind == "human"
    assert last.actor.name == "sam"


@pytest.mark.parametrize(
    ("route", "review", "reason"),
    [
        (Route.FAST_TRACK, None, None),
        (Route.ADJUSTER_REVIEW, None, "waiting for human review"),
        (Route.ADJUSTER_REVIEW, ("approve", None), None),
        (Route.ADJUSTER_REVIEW, ("deny", None), "denied by a reviewer"),
        (Route.ADJUSTER_REVIEW, ("request_info", None), "waiting for human review"),
        (Route.FAST_TRACK, ("deny", None), "denied by a reviewer"),
        (Route.FRAUD_REVIEW, ("approve", None), "fraud-routed claims are never paid here"),
        (
            Route.FRAUD_REVIEW,
            ("override", Route.FAST_TRACK),
            "fraud-routed claims are never paid here",
        ),
    ],
)
def test_payable(
    tmp_path: Path,
    route: Route,
    review: tuple[str, Route | None] | None,
    reason: str | None,
) -> None:
    store, claim = decided_claim(tmp_path, route)
    if review is not None:
        action, final = review
        record_review(
            store,
            claim,
            HumanReviewed(
                reviewer="sam",
                action=ReviewAction(action),
                final_route=final,
                note="n" if action in {"deny", "override"} else "",
            ),
        )
    from claimlens.events.projection import fold

    assert payable(fold(store.load(claim))) == reason
