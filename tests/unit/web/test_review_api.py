import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from claimlens.intake import submit_claim
from claimlens.web.services import WebServices
from tests.unit.web.helpers import decided_claim, image
from tests.unit.web.test_claims_api import TAMPER


def _decided(services: WebServices, tmp_path: Path) -> str:
    with services.open_store() as store:
        return str(decided_claim(tmp_path, store))


def _review(client: TestClient, claim: str, body: dict[str, str]) -> int:
    return client.post(f"/api/claims/{claim}/review", json=body).status_code


def test_a_person_approves(client: TestClient, services: WebServices, tmp_path: Path) -> None:
    claim = _decided(services, tmp_path)
    assert _review(client, claim, {"reviewer": "Sam", "action": "approve", "note": "fine"}) == 201
    detail = client.get(f"/api/claims/{claim}").json()
    assert detail["status"] == "reviewed"
    assert detail["review_action"] == "approve"


def test_a_person_denies(client: TestClient, services: WebServices, tmp_path: Path) -> None:
    claim = _decided(services, tmp_path)
    assert _review(client, claim, {"reviewer": "Sam", "action": "deny", "note": "staged"}) == 201
    assert client.get(f"/api/claims/{claim}").json()["review_action"] == "deny"


def test_an_override_needs_a_final_route(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    response = client.post(
        f"/api/claims/{claim}/review", json={"reviewer": "Sam", "action": "override"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]
    body = {
        "reviewer": "Sam",
        "action": "override",
        "final_route": "FRAUD_REVIEW",
        "note": "same photo as an earlier claim",
    }
    assert _review(client, claim, body) == 201


def test_a_blank_reviewer_or_an_unknown_action_is_422(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    assert _review(client, claim, {"reviewer": "", "action": "approve"}) == 422
    assert _review(client, claim, {"reviewer": "   ", "action": "approve"}) == 422
    assert _review(client, claim, {"reviewer": "Sam", "action": "pay"}) == 422
    assert _review(client, claim, {"reviewer": "Sam", "action": "deny"}) == 422  # needs a note


def test_an_undecided_claim_cannot_be_reviewed(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    with services.open_store() as store:
        claim = submit_claim(
            store,
            services.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[image(tmp_path / "in" / "u.png")],
        )
    assert _review(client, str(claim), {"reviewer": "Sam", "action": "approve"}) == 409


def test_a_tampered_claim_cannot_be_reviewed(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    with sqlite3.connect(services.db_path) as conn:
        conn.execute(TAMPER)
    assert _review(client, claim, {"reviewer": "Sam", "action": "approve"}) == 409
