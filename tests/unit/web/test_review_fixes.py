"""Regression tests for the final M8a review findings."""

import io
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

from claimlens.intake import submit_claim
from claimlens.llm.provider import ProviderFatalError, ProviderReply
from claimlens.web.services import WebServices
from claimlens.web.uploads import check_upload
from tests.unit.test_intake_session import SCRIPT
from tests.unit.web.helpers import decided_claim, image
from tests.unit.web.test_intake_api import _chat, _photo

OUTAGE_BEFORE_FINISH: list[ProviderReply | Exception] = [
    *SCRIPT[:5],
    ProviderFatalError("overloaded"),
    SCRIPT[5],
]


def _to_last_photo(client: TestClient) -> tuple[str, int]:
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001, bollard yesterday"})
    _photo(client, session, 1)
    _photo(client, session, 2)
    files = {"photo": ("p3.png", _png3(), "image/png")}
    response = client.post(f"/api/intake/{session}/reply", data={"text": ""}, files=files)
    return session, response.status_code


def _png3() -> bytes:
    from tests.unit.web.test_intake_api import _png

    return _png(3)


# 1. An LLM outage during intake: retry or reload finishes the hand-over and runs the claim.


def test_after_an_outage_sending_again_resumes_and_runs_the_claim(
    services: WebServices, tmp_path: Path
) -> None:
    client = _chat(services, tmp_path, list(OUTAGE_BEFORE_FINISH))
    session, status = _to_last_photo(client)
    assert status == 503
    retry = client.post(f"/api/intake/{session}/reply", data={"text": "hello?"})
    assert retry.status_code == 200
    claim = retry.json()["claim_id"]
    assert claim is not None
    assert client.get(f"/api/claims/{claim}").json()["status"] == "decided"


def test_after_an_outage_a_reload_resumes_and_runs_the_claim(
    services: WebServices, tmp_path: Path
) -> None:
    client = _chat(services, tmp_path, list(OUTAGE_BEFORE_FINISH))
    session, status = _to_last_photo(client)
    assert status == 503
    claim = client.get(f"/api/intake/{session}").json()["claim_id"]
    assert claim is not None
    assert client.get(f"/api/claims/{claim}").json()["status"] == "decided"


# 7. A reload after the hand-over shows the reference again.


def test_a_reload_after_the_hand_over_shows_the_reference(
    services: WebServices, tmp_path: Path
) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    last = [_photo(client, session, seed) for seed in (1, 2, 3)][-1]
    again = client.get(f"/api/intake/{session}")
    assert again.status_code == 200
    assert again.json()["claim_id"] == last["claim_id"]
    assert again.json()["message"] == last["message"]


# 2. A reviewer's override counts in the lists, as in the review queue.


def _decided(services: WebServices, tmp_path: Path) -> str:
    with services.open_store() as store:
        return str(decided_claim(tmp_path, store))


def test_an_override_into_fraud_review_shows_in_both_queues(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    body = {
        "reviewer": "Sam",
        "action": "override",
        "final_route": "FRAUD_REVIEW",
        "note": "same photo as before",
    }
    assert client.post(f"/api/claims/{claim}/review", json=body).status_code == 201
    (row,) = client.get("/api/claims").json()
    assert row["final_route"] == "FRAUD_REVIEW"
    assert row["needs_review"] is True
    assert claim in client.get("/claims?filter=fraud").text
    assert claim in client.get("/claims?filter=review").text
    closing = {"reviewer": "Sam", "action": "approve"}
    assert client.post(f"/api/claims/{claim}/review", json=closing).status_code == 201
    assert claim not in client.get("/claims?filter=review").text


# 3. Host and Origin checks; nothing is saved for an unknown chat.


def test_a_foreign_host_is_refused(client: TestClient) -> None:
    assert client.get("/api/claims", headers={"Host": "evil.example:8000"}).status_code == 400


def test_a_cross_site_post_is_refused(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    body = {"reviewer": "x", "action": "deny", "note": "n"}
    headers = {"Origin": "http://evil.example"}
    response = client.post(f"/api/claims/{claim}/review", json=body, headers=headers)
    assert response.status_code == 403
    same = {"Origin": "http://testserver"}
    ok = {"reviewer": "Sam", "action": "approve"}
    assert client.post(f"/api/claims/{claim}/review", json=ok, headers=same).status_code == 201


def test_an_upload_to_an_unknown_chat_saves_nothing(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    files = {"photo": ("p.png", _png3(), "image/png")}
    response = client.post("/api/intake/nope/reply", data={"text": ""}, files=files)
    assert response.status_code == 404
    assert not services.upload_dir.exists() or not any(services.upload_dir.iterdir())


# 5. Phone JPEGs that Pillow reads as MPO are accepted.


def test_a_multi_picture_jpeg_is_accepted(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    first, second = Image.new("RGB", (64, 48), (9, 9, 9)), Image.new("RGB", (32, 24))
    first.save(buffer, "MPO", save_all=True, append_images=[second])
    kept = check_upload(buffer.getvalue(), 1_000_000, tmp_path)
    assert kept.suffix == ".jpg"


# 9. A pipeline error never shows its text (paths) in the browser; the stage is named.


def test_a_stopped_claim_names_the_stage_not_the_error(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    with services.open_store() as store:
        claim: UUID = submit_claim(
            store,
            services.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[image(tmp_path / "in" / "e.png")],
        )

    def boom(claim_id: UUID) -> None:
        raise FileNotFoundError(r"C:\secret\weights\damage.onnx")

    services.runner._process = boom
    services.runner.start(claim)
    detail = client.get(f"/api/claims/{claim}").json()
    assert detail["status"] == "stopped"
    assert "secret" not in str(detail)
    assert detail["stopped_at"] == "Photo check"
    page = client.get(f"/claims/{claim}").text
    assert "secret" not in page
    assert "Photo check" in page


# 11. No review while the pipeline is still working on the claim.


def test_a_review_while_the_claim_is_running_is_409(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    services.runner._running.add(UUID(claim))
    body = {"reviewer": "Sam", "action": "approve"}
    assert client.post(f"/api/claims/{claim}/review", json=body).status_code == 409


def test_the_pages_show_a_reviewers_route_change(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    body = {
        "reviewer": "Sam",
        "action": "override",
        "final_route": "FRAUD_REVIEW",
        "note": "same photo as before",
    }
    client.post(f"/api/claims/{claim}/review", json=body)
    assert "changed by a reviewer to" in client.get(f"/claims/{claim}").text
    assert "FRAUD_REVIEW" in client.get("/claims").text
