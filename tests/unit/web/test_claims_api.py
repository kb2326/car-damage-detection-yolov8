import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from claimlens.intake import submit_claim
from claimlens.web.services import WebServices
from tests.unit.web.helpers import decided_claim, image

TAMPER = "UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1"


def _decided(
    services: WebServices, tmp_path: Path, color: tuple[int, int, int] = (200, 30, 30)
) -> str:
    with services.open_store() as store:
        return str(decided_claim(tmp_path, store, color))


def _filed(services: WebServices, tmp_path: Path, name: str) -> UUID:
    with services.open_store() as store:
        return submit_claim(
            store,
            services.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[image(tmp_path / "in" / name)],
        )


def test_the_list_and_the_detail(client: TestClient, services: WebServices, tmp_path: Path) -> None:
    claim = _decided(services, tmp_path)
    listed = client.get("/api/claims").json()
    assert [c["claim_id"] for c in listed] == [claim]
    assert listed[0]["status"] == "decided"
    detail = client.get(f"/api/claims/{claim}").json()
    assert detail["chain_ok"] is True
    assert detail["photos"][0]["boxes"][0]["label"].startswith("dent")


def test_an_unknown_claim_is_404(client: TestClient) -> None:
    response = client.get(f"/api/claims/{uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"detail": "No such claim."}


def test_a_malformed_claim_id_is_404(client: TestClient) -> None:
    assert client.get("/api/claims/not-a-uuid").status_code == 404


def test_a_tampered_log_is_409(client: TestClient, services: WebServices, tmp_path: Path) -> None:
    claim = _decided(services, tmp_path)
    with sqlite3.connect(services.db_path) as conn:
        conn.execute(TAMPER)
    response = client.get(f"/api/claims/{claim}")
    assert response.status_code == 409
    assert "failed verification" in response.json()["detail"]
    assert client.get("/api/claims").json()[0]["status"] == "log_failed"


def test_a_photo_is_served_for_its_own_claim(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    response = client.get(f"/api/claims/{claim}/photos/p1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_a_photo_of_another_claim_is_404(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    first = _decided(services, tmp_path, (200, 30, 30))
    _decided(services, tmp_path, (30, 200, 30))
    assert client.get(f"/api/claims/{first}/photos/p9").status_code == 404
    assert client.get(f"/api/claims/{first}/photos/..%2F..%2Fevents.db").status_code == 404


def test_resume_finishes_a_stopped_claim(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _filed(services, tmp_path, "a.png")
    assert client.get(f"/api/claims/{claim}").json()["status"] == "stopped"
    assert client.post(f"/api/claims/{claim}/resume").status_code == 202
    assert client.get(f"/api/claims/{claim}").json()["status"] == "decided"


def test_resume_of_a_decided_claim_is_409(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    assert client.post(f"/api/claims/{claim}/resume").status_code == 409


def test_resume_while_running_is_409(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _filed(services, tmp_path, "b.png")
    services.runner._running.add(claim)  # as if a worker were busy with it
    assert client.post(f"/api/claims/{claim}/resume").status_code == 409


def test_a_failed_run_shows_its_error(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _filed(services, tmp_path, "c.png")
    services.runner._errors[claim] = "Processing stopped unexpectedly (RuntimeError)."
    detail = client.get(f"/api/claims/{claim}").json()
    assert detail["status"] == "stopped"
    assert detail["error"] == "Processing stopped unexpectedly (RuntimeError)."


def test_the_api_is_documented(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/claims/{claim_id}" in paths
