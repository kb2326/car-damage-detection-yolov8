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


def test_the_chat_page_renders(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "File a claim" in page.text
    assert "/static/chat.js" in page.text
    assert client.get("/static/chat.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_the_claims_page_lists_and_filters(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    assert claim in client.get("/claims").text
    assert claim not in client.get("/claims?filter=fraud").text
    assert client.get("/claims?filter=nonsense").status_code == 422


def test_the_claim_page_shows_the_evidence_and_escapes_customer_text(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    page = client.get(f"/claims/{claim}").text
    assert "Log verified" in page
    assert "dent · back_bumper · 0.90" in page
    assert "&lt;b&gt;slowly&lt;/b&gt;" in page
    assert "<b>slowly</b>" not in page
    assert 'id="review-form"' in page
    assert "/static/claim.js" in page


def test_a_stopped_claim_offers_resume_and_no_review(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    with services.open_store() as store:
        claim = submit_claim(
            store,
            services.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[image(tmp_path / "in" / "s.png")],
        )
    page = client.get(f"/claims/{claim}").text
    assert 'id="resume"' in page
    assert 'id="review-form"' not in page


def test_a_tampered_claim_page_hides_the_review_form(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    with sqlite3.connect(services.db_path) as conn:
        conn.execute(TAMPER)
    page = client.get(f"/claims/{claim}")
    assert page.status_code == 409
    assert "Log failed verification" in page.text
    assert 'id="review-form"' not in page.text


def test_an_unknown_claim_page_is_a_404_page(client: TestClient) -> None:
    page = client.get("/claims/not-a-claim")
    assert page.status_code == 404
    assert "not found" in page.text.lower()
    assert "not-a-claim" in page.text


def test_the_chat_page_never_mentions_routes_or_cover(client: TestClient) -> None:
    page = client.get("/").text
    for word in ("FAST_TRACK", "ADJUSTER_REVIEW", "FRAUD_REVIEW", "deductible"):
        assert word not in page
