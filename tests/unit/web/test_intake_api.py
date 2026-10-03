import io
import sqlite3
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image

from claimlens.agent.chat_model import GatewayChatModel
from claimlens.events.store import SQLiteEventStore
from claimlens.intake_agent.graph import IntakeState, build_intake_graph
from claimlens.intake_agent.session import IntakeSessions, file_intake_claim
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.gateway import Gateway
from claimlens.llm.provider import FakeProvider, ProviderFatalError, ProviderReply
from claimlens.web.app import create_app
from claimlens.web.services import WebServices
from tests.unit.test_intake_graph import CONFIG, LLM, TODAY, lookup
from tests.unit.test_intake_session import SCRIPT


def _png(seed: int) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise((640, 480), 60 + seed).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def _chat(
    services: WebServices, tmp_path: Path, script: list[ProviderReply | Exception]
) -> TestClient:
    """Give the services an intake chat on a scripted fake LLM; return a client for the app."""

    def build() -> IntakeSessions:
        gateway = Gateway(
            LLM,
            FakeProvider(script),
            Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: TODAY),
            ResponseCache(tmp_path / "c.sqlite"),
            lambda c: None,
            sleep=lambda s: None,
        )
        model = GatewayChatModel(gateway=gateway, tier="fast", max_tokens=400)

        def submit(state: IntakeState) -> str:
            store = SQLiteEventStore(services.db_path)
            try:
                return str(file_intake_claim(store, services.blobs, state, CONFIG))
            finally:
                store.close()

        conn = sqlite3.connect(str(tmp_path / "intake.sqlite"), check_same_thread=False)
        graph = build_intake_graph(
            model, lookup, CONFIG, submit, lambda: TODAY, checkpointer=SqliteSaver(conn)
        )
        return IntakeSessions(graph, system="system prompt")

    services.intake = build
    return TestClient(create_app(services))


def _photo(client: TestClient, session: str, seed: int) -> dict[str, Any]:
    files = {"photo": (f"p{seed}.png", _png(seed), "image/png")}
    response = client.post(f"/api/intake/{session}/reply", data={"text": ""}, files=files)
    body: dict[str, Any] = response.json()
    return body


def test_a_chat_files_a_claim_and_starts_the_pipeline(
    services: WebServices, tmp_path: Path
) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    start = client.post("/api/intake")
    assert start.status_code == 201
    session = start.json()["session_id"]
    turn = client.post(
        f"/api/intake/{session}/reply", data={"text": "P-1001, bollard yesterday"}
    ).json()
    assert turn["photo_kind"] == "overview"
    for seed in (1, 2, 3):
        turn = _photo(client, session, seed)
    assert turn["claim_id"] is not None
    detail = client.get(f"/api/claims/{turn['claim_id']}").json()
    assert detail["status"] == "decided"  # the inline runner processed it
    assert detail["stages"][0]["state"] == "done"  # the intake chat
    assert len(detail["photos"]) == 3


def test_the_customer_never_sees_a_route_or_price(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    last = [_photo(client, session, seed) for seed in (1, 2, 3)][-1]
    assert set(last) == {"session_id", "message", "photo_kind", "claim_id"}
    for word in ("FAST_TRACK", "ADJUSTER_REVIEW", "FRAUD_REVIEW", "$"):
        assert word not in last["message"]


def test_a_reloaded_chat_continues(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    again = client.get(f"/api/intake/{session}").json()
    assert again["session_id"] == session
    assert again["photo_kind"] == "overview"


def test_an_unknown_session_is_404(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    assert client.get("/api/intake/nope").status_code == 404
    assert client.post("/api/intake/nope/reply", data={"text": "hi"}).status_code == 404


def test_a_bad_photo_is_422_and_the_chat_waits(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    files = {"photo": ("car.jpg", b"not an image", "image/jpeg")}
    response = client.post(f"/api/intake/{session}/reply", data={"text": ""}, files=files)
    assert response.status_code == 422
    assert "JPEG, PNG or WebP" in response.json()["detail"]
    assert client.get(f"/api/intake/{session}").json()["photo_kind"] == "overview"


def test_a_photo_over_the_limit_is_422(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    files = {"photo": ("big.png", b"\x89PNG" + b"0" * 1_000_001, "image/png")}
    response = client.post(f"/api/intake/{session}/reply", data={"text": ""}, files=files)
    assert response.status_code == 422
    assert "larger than 1 MB" in response.json()["detail"]


def test_an_llm_outage_is_503_and_the_answers_are_kept(
    services: WebServices, tmp_path: Path
) -> None:
    script: list[ProviderReply | Exception] = [SCRIPT[0], ProviderFatalError("overloaded")]
    client = _chat(services, tmp_path, script)
    session = client.post("/api/intake").json()["session_id"]
    response = client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    assert response.status_code == 503
    assert "answers are saved" in response.json()["detail"]


def test_no_intake_configured_is_503(client: TestClient) -> None:
    assert client.post("/api/intake").status_code == 503
    assert client.get("/api/intake/abc").status_code == 503
