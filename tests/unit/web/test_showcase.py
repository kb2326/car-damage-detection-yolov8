"""Showcase mode: the public, read-only demo (M8b)."""

import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from claimlens.web.app import create_app
from claimlens.web.services import WebServices
from tests.unit.web.helpers import decided_claim

TRANSCRIPT = [
    {"role": "agent", "text": "Hello! What is your policy number?"},
    {"role": "customer", "text": "P-1001, a van clipped my mirror <b>yesterday</b>"},
]


@pytest.fixture
def showcase(services: WebServices, tmp_path: Path) -> Iterator[tuple[TestClient, str]]:
    with services.open_store() as store:
        claim = str(decided_claim(tmp_path, store))
    shown = replace(
        services,
        settings=services.settings.model_copy(update={"showcase": True}),
        read_only=True,
        transcript=TRANSCRIPT,
    )
    with TestClient(create_app(shown)) as client:
        yield client, claim


def test_every_write_route_is_refused(showcase: tuple[TestClient, str]) -> None:
    client, claim = showcase
    paths = client.get("/openapi.json").json()["paths"]
    writes = [
        (method, path)
        for path, item in paths.items()
        for method in item
        if method in ("post", "put", "patch", "delete")
    ]
    assert len(writes) >= 4  # start chat, reply, review, resume
    for method, path in writes:
        url = path.replace("{claim_id}", claim).replace("{session_id}", "s1")
        response = client.request(method.upper(), url, json={})
        assert response.status_code == 403, url
        assert "read-only showcase" in response.json()["detail"]


def test_the_pages_show_the_samples_and_the_banner(showcase: tuple[TestClient, str]) -> None:
    client, claim = showcase
    for path in ("/", "/claims", f"/claims/{claim}"):
        page = client.get(path)
        assert page.status_code == 200, path
        assert "Showcase" in page.text
        assert "github.com/kb2326/claimlens" in page.text
    assert claim[:8] in client.get("/claims").text


def test_the_chat_page_shows_the_recorded_chat_and_takes_no_input(
    showcase: tuple[TestClient, str],
) -> None:
    client, _ = showcase
    page = client.get("/").text
    assert "Hello! What is your policy number?" in page
    assert "&lt;b&gt;yesterday&lt;/b&gt;" in page
    assert 'id="chat-form"' not in page
    assert "/static/chat.js" not in page


def test_the_claim_page_has_no_review_or_resume(showcase: tuple[TestClient, str]) -> None:
    client, claim = showcase
    page = client.get(f"/claims/{claim}").text
    assert 'id="review-form"' not in page
    assert 'id="resume"' not in page
    assert "Log verified" in page


def test_the_store_is_opened_read_only(showcase: tuple[TestClient, str]) -> None:
    import sqlite3

    client, _ = showcase
    services: WebServices = client.app.state.services  # type: ignore[attr-defined]
    with services.open_store() as store, pytest.raises(sqlite3.OperationalError):
        store._conn.execute("DELETE FROM events")


def test_security_headers_on_every_page(client: TestClient) -> None:
    page = client.get("/claims")
    assert page.headers["x-content-type-options"] == "nosniff"
    assert page.headers["referrer-policy"] == "same-origin"
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert "content-security-policy" not in client.get("/docs").headers  # Swagger UI loads a CDN


def test_allowed_hosts_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from claimlens.web.settings import load_web_settings

    path = tmp_path / "web.toml"
    path.write_text("port = 8000\n", encoding="utf-8")
    monkeypatch.setenv("CLAIMLENS_ALLOWED_HOSTS", "kb2326-claimlens.hf.space, localhost")
    assert load_web_settings(path).allowed_hosts == ("kb2326-claimlens.hf.space", "localhost")


def test_serve_allows_a_public_host_only_in_showcase_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.cli import main
    from tests.fakes import CONFIG_DIR, FakeDetector

    data = tmp_path / "showcase"
    data.mkdir()
    db = tmp_path / "src.db"
    from claimlens.events.store import SQLiteEventStore

    store = SQLiteEventStore(db)
    decided_claim(tmp_path, store)
    store.close()
    shutil.copy(db, data / "claims.db")
    shutil.copytree(tmp_path / "blobs", data / "blobs")
    (data / "transcript.json").write_text("[]", encoding="utf-8")
    started: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.update(kw, app=app))
    base = ["--config", str(CONFIG_DIR)]
    refused = main(
        [*base, "serve", "--host", "0.0.0.0"], detector_factory=lambda *_: FakeDetector()
    )
    assert refused == 2
    assert "local only" in capsys.readouterr().err
    code = main(
        [*base, "serve", "--showcase", "--data", str(data), "--host", "0.0.0.0", "--port", "7860"],
        detector_factory=lambda *_: FakeDetector(),
    )
    assert code == 0
    assert started["host"] == "0.0.0.0"
    assert started["port"] == 7860
    services = started["app"].state.services
    assert services.settings.showcase
    assert services.read_only
