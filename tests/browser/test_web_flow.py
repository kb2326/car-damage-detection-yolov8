"""File a claim through the chat with photos, see it decided, approve it. Local only: needs
`uv sync --group browser` and `uv run playwright install chromium`."""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, expect, sync_playwright

from claimlens.web.app import create_app
from claimlens.web.services import WebServices
from tests.unit.test_intake_session import SCRIPT
from tests.unit.web.conftest import services  # noqa: F401  (the fixture)
from tests.unit.web.test_intake_api import _chat, _png

pytestmark = pytest.mark.browser
PORT = 8765


@pytest.fixture
def server(services: WebServices, tmp_path: Path) -> Iterator[str]:  # noqa: F811
    _chat(services, tmp_path, list(SCRIPT))  # an intake chat on the scripted fake LLM
    config = uvicorn.Config(create_app(services), host="127.0.0.1", port=PORT, log_level="warning")
    app_server = uvicorn.Server(config)
    thread = threading.Thread(target=app_server.run, daemon=True)
    thread.start()
    while not app_server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{PORT}"
    app_server.should_exit = True
    thread.join(timeout=5)


def _agent_says(page: Page, count: int) -> None:
    expect(page.locator(".messages li.agent")).to_have_count(count)


def test_file_follow_and_review_a_claim(server: str, tmp_path: Path) -> None:
    photos = []
    for seed in (1, 2, 3):
        path = tmp_path / f"photo{seed}.png"
        path.write_bytes(_png(seed))
        photos.append(path)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except PlaywrightError:
            pytest.skip("no browser installed: uv run playwright install chromium")
        page = browser.new_page()
        page.goto(server + "/")
        page.click("#start")
        _agent_says(page, 1)
        page.fill("#text", "P-1001, I reversed into a bollard yesterday")
        page.click("#send")
        _agent_says(page, 2)
        for n, path in enumerate(photos, start=3):
            page.set_input_files("#photo", str(path))
            page.click("#send")
            _agent_says(page, n)
        expect(page.locator("#done")).to_be_visible()
        assert page.url.startswith(server + "/?session=")
        page.click("#done a")
        expect(page.locator(".verified")).to_contain_text("Log verified")
        expect(page.locator(".stage.st-done").first).to_be_visible()
        page.fill("#reviewer", "Sam")
        page.select_option("#action", "approve")
        page.click("#review-form button[type=submit]")
        expect(page.locator(".chip.s-reviewed")).to_be_visible()
        page.screenshot(path=str(tmp_path / "claim-page.png"), full_page=True)
        browser.close()
