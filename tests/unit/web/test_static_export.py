"""The showcase exported as plain static pages, for a free static host (M8b)."""

import re
from dataclasses import replace
from pathlib import Path

from claimlens.web.services import WebServices
from claimlens.web.static_export import export_site
from tests.unit.web.helpers import decided_claim

TRANSCRIPT = [{"role": "agent", "text": "Hello!"}, {"role": "customer", "text": "P-1001"}]


def _showcase(services: WebServices, tmp_path: Path) -> tuple[WebServices, str]:
    with services.open_store() as store:
        claim = str(decided_claim(tmp_path, store))
    shown = replace(
        services,
        settings=services.settings.model_copy(update={"showcase": True}),
        read_only=True,
        transcript=TRANSCRIPT,
    )
    return shown, claim


def test_every_page_and_photo_is_written(services: WebServices, tmp_path: Path) -> None:
    shown, claim = _showcase(services, tmp_path)
    out = tmp_path / "site"
    export_site(shown, out)
    for page in ("index.html", "claims.html", "claims-review.html", "claims-fraud.html"):
        assert (out / page).is_file(), page
    detail = (out / f"claim-{claim}.html").read_text(encoding="utf-8")
    assert "Log verified" in detail
    assert (out / "static" / "app.css").is_file()
    (photo,) = list((out / "photos").iterdir())
    assert photo.name.startswith(claim)
    assert f'src="photos/{photo.name}"' in detail


def test_no_link_points_at_the_server(services: WebServices, tmp_path: Path) -> None:
    shown, claim = _showcase(services, tmp_path)
    out = tmp_path / "site"
    export_site(shown, out)
    for page in out.glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert not re.search(r'(href|src)="/(?!/)', text), page.name  # no root-relative links
        assert "/api/" not in text, page.name
        assert 'id="review-form"' not in text
    claims = (out / "claims.html").read_text(encoding="utf-8")
    assert f'href="claim-{claim}.html"' in claims
    assert 'href="claims-review.html"' in claims
    assert "github.com/kb2326/claimlens" in claims


def test_the_recorded_chat_is_the_home_page(services: WebServices, tmp_path: Path) -> None:
    shown, _ = _showcase(services, tmp_path)
    out = tmp_path / "site"
    export_site(shown, out)
    home = (out / "index.html").read_text(encoding="utf-8")
    assert "A recorded claim chat" in home
    assert "Hello!" in home


def test_external_links_open_in_a_new_tab(services: WebServices, tmp_path: Path) -> None:
    """Hugging Face shows the Space in a frame, and GitHub refuses to load inside one."""
    shown, _ = _showcase(services, tmp_path)
    out = tmp_path / "site"
    export_site(shown, out)
    for page in out.glob("*.html"):
        for tag in re.findall(
            r'<a [^>]*href="https?://[^"]+"[^>]*>', page.read_text(encoding="utf-8")
        ):
            assert 'target="_blank"' in tag, (page.name, tag)
            assert 'rel="noopener noreferrer"' in tag, (page.name, tag)
