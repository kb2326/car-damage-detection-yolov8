"""Export the read-only showcase as plain static pages (M8b, ADR 0020).

The showcase never changes, so its pages can be rendered once by the real app and served by any
static host. Root-relative links become file names: `/` → `index.html`, `/claims` → `claims.html`,
`/claims/<id>` → `claim-<id>.html`, photos → `photos/<claim>-<photo>.<ext>`.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from claimlens.events.projection import fold
from claimlens.web.app import HERE, create_app
from claimlens.web.services import WebServices

REPO = "https://github.com/kb2326/claimlens"
PAGES = {
    "/": "index.html",
    "/claims": "claims.html",
    "/claims?filter=review": "claims-review.html",
    "/claims?filter=fraud": "claims-fraud.html",
}


def _rewrite(html: str, photos: dict[tuple[str, str], str]) -> str:
    html = re.sub(
        r"/api/claims/([0-9a-f-]{36})/photos/([A-Za-z0-9_-]+)",
        lambda m: f"photos/{photos[(m.group(1), m.group(2))]}",
        html,
    )
    html = html.replace('href="/static/', 'href="static/').replace('src="/static/', 'src="static/')
    html = re.sub(r'href="/claims/([0-9a-f-]{36})"', r'href="claim-\1.html"', html)
    for path, name in sorted(PAGES.items(), key=lambda item: -len(item[0])):
        html = html.replace(f'href="{path}"', f'href="{name}"')
    html = html.replace('<a href="/docs">API</a>', f'<a href="{REPO}">Code</a>')
    # The page scripts only follow a running claim or post forms; a static page needs neither.
    html = re.sub(r'<script[^>]*src="static/[^"]+"></script>', "", html)
    return html


def export_site(services: WebServices, out: Path) -> list[Path]:
    """Render every showcase page into `out`; returns the files written."""
    # The test client calls itself "testserver"; the showcase's host list is otherwise kept.
    hosts = (*services.settings.allowed_hosts, "testserver")
    shown = replace(
        services, settings=services.settings.model_copy(update={"allowed_hosts": hosts})
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "photos").mkdir(exist_ok=True)
    photos: dict[tuple[str, str], str] = {}
    claim_ids: list[str] = []
    with services.open_store() as store:
        for claim_id in store.claim_ids():
            claim = str(claim_id)
            claim_ids.append(claim)
            for photo in fold(store.load(claim_id)).photos.values():
                source = services.blobs.path(photo.blob_name)
                name = f"{claim}-{photo.photo_id}{source.suffix}"
                shutil.copy2(source, out / "photos" / name)
                photos[(claim, photo.photo_id)] = name
    written: list[Path] = []
    with TestClient(create_app(shown)) as client:
        pages = dict(PAGES)
        pages.update({f"/claims/{claim}": f"claim-{claim}.html" for claim in claim_ids})
        for path, name in pages.items():
            response = client.get(path)
            response.raise_for_status()
            target = out / name
            target.write_text(_rewrite(response.text, photos), encoding="utf-8")
            written.append(target)
    shutil.copytree(HERE / "static", out / "static", dirs_exist_ok=True)
    return written
