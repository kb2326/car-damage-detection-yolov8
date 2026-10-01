"""Download raw data sources into data/raw/<id>/ and record a checksum manifest."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from claimlens.data.config import SourceConfig

HttpGet = Callable[[str], bytes]


def urllib_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as response:
        data: bytes = response.read()
    return data


def roboflow_download_url(source: SourceConfig, api_key: str, http_get: HttpGet) -> str:
    endpoint = (
        f"https://api.roboflow.com/{source.workspace}/{source.project}/"
        f"{source.version}/{source.export_format}?api_key={api_key}"
    )
    try:
        payload = json.loads(http_get(endpoint))
    except Exception as exc:
        # Never echo the endpoint or the original message: both can contain the API key.
        raise RuntimeError(f"Roboflow export request failed: {type(exc).__name__}") from None
    return str(payload["export"]["link"])


def extract_zip(data: bytes, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"zip entry {member.filename!r} would extract outside {dest}")
        archive.extractall(root)


def write_manifest(root: Path) -> Path:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    ]
    manifest = root / "MANIFEST.json"
    manifest.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def fetch_source(
    source: SourceConfig,
    raw_root: Path,
    *,
    api_key: str | None = None,
    http_get: HttpGet = urllib_get,
) -> Path:
    dest = raw_root / source.id
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} already exists; raw data is immutable, delete it to refetch")
    if source.kind == "local":
        raise ValueError(f"{source.id} is a local source and cannot be downloaded")
    if source.kind == "roboflow":
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set (environment or .env)")
        url = roboflow_download_url(source, api_key, http_get)
    else:
        url = str(source.url)
    extract_zip(http_get(url), dest)
    write_manifest(dest)
    return dest
