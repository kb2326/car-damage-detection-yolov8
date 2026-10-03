"""Deploy the read-only showcase to a free static Hugging Face Space (M8b, ADR 0020).

Hugging Face charges for Docker Spaces, so the showcase is exported as plain static pages: the
real app renders every page once (claimlens.web.static_export). Only those pages, the stylesheet,
the credited photos and the Space README are uploaded. Used by .github/workflows/deploy-space.yml;
also runnable locally:

    HF_TOKEN=... uv run --with huggingface_hub==0.35.3 \\
        python scripts/deploy_space.py <owner>/claimlens
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stage(root: Path, out: Path) -> None:
    """Export the static showcase from root/showcase into `out`, with the Space README."""
    from claimlens.web.serve import showcase_services
    from claimlens.web.settings import load_web_settings
    from claimlens.web.static_export import export_site

    settings = load_web_settings(root / "config" / "web.toml")
    export_site(showcase_services(root / "showcase", settings), out)
    shutil.copy2(root / "hf-space" / "README.md", out / "README.md")


def deploy(space: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        stage(ROOT, Path(folder))
        api.upload_folder(
            folder_path=folder,
            repo_id=space,
            repo_type="space",
            commit_message="Deploy the ClaimLens showcase",
            delete_patterns=["*"],  # the Space mirrors the staged folder exactly
        )
    print(f"Deployed to https://huggingface.co/spaces/{space}")


def main() -> int:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        print("usage: deploy_space.py <owner>/<space>", file=sys.stderr)
        return 2
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("error: set HF_TOKEN (a Hugging Face write token)", file=sys.stderr)
        return 2
    deploy(sys.argv[1], token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
