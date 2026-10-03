"""Deploy the read-only showcase to a Hugging Face Space (M8b, ADR 0020).

Stages only what the showcase image needs (no data, weights, tests or secrets), then uploads it
with the Hugging Face Hub API. Used by .github/workflows/deploy-space.yml; also runnable locally:

    HF_TOKEN=... uv run --no-project --with huggingface_hub \
        python scripts/deploy_space.py <owner>/claimlens
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "LICENSE")
FOLDERS = ("src", "config", "showcase")
# Never staged, even if present in the working tree: secrets and SQLite side files.
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".env*", "*-journal", "*.db-wal", "*.db-shm"
)


def stage(root: Path, out: Path) -> None:
    """Copy the showcase app into `out`, with the Space's README (front matter) as README.md."""
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(root / name, out / name)
    for folder in FOLDERS:
        shutil.copytree(root / folder, out / folder, ignore=IGNORE, dirs_exist_ok=True)
    shutil.copy2(root / "hf-space" / "README.md", out / "README.md")


def deploy(space: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
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
