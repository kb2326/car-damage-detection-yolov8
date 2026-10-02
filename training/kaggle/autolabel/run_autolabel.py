"""Kaggle job: run the fusion-eval-v1 auto-label job on a free Kaggle GPU.

Pushed with `kaggle kernels push -p training/kaggle/autolabel`. It reads the private bundle
dataset, clones the repository, installs the locked environment with uv, runs the job, and leaves
only the results in /kaggle/working so they can be downloaded with `kaggle kernels output`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = "https://github.com/kb2326/claimlens.git"
# The exact commit this job runs; update it (and re-push the kernel) to run newer code.
COMMIT = "0396bfd23f9c99a1f4b1748d8c9b1b7a1a176ffe"
WORK = Path("/tmp/claimlens")
OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RESULTS = (
    "data/interim/fusion-eval-v1/autolabels.jsonl",
    "reports/data/fusion-eval-v1-autolabel.json",
)


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def unpack_bundle() -> None:
    """Kaggle may keep the uploaded zip or extract it; handle both."""
    archives = sorted(INPUT.rglob("autolabel-bundle.zip"))
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(WORK)
        return
    records = sorted(INPUT.rglob("data/interim/damage-v1/records.jsonl"))
    if not records:
        raise SystemExit("bundle not found under /kaggle/input")
    shutil.copytree(records[0].parents[3] / "data", WORK / "data", dirs_exist_ok=True)


def main() -> None:
    run("nvidia-smi")
    run("git", "clone", "--filter=blob:none", REPO, str(WORK))
    run("git", "checkout", "--detach", COMMIT, cwd=WORK)
    unpack_bundle()
    run(sys.executable, "-m", "pip", "install", "-q", "uv")
    run("uv", "sync", "--locked", "--no-dev", "--group", "autolabel", cwd=WORK)
    run("uv", "run", "--no-sync", "claimlens", "data", "autolabel", "fusion-eval-v1", cwd=WORK)
    for relative in RESULTS:
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORK / relative, destination)
    (OUT / "commit.txt").write_text(f"{COMMIT}\n", encoding="utf-8")
    print(f"Results for commit {COMMIT} copied to /kaggle/working", flush=True)


if __name__ == "__main__":
    main()
