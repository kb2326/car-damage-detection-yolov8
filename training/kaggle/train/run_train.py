"""Kaggle job: train the M3a damage models on a free Kaggle GPU.

Pushed with `kaggle kernels push -p .` from this folder. It reads the private `claimlens-damage-v1`
dataset, checks out the pinned commit, installs the locked `vision` group with uv, and runs
`claimlens train run` for each run. Results go straight to /kaggle/working/<run>/ so they survive
a time-out, and are downloaded with `kaggle kernels output`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = "https://github.com/kb2326/claimlens.git"
# The exact commit this job runs: the pushed commit that contains the training code.
COMMIT = "SET-IN-TASK-9"
RUNS = ("damage-yolo11n-v1", "damage-yolo11s-v1")
WORK = Path("/tmp/claimlens")
OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def bundle_root() -> Path:
    """Kaggle may keep the uploaded zip or extract it; return the folder holding dataset.json."""
    archives = sorted(INPUT.rglob("claimlens-damage-v1.zip"))
    if archives:
        target = Path("/tmp/bundle")
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(target)
        return target
    manifests = sorted(INPUT.rglob("dataset.json"))
    if not manifests:
        raise SystemExit("dataset.json not found under /kaggle/input")
    return manifests[0].parent


def main() -> None:
    if COMMIT == "SET-IN-TASK-9":
        raise SystemExit("set COMMIT to the pushed commit before pushing this kernel")
    run("nvidia-smi")
    run("git", "clone", "--filter=blob:none", REPO, str(WORK))
    run("git", "checkout", "--detach", COMMIT, cwd=WORK)
    root = bundle_root()
    shutil.copytree(root / "data", WORK / "data", dirs_exist_ok=True)
    run(sys.executable, "-m", "pip", "install", "-q", "uv")
    run("uv", "sync", "--locked", "--no-dev", "--group", "vision", cwd=WORK)
    (OUT / "commit.txt").write_text(f"{COMMIT}\n", encoding="utf-8")
    failed = []
    for name in RUNS:
        try:
            run(
                "uv",
                "run",
                "--no-sync",
                "claimlens",
                "train",
                "run",
                name,
                "--dataset-dir",
                "data/processed/damage-v1",
                "--bundle",
                str(root / "dataset.json"),
                "--out",
                str(OUT),
                "--commit",
                COMMIT,
                cwd=WORK,
            )
        except subprocess.CalledProcessError as exc:
            print(f"run {name} failed: {exc}", flush=True)
            failed.append(name)
    print(f"Done. Failed runs: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
