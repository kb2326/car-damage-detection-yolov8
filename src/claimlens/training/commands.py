"""`claimlens train …`: run a training job, import it, choose the champion, benchmark it."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from claimlens.domain import Frozen
from claimlens.training.config import dvc_out_md5, load_training_runs
from claimlens.training.manifest import read_json
from claimlens.training.run import Trainer, run_training
from claimlens.vision.base import Detector

TrainerFactory = Callable[[], Trainer]
DetectorFactory = Callable[[str, Path], Detector]


class BundleInfo(Frozen):
    dataset: str
    md5: str


def add_train_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train = sub.add_parser("train", help="train models and manage the model registry")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    run = train_sub.add_parser("run", help="train one run (used inside the Kaggle job)")
    run.add_argument("run")
    run.add_argument("--dataset-dir", type=Path, required=True)
    run.add_argument("--bundle", type=Path, required=True, help="dataset.json from the bundle")
    run.add_argument("--out", type=Path, default=Path("var/runs"))
    run.add_argument("--commit", default=None, help="defaults to `git rev-parse HEAD`")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _run(args: argparse.Namespace, trainer_factory: TrainerFactory) -> int:
    runs = load_training_runs(args.config / "training.toml")
    if args.run not in runs:
        raise ValueError(f"unknown training run {args.run!r}; known: {sorted(runs)}")
    run = runs[args.run]
    bundle = read_json(args.bundle, BundleInfo)
    if bundle.dataset != run.dataset:
        raise ValueError(
            f"bundle holds {bundle.dataset!r} but run {run.name} needs {run.dataset!r}"
        )
    expected = dvc_out_md5(Path.cwd() / "dvc.lock", f"data/processed/{run.dataset}")
    manifest = run_training(
        run,
        dataset_dir=args.dataset_dir,
        bundle_md5=bundle.md5,
        expected_md5=expected,
        commit=args.commit or _git_commit(),
        out_dir=args.out,
        trainer=trainer_factory(),
    )
    print(
        f"Trained {manifest.run}: {manifest.status}, {manifest.epochs_run} epochs, "
        f"best epoch {manifest.best_epoch}"
    )
    return 0


def run_train_command(
    args: argparse.Namespace,
    *,
    trainer_factory: TrainerFactory,
    detector_factory: DetectorFactory,
) -> int:
    try:
        if args.train_command == "run":
            return _run(args, trainer_factory)
        raise ValueError(f"unknown train command {args.train_command!r}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
