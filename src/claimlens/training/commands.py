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
    imp = train_sub.add_parser("import", help="log a finished run into the local MLflow logbook")
    imp.add_argument("run")
    imp.add_argument(
        "--from", dest="source", type=Path, required=True, help="downloaded run folder"
    )
    imp.add_argument("--allow-incomplete", action="store_true")
    train_sub.add_parser("select", help="choose the champion on validation mask mAP50")
    report = train_sub.add_parser("report", help="write the damage model report")
    report.add_argument("--out", type=Path, required=True)
    bench = train_sub.add_parser("benchmark", help="median CPU ms per image on test photos")
    bench.add_argument("run")
    bench.add_argument("--images", type=int, default=20)


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


def _import(args: argparse.Namespace) -> int:
    from claimlens.training.tracking import default_tracking, import_run

    repo_root = Path.cwd()
    tracking_uri, artifact_root = default_tracking(repo_root)
    report = import_run(
        args.source,
        run_name=args.run,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        models_dir=repo_root / "models" / "damage",
        reports_dir=repo_root / "reports" / "models",
        allow_incomplete=args.allow_incomplete,
    )
    print(
        f"Imported {report.run} as {report.model_version}: "
        f"val mask mAP50 {report.val.mask_map50:.3f}, test {report.test.mask_map50:.3f}"
    )
    return 0


def _select(args: argparse.Namespace) -> int:
    from claimlens.training.select import load_model_reports, select_champion, write_models_config
    from claimlens.training.tracking import default_tracking, set_champion_alias

    repo_root = Path.cwd()
    champion = select_champion(load_model_reports(repo_root / "reports" / "models"))
    tracking_uri, _ = default_tracking(repo_root)
    set_champion_alias(tracking_uri, champion.model_version)
    write_models_config(args.config / "models.toml", champion)
    print(f"Champion: {champion.run} (val mask mAP50 {champion.val.mask_map50:.3f})")
    return 0


def _report(args: argparse.Namespace) -> int:
    from datetime import date

    from claimlens.training.select import (
        load_model_reports,
        load_models_config,
        render_model_report,
    )

    reports = load_model_reports(Path.cwd() / "reports" / "models")
    models = load_models_config(args.config / "models.toml")
    if models is None:
        raise ValueError("no champion yet: run `claimlens train select` first")
    champion = next(r for r in reports if r.run == models.damage.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_model_report(reports, champion, date.today()), encoding="utf-8")
    print(f"Report written to {args.out}")
    return 0


def _benchmark(args: argparse.Namespace, detector_factory: DetectorFactory) -> int:
    from claimlens.training.benchmark import benchmark_detector
    from claimlens.training.manifest import ModelReport, read_json, write_json

    repo_root = Path.cwd()
    report_path = repo_root / "reports" / "models" / f"{args.run}.json"
    report = read_json(report_path, ModelReport)
    test_dir = repo_root / "data" / "processed" / report.dataset / "images" / "test"
    images = sorted(test_dir.glob("*.jpg"))[: args.images]
    detector = detector_factory("yolo-seg", repo_root / "models" / "damage" / args.run / "best.pt")
    ms = benchmark_detector(detector, images)
    write_json(report_path, report.model_copy(update={"cpu_ms_per_image": round(ms, 1)}))
    print(f"{args.run}: median {ms:.0f} ms per image on CPU over {len(images)} images")
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
        if args.train_command == "import":
            return _import(args)
        if args.train_command == "select":
            return _select(args)
        if args.train_command == "report":
            return _report(args)
        if args.train_command == "benchmark":
            return _benchmark(args, detector_factory)
        raise ValueError(f"unknown train command {args.train_command!r}")
    except Exception as exc:
        from claimlens.training.tracking import TrainingUnavailableError

        if isinstance(exc, TrainingUnavailableError):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if isinstance(exc, OSError | ValueError | subprocess.CalledProcessError):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise
