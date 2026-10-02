"""`claimlens train …`: run a training job, import it, choose the champion, benchmark it."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from claimlens.domain import Frozen
from claimlens.training.config import dvc_out_md5, load_training_runs
from claimlens.training.manifest import read_json
from claimlens.training.run import Trainer, run_training
from claimlens.vision.instances import Segmenter

TrainerFactory = Callable[[], Trainer]
SegmenterFactory = Callable[[Path, str], Segmenter]
ExporterFactory = Callable[[Path], Path]


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
    select = train_sub.add_parser("select", help="choose the champion on validation mask mAP50")
    select.add_argument("--task", choices=["damage", "parts"], default="damage")
    report = train_sub.add_parser("report", help="write a model report")
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--task", choices=["damage", "parts"], default="damage")
    cal = train_sub.add_parser("calibrate", help="fit a temperature on validation predictions")
    cal.add_argument("run")
    cal.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all)")
    fusion = train_sub.add_parser("fusion-eval", help="part agreement on fusion-eval-v1")
    fusion.add_argument("run", help="a parts run")
    bench = train_sub.add_parser("benchmark", help="median CPU ms per image on test photos")
    bench.add_argument("run")
    bench.add_argument("--images", type=int, default=20)
    bench.add_argument("--format", choices=["pt", "onnx"], default="pt")
    export = train_sub.add_parser("export", help="export a run's weights to ONNX")
    export.add_argument("run")


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
    from claimlens.training.manifest import RunManifest
    from claimlens.training.tracking import default_tracking, import_run

    repo_root = Path.cwd()
    task = read_json(args.source / "manifest.json", RunManifest).config.task
    tracking_uri, artifact_root = default_tracking(repo_root)
    report = import_run(
        args.source,
        run_name=args.run,
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        models_dir=repo_root / "models" / task,
        reports_dir=repo_root / "reports" / "models",
        allow_incomplete=args.allow_incomplete,
    )
    print(
        f"Imported {report.run} as {report.model_version}: "
        f"val mask mAP50 {report.val.mask_map50:.3f}, test {report.test.mask_map50:.3f}"
    )
    return 0


def _select(args: argparse.Namespace) -> int:
    from claimlens.training.select import (
        champion_from_report,
        load_model_reports,
        select_champion,
        update_models_config,
    )
    from claimlens.training.tracking import default_tracking, set_champion_alias

    repo_root = Path.cwd()
    champion = select_champion(load_model_reports(repo_root / "reports" / "models", args.task))
    tracking_uri, _ = default_tracking(repo_root)
    set_champion_alias(tracking_uri, champion.model_version, args.task)
    update_models_config(args.config / "models.toml", args.task, champion_from_report(champion))
    print(f"Champion: {champion.run} (val mask mAP50 {champion.val.mask_map50:.3f})")
    return 0


def _report(args: argparse.Namespace) -> int:
    from datetime import date

    from claimlens.training.select import (
        load_model_reports,
        load_models_config,
        render_model_report,
    )

    reports = load_model_reports(Path.cwd() / "reports" / "models", args.task)
    models = load_models_config(args.config / "models.toml")
    chosen = getattr(models, args.task) if models is not None else None
    if chosen is None:
        raise ValueError(f"no champion yet: run `claimlens train select --task {args.task}` first")
    champion = next(r for r in reports if r.run == chosen.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_model_report(reports, champion, date.today()), encoding="utf-8")
    print(f"Report written to {args.out}")
    return 0


def _benchmark(args: argparse.Namespace, segmenter_factory: SegmenterFactory) -> int:
    from claimlens.training.benchmark import benchmark_callable
    from claimlens.training.manifest import ModelReport, write_json

    repo_root = Path.cwd()
    report_path = repo_root / "reports" / "models" / f"{args.run}.json"
    report = read_json(report_path, ModelReport)
    test_dir = repo_root / "data" / "processed" / report.dataset / "images" / "test"
    images = sorted(test_dir.glob("*.jpg"))[: args.images]
    weights = repo_root / "models" / report.task / args.run / f"best.{args.format}"
    segmenter = segmenter_factory(weights, args.run)
    ms = round(benchmark_callable(segmenter.segment, images), 1)
    field = "cpu_ms_per_image_onnx" if args.format == "onnx" else "cpu_ms_per_image"
    write_json(report_path, report.model_copy(update={field: ms}))
    print(f"{args.run} ({args.format}): median {ms:.0f} ms per image on CPU over {len(images)}")
    return 0


def _export(args: argparse.Namespace, exporter: ExporterFactory) -> int:
    from claimlens.training.manifest import ModelReport

    repo_root = Path.cwd()
    report = read_json(repo_root / "reports" / "models" / f"{args.run}.json", ModelReport)
    weights = repo_root / "models" / report.task / args.run / "best.pt"
    exported = exporter(weights)
    target = weights.with_suffix(".onnx")
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    print(f"Exported {args.run} to {target}")
    return 0


def _calibrate(args: argparse.Namespace, segmenter_factory: SegmenterFactory) -> int:
    import yaml

    from claimlens.fusion import calibrate_confidence
    from claimlens.training.calibration import (
        CalibrationResult,
        ece,
        fit_temperature,
        match_predictions,
        read_yolo_labels,
        recommend_threshold,
    )
    from claimlens.training.manifest import ModelReport, write_json
    from claimlens.training.select import load_models_config, update_models_config

    repo_root = Path.cwd()
    report = read_json(repo_root / "reports" / "models" / f"{args.run}.json", ModelReport)
    dataset = repo_root / "data" / "processed" / report.dataset
    data_yaml = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in data_yaml["names"].items()}
    images = sorted((dataset / "images" / "val").glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    weights = repo_root / "models" / report.task / args.run / "best.pt"
    segmenter = segmenter_factory(weights, args.run)
    pairs: list[tuple[float, bool]] = []
    for image in images:
        label_file = dataset / "labels" / "val" / f"{image.stem}.txt"
        truths = read_yolo_labels(label_file, names) if label_file.is_file() else []
        pairs += match_predictions(segmenter.segment(image).instances, truths)
    temperature = fit_temperature(pairs)
    threshold = recommend_threshold(pairs, temperature)
    kept = (
        []
        if threshold is None
        else [c for p, c in pairs if calibrate_confidence(p, temperature) >= threshold]
    )
    result = CalibrationResult(
        run=args.run,
        images=len(images),
        predictions=len(pairs),
        correct=sum(c for _, c in pairs),
        temperature=temperature,
        ece_before=round(ece(pairs), 4),
        ece_after=round(ece(pairs, temperature=temperature), 4),
        recommended_threshold=threshold,
        precision_at_threshold=round(sum(kept) / len(kept), 4) if kept else None,
        kept_at_threshold=len(kept),
    )
    write_json(repo_root / "reports" / "models" / "calibration" / f"{args.run}.json", result)
    models = load_models_config(args.config / "models.toml")
    if models is not None and models.damage is not None and models.damage.run == args.run:
        update_models_config(
            args.config / "models.toml",
            "damage",
            models.damage.model_copy(
                update={"temperature": temperature, "recommended_threshold": threshold}
            ),
        )
    print(
        f"{args.run}: T={temperature:.2f}, ECE {result.ece_before:.3f} -> "
        f"{result.ece_after:.3f}, recommended threshold {threshold}"
    )
    return 0


def _fusion_eval(args: argparse.Namespace, segmenter_factory: SegmenterFactory) -> int:
    import json

    from claimlens.data.records import read_records
    from claimlens.data.taxonomy import load_part_groups
    from claimlens.training.calibration import Truth, part_agreement
    from claimlens.vision.instances import SegInstance

    repo_root = Path.cwd()
    records = read_records(repo_root / "data" / "processed" / "fusion-eval-v1" / "parts.jsonl")
    segmenter = segmenter_factory(repo_root / "models" / "parts" / args.run / "best.pt", args.run)
    truths: dict[str, list[Truth]] = {
        r.image_id: [(a.label, tuple(a.polygon)) for a in r.annotations] for r in records
    }
    preds: dict[str, list[SegInstance]] = {
        r.image_id: list(segmenter.segment(repo_root / r.path).instances) for r in records
    }
    agreement = part_agreement(truths, preds, load_part_groups(args.config / "taxonomy.toml"))
    out = repo_root / "reports" / "models" / "fusion-eval" / f"{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {g: {"agreed": a, "total": t} for g, (a, t) in agreement.items()}
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    agreed = sum(a for a, _ in agreement.values())
    total = sum(t for _, t in agreement.values())
    print(f"{args.run}: {agreed} of {total} reviewed parts found ({agreed / max(total, 1):.0%})")
    return 0


def run_train_command(
    args: argparse.Namespace,
    *,
    trainer_factory: TrainerFactory,
    segmenter_factory: SegmenterFactory,
    exporter: ExporterFactory,
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
        if args.train_command == "fusion-eval":
            return _fusion_eval(args, segmenter_factory)
        if args.train_command == "calibrate":
            return _calibrate(args, segmenter_factory)
        if args.train_command == "benchmark":
            return _benchmark(args, segmenter_factory)
        if args.train_command == "export":
            return _export(args, exporter)
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
