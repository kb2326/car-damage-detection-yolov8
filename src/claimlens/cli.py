"""Command-line interface for claims, datasets, evaluation and review (see `claimlens --help`)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Literal
from uuid import UUID

from claimlens.agent.stub import StubTriageAgent
from claimlens.autolabel.job import AutolabelJob, load_autolabel_jobs, run_autolabel
from claimlens.autolabel.labeller import PartLabeller
from claimlens.blobs import BlobStore
from claimlens.data.config import load_data_config, read_secret
from claimlens.data.fetch import fetch_source
from claimlens.data.pipeline import DataContractError, build_dataset
from claimlens.data.records import read_records, write_records
from claimlens.data.taxonomy import load_part_groups
from claimlens.decision import load_decision_config
from claimlens.domain import Frozen
from claimlens.evals.golden import load_golden, write_golden
from claimlens.evals.metrics import compute_triage_metrics
from claimlens.evals.triage import ReportMeta, render_report, run_triage_eval, what_if_thresholds
from claimlens.events.envelope import ChainIntegrityError, ClaimEvent
from claimlens.events.projection import ClaimState, fold
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card
from claimlens.review.decisions import (
    GoldenReview,
    PartReview,
    apply_golden_review,
    apply_part_review,
    check_part_review,
    file_sha256,
    merge_part_reviews,
    read_review,
    write_review,
)
from claimlens.training.commands import (
    ExporterFactory,
    SegmenterFactory,
    TrainerFactory,
    add_train_parser,
    run_train_command,
)
from claimlens.training.run import Trainer
from claimlens.training.select import load_models_config
from claimlens.vision.base import Detector
from claimlens.vision.instances import Segmenter
from claimlens.workflow import PipelineDeps, process_claim

DEFAULT_WEIGHTS = Path("models/legacy/yolov8n-cardamage-v6.pt")
DetectorFactory = Callable[["DetectorSpec"], Detector]
LabellerFactory = Callable[[AutolabelJob], PartLabeller]


class DetectorSpec(Frozen):
    """Which detector to build: legacy, our damage model, or damage + parts fused."""

    kind: Literal["legacy", "yolo-seg", "fused"]
    weights: Path
    parts_weights: Path | None = None
    temperature: float | None = None
    taxonomy: Path | None = None


def _default_detector(spec: DetectorSpec) -> Detector:
    if spec.kind == "fused":
        from claimlens.fusion import FusedDetector
        from claimlens.vision.ultralytics_segmenter import UltralyticsSegmenter

        if spec.parts_weights is None or spec.taxonomy is None:
            raise ValueError("a fused detector needs parts weights and a taxonomy")
        return FusedDetector(
            UltralyticsSegmenter(spec.weights, name=spec.weights.parent.name),
            UltralyticsSegmenter(spec.parts_weights, name=spec.parts_weights.parent.name),
            load_part_groups(spec.taxonomy),
            temperature=spec.temperature,
        )
    if spec.kind == "yolo-seg":
        from claimlens.vision.yolo_seg import YoloSegDetector

        return YoloSegDetector(spec.weights, run=spec.weights.parent.name)
    from claimlens.vision.legacy_yolo import LegacyYoloDetector

    return LegacyYoloDetector(spec.weights)


def resolve_detector(detector: str | None, weights: Path | None, config_dir: Path) -> DetectorSpec:
    """Explicit flags win; otherwise fused (both champions), the damage champion, or legacy."""
    models = load_models_config(config_dir / "models.toml")
    if detector is None and weights is not None:
        return DetectorSpec(kind="legacy", weights=weights)  # the pre-M3 meaning of --weights
    damage = models.damage if models is not None else None
    parts = models.parts if models is not None else None
    default = "fused" if damage and parts else "yolo-seg" if damage else "legacy"
    kind = detector or default
    if kind == "legacy":
        return DetectorSpec(kind="legacy", weights=weights or DEFAULT_WEIGHTS)
    if weights is not None:
        damage_weights = weights
    elif damage is not None:
        damage_weights = Path(damage.weights)
    else:
        raise ValueError("no champion model: run `claimlens train select` or pass --weights")
    if kind == "yolo-seg":
        return DetectorSpec(kind="yolo-seg", weights=damage_weights)
    if parts is None:
        raise ValueError("fused needs a parts champion: run `claimlens train select --task parts`")
    return DetectorSpec(
        kind="fused",
        weights=damage_weights,
        parts_weights=Path(parts.weights),
        temperature=damage.temperature if damage is not None else None,
        taxonomy=config_dir / "taxonomy.toml",
    )


def _thresholds(text: str) -> list[float]:
    try:
        values = [float(t) for t in text.split(",") if t.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a list of numbers: {text!r}") from None
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise argparse.ArgumentTypeError("thresholds must be between 0 and 1")
    return values


def _make_detector(args: argparse.Namespace, factory: DetectorFactory) -> Detector:
    try:
        return factory(resolve_detector(args.detector, args.weights, args.config))
    except (ValueError, FileNotFoundError) as exc:
        raise DetectorUnavailableError(str(exc)) from exc


def _ultralytics_segmenter(weights: Path, name: str) -> Segmenter:
    from claimlens.vision.ultralytics_segmenter import UltralyticsSegmenter

    return UltralyticsSegmenter(weights, name=name)


def _export_onnx(weights: Path) -> Path:
    from claimlens.vision.ultralytics_segmenter import export_onnx

    return export_onnx(weights)


def _ultralytics_trainer() -> Trainer:
    from claimlens.training.ultralytics_trainer import UltralyticsTrainer

    return UltralyticsTrainer()


def _grounded_sam(job: AutolabelJob) -> PartLabeller:
    from claimlens.autolabel.grounded_sam import GroundedSamLabeller

    return GroundedSamLabeller(
        job.prompts,
        box_threshold=job.box_threshold,
        text_threshold=job.text_threshold,
        min_score=job.min_score,
        max_per_group=job.max_per_group,
        detector=job.detector,
        segmenter=job.segmenter,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claimlens", description="ClaimLens claims triage")
    parser.add_argument("--db", type=Path, default=Path("var/claimlens.db"))
    parser.add_argument("--blobs", type=Path, default=Path("var/blobs"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--detector", choices=["legacy", "yolo-seg", "fused"], default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="submit a claim and process it")
    run.add_argument("--policy", required=True, help="policy number, e.g. P-1001")
    run.add_argument("--description", default="", help="what happened")
    run.add_argument("photos", nargs="+", type=Path, help="photo files")

    resume = sub.add_parser("resume", help="finish processing a claim that was interrupted")
    resume.add_argument("claim_id", type=UUID)

    show = sub.add_parser("show", help="print a claim's decision and audit trail")
    show.add_argument("claim_id", type=UUID)

    verify = sub.add_parser("verify", help="verify a claim's hash chain")
    verify.add_argument("claim_id", type=UUID)

    evaluate = sub.add_parser("eval-triage", help="score the pipeline on golden claims")
    evaluate.add_argument("--golden", type=Path, required=True, help="golden claims .jsonl")
    evaluate.add_argument("--report", type=Path, required=True, help="Markdown report to write")
    evaluate.add_argument(
        "--what-if",
        type=_thresholds,
        default=[],
        help="comma-separated confidence thresholds, e.g. 0.25,0.40,0.55",
    )

    data = sub.add_parser("data", help="fetch and build datasets")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="download a raw data source into data/raw/")
    fetch.add_argument("source_id")
    build = data_sub.add_parser("build", help="build a dataset from raw sources")
    build.add_argument("dataset_id")
    autolabel = data_sub.add_parser("autolabel", help="propose part masks with foundation models")
    autolabel.add_argument("job_id")

    review = sub.add_parser("review", help="human review in FiftyOne")
    review.add_argument("action", choices=["launch", "export", "apply"])
    review.add_argument("target", choices=["parts", "golden"])
    review.add_argument("--job", default="fusion-eval-v1", help="auto-label job id")
    review.add_argument("--golden", type=Path, default=Path("evals/golden/v1/claims.jsonl"))
    review.add_argument("--reviewer", default="reviewer")
    add_train_parser(sub)
    return parser


def make_deps(
    store: SQLiteEventStore, blobs: BlobStore, config_dir: Path, detector: Detector
) -> PipelineDeps:
    return PipelineDeps(
        store=store,
        blobs=blobs,
        detector=detector,
        policies=load_policies(config_dir / "policies.toml"),
        rate_card=load_rate_card(config_dir / "rate_card.toml"),
        decision_config=load_decision_config(config_dir / "decision_policy.toml"),
        agent=StubTriageAgent(),
    )


def format_summary(state: ClaimState) -> str:
    lines = [f"Claim: {state.claim_id}"]
    if state.decision is not None:
        lines.append(f"Route: {state.decision.route.value} ({state.decision.rule_id})")
        lines.append(f"Reason: {state.decision.reason}")
    found = ", ".join(f"{f.damage_type.value} {f.confidence:.2f}" for f in state.findings)
    lines.append(f"Findings: {len(state.findings)} ({found or 'none'})")
    if state.cost_estimate is not None:
        est = state.cost_estimate
        lines.append(f"Cost estimate: ${est.low:,}-${est.high:,} {est.currency}")
    for photo in state.photos.values():
        if photo.reject_reason:
            lines.append(f"Rejected {photo.photo_id} ({photo.filename}): {photo.reject_reason}")
    return "\n".join(lines)


def format_audit_trail(events: Sequence[ClaimEvent]) -> str:
    rows = ["Audit trail:"]
    for event in events:
        actor = f"{event.actor.kind.value}:{event.actor.name}"
        rows.append(f"  #{event.seq:<3} {event.type:<18} {actor:<22} {event.hash[:12]}")
    return "\n".join(rows)


def main(
    argv: Sequence[str] | None = None,
    *,
    detector_factory: DetectorFactory = _default_detector,
    labeller_factory: LabellerFactory = _grounded_sam,
    trainer_factory: TrainerFactory = _ultralytics_trainer,
    segmenter_factory: SegmenterFactory = _ultralytics_segmenter,
    exporter: ExporterFactory = _export_onnx,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(
            args, detector_factory, labeller_factory, trainer_factory, segmenter_factory, exporter
        )
    except DetectorUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


class DetectorUnavailableError(RuntimeError):
    pass


def _dispatch(
    args: argparse.Namespace,
    detector_factory: DetectorFactory,
    labeller_factory: LabellerFactory,
    trainer_factory: TrainerFactory,
    segmenter_factory: SegmenterFactory,
    exporter: ExporterFactory,
) -> int:
    if args.command == "eval-triage":
        return _eval_triage(args, detector_factory)
    if args.command == "data":
        return _data(args, labeller_factory)
    if args.command == "review":
        return _review(args)
    if args.command == "train":
        return run_train_command(
            args,
            trainer_factory=trainer_factory,
            segmenter_factory=segmenter_factory,
            exporter=exporter,
        )
    store = SQLiteEventStore(args.db)
    try:
        if args.command == "run":
            return _run(args, store, detector_factory)
        if args.command == "resume":
            return _resume(args, store, detector_factory)
        return _inspect(args, store)
    finally:
        store.close()


def _run(args: argparse.Namespace, store: SQLiteEventStore, factory: DetectorFactory) -> int:
    blobs = BlobStore(args.blobs)
    deps = make_deps(store, blobs, args.config, _make_detector(args, factory))
    claim_id = submit_claim(
        store, blobs, policy_id=args.policy, description=args.description, photo_paths=args.photos
    )
    print(
        f"Submitted claim {claim_id} (if interrupted: claimlens resume {claim_id})",
        file=sys.stderr,
        flush=True,
    )
    process_claim(claim_id, deps)
    print(format_summary(fold(store.load(claim_id))))
    return 0


def _resume(args: argparse.Namespace, store: SQLiteEventStore, factory: DetectorFactory) -> int:
    try:
        store.load(args.claim_id)
    except ClaimNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ChainIntegrityError as exc:
        print(f"error: audit log failed verification: {exc}", file=sys.stderr)
        return 1
    deps = make_deps(store, BlobStore(args.blobs), args.config, _make_detector(args, factory))
    process_claim(args.claim_id, deps)
    print(format_summary(fold(store.load(args.claim_id))))
    return 0


def _inspect(args: argparse.Namespace, store: SQLiteEventStore) -> int:
    try:
        events = store.load(args.claim_id)
    except ClaimNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ChainIntegrityError as exc:
        print(f"error: audit log failed verification: {exc}", file=sys.stderr)
        return 1
    if args.command == "verify":
        print(f"Chain OK: {len(events)} events")
        return 0
    print(format_summary(fold(events)))
    print(format_audit_trail(events))
    return 0


def _eval_triage(args: argparse.Namespace, factory: DetectorFactory) -> int:
    cases = load_golden(args.golden)
    detector = _make_detector(args, factory)

    def make(case_dir: Path) -> PipelineDeps:
        store = SQLiteEventStore(case_dir / "claims.db")
        return make_deps(store, BlobStore(case_dir / "blobs"), args.config, detector)

    with tempfile.TemporaryDirectory() as workdir:
        results = run_triage_eval(cases, make, repo_root=Path.cwd(), workdir=Path(workdir))
    metrics = compute_triage_metrics([(r.expected, r.predicted) for r in results])
    thresholds: list[float] = args.what_if
    decision_config = load_decision_config(args.config / "decision_policy.toml")
    what_if = what_if_thresholds(results, decision_config, thresholds)
    meta = ReportMeta(
        golden_path=args.golden.as_posix(),
        model_version=detector.model_version,
        agent_version=StubTriageAgent.agent_version,
        decision_policy_version=decision_config.version,
        generated_on=date.today(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(cases, results, metrics, meta, what_if=what_if)
    args.report.write_text(report, encoding="utf-8")
    recall = "n/a" if metrics.escalation_recall is None else f"{metrics.escalation_recall:.2f}"
    print(
        f"Cases: {metrics.total}  Route accuracy: {metrics.route_accuracy:.2f}  "
        f"Escalation recall: {recall}"
    )
    print(f"Report written to {args.report}")
    return 0


def _data(args: argparse.Namespace, labeller_factory: LabellerFactory) -> int:
    repo_root = Path.cwd()
    if args.data_command == "autolabel":
        return _autolabel(args, repo_root, labeller_factory)
    try:
        config = load_data_config(args.config / "datasets.toml")
    except (OSError, ValueError) as exc:
        print(f"error: cannot read data config: {exc}", file=sys.stderr)
        return 2
    if args.data_command == "fetch":
        if args.source_id not in config.sources:
            known = ", ".join(sorted(config.sources))
            print(f"error: unknown source {args.source_id!r} (known: {known})", file=sys.stderr)
            return 2
        try:
            dest = fetch_source(
                config.sources[args.source_id],
                repo_root / "data" / "raw",
                api_key=read_secret("ROBOFLOW_API_KEY"),
            )
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Fetched {args.source_id} into {dest}")
        return 0
    if args.dataset_id not in config.datasets:
        known = ", ".join(sorted(config.datasets))
        print(f"error: unknown dataset {args.dataset_id!r} (known: {known})", file=sys.stderr)
        return 2
    try:
        result = build_dataset(args.dataset_id, repo_root=repo_root, config_dir=args.config)
    except DataContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for issue in exc.report.errors[:10]:
            print(f"  {issue.image_id}: {issue.code}: {issue.message}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Built {result.dataset_id}: images {result.stats['images']}")
    print(f"Leaks prevented: {result.stats['leaks_prevented']}")
    return 0


def _autolabel(args: argparse.Namespace, repo_root: Path, factory: LabellerFactory) -> int:
    jobs = load_autolabel_jobs(args.config / "autolabel.toml")
    if args.job_id not in jobs:
        known = ", ".join(sorted(jobs))
        print(f"error: unknown auto-label job {args.job_id!r} (known: {known})", file=sys.stderr)
        return 2
    job = jobs[args.job_id]

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} images", flush=True)

    report = run_autolabel(
        job,
        repo_root=repo_root,
        labeller=factory(job),
        part_groups=load_part_groups(args.config / "taxonomy.toml"),
        progress=progress,
    )
    print(f"Auto-labelled {report['images']} images: proposals {report['proposals']}")
    print(f"Dropped: {report['dropped']}")
    return 0


def _review(args: argparse.Namespace) -> int:
    from claimlens.review import fiftyone_app

    repo_root = Path.cwd()
    golden_version = args.golden.parent.name
    parts_file = repo_root / "data" / "interim" / args.job / "autolabels.jsonl"
    parts_review = repo_root / "reviews" / f"{args.job}.json"
    golden_review = repo_root / "reviews" / f"golden-{golden_version}.json"
    try:
        if args.action == "launch" and args.target == "parts":
            damage_file = repo_root / "data" / "interim" / "damage-v1" / "records.jsonl"
            fiftyone_app.launch_parts_review(
                read_records(parts_file),
                repo_root=repo_root,
                name=f"claimlens-{args.job}",
                damage={r.image_id: r for r in read_records(damage_file)},
                existing=(
                    read_review(parts_review, PartReview).decisions
                    if parts_review.exists()
                    else None
                ),
            )
        elif args.action == "launch":
            fiftyone_app.launch_golden_review(
                load_golden(args.golden),
                repo_root=repo_root,
                name=f"claimlens-golden-{golden_version}",
            )
        elif args.action == "export" and args.target == "parts":
            review = fiftyone_app.export_parts_review(
                f"claimlens-{args.job}", job_id=args.job, reviewer=args.reviewer
            ).model_copy(update={"proposals_sha256": file_sha256(parts_file)})
            if parts_review.exists():
                review = merge_part_reviews(read_review(parts_review, PartReview), review)
            write_review(parts_review, review)
            print(f"Wrote {len(review.decisions)} decisions to {parts_review}")
        elif args.action == "export":
            golden = fiftyone_app.export_golden_review(
                f"claimlens-golden-{golden_version}", reviewer=args.reviewer
            )
            write_review(golden_review, golden)
            print(f"Wrote {len(golden.decisions)} decisions to {golden_review}")
        elif args.target == "parts":
            review = read_review(parts_review, PartReview)
            check_part_review(review, job_id=args.job, proposals_sha256=file_sha256(parts_file))
            kept, counts = apply_part_review(read_records(parts_file), review)
            out = repo_root / "data" / "processed" / args.job / "parts.jsonl"
            write_records(out, kept)
            report = {"job": args.job, "images": len(kept), "counts": counts}
            report_path = repo_root / "reports" / "data" / f"{args.job}-review.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
            )
            approved, rejected = counts.get("approved", 0), counts.get("rejected", 0)
            print(f"{approved} approved, {rejected} rejected; {len(kept)} images in {out}")
        else:
            cases = apply_golden_review(
                load_golden(args.golden), read_review(golden_review, GoldenReview)
            )
            write_golden(args.golden, cases)
            print(f"{sum(c.reviewed for c in cases)} of {len(cases)} golden cases reviewed")
    except fiftyone_app.ReviewUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
