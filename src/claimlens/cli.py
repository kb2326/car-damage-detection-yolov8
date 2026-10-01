"""Command-line interface: `claimlens run | show | verify`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import UUID

from claimlens.agent.stub import StubTriageAgent
from claimlens.blobs import BlobStore
from claimlens.decision import load_decision_config
from claimlens.events.envelope import ChainIntegrityError, ClaimEvent
from claimlens.events.projection import ClaimState, fold
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card
from claimlens.vision.base import Detector
from claimlens.workflow import PipelineDeps, process_claim

DEFAULT_WEIGHTS = Path("models/legacy/yolov8n-cardamage-v6.pt")
DetectorFactory = Callable[[Path], Detector]


def _legacy_detector(weights: Path) -> Detector:
    from claimlens.vision.legacy_yolo import LegacyYoloDetector

    return LegacyYoloDetector(weights)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claimlens", description="ClaimLens claims triage")
    parser.add_argument("--db", type=Path, default=Path("var/claimlens.db"))
    parser.add_argument("--blobs", type=Path, default=Path("var/blobs"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="submit a claim and process it")
    run.add_argument("--policy", required=True, help="policy number, e.g. P-1001")
    run.add_argument("--description", default="", help="what happened")
    run.add_argument("photos", nargs="+", type=Path, help="photo files")

    show = sub.add_parser("show", help="print a claim's decision and audit trail")
    show.add_argument("claim_id", type=UUID)

    verify = sub.add_parser("verify", help="verify a claim's hash chain")
    verify.add_argument("claim_id", type=UUID)
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
    argv: Sequence[str] | None = None, *, detector_factory: DetectorFactory = _legacy_detector
) -> int:
    args = build_parser().parse_args(argv)
    store = SQLiteEventStore(args.db)
    try:
        if args.command == "run":
            return _run(args, store, detector_factory)
        return _inspect(args, store)
    finally:
        store.close()


def _run(args: argparse.Namespace, store: SQLiteEventStore, factory: DetectorFactory) -> int:
    blobs = BlobStore(args.blobs)
    deps = make_deps(store, blobs, args.config, factory(args.weights))
    claim_id = submit_claim(
        store, blobs, policy_id=args.policy, description=args.description, photo_paths=args.photos
    )
    process_claim(claim_id, deps)
    print(format_summary(fold(store.load(claim_id))))
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
