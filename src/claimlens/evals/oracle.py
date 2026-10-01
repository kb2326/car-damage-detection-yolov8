"""Expected routes from ground-truth labels: what a perfect perception model would produce."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from claimlens.decision import DecisionConfig, decide
from claimlens.domain import (
    AgentRecommendation,
    BoundingBox,
    Confidence,
    Coverage,
    DamageFinding,
    Route,
)
from claimlens.events.projection import ClaimState, PhotoRecord, PhotoStatus
from claimlens.pricing import RateCard, estimate_cost
from claimlens.vision.base import normalize_class_name


def findings_from_yolo_label(
    text: str, class_names: Sequence[str], photo_id: str, image_width: int, image_height: int
) -> list[DamageFinding]:
    """Parse YOLO box (`cls cx cy w h`) or polygon (`cls x1 y1 x2 y2 ...`) lines."""
    findings: list[DamageFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        class_index = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        if len(coords) < 4 or len(coords) % 2:
            raise ValueError(f"line {line_no}: expected box or polygon coordinates")
        if not 0 <= class_index < len(class_names):
            raise ValueError(f"line {line_no}: class index {class_index} is out of range")
        if len(coords) == 4:
            cx, cy, w, h = coords
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, 1.0), min(y2, 1.0)
        findings.append(
            DamageFinding(
                photo_id=photo_id,
                damage_type=normalize_class_name(class_names[class_index]),
                confidence=1.0,
                bbox=BoundingBox(
                    x1=x1 * image_width,
                    y1=y1 * image_height,
                    x2=x2 * image_width,
                    y2=y2 * image_height,
                ),
                image_area_fraction=(x2 - x1) * (y2 - y1),
            )
        )
    return findings


def oracle_route(
    findings: Sequence[DamageFinding],
    coverage: Coverage,
    card: RateCard,
    config: DecisionConfig,
) -> Route:
    """Apply the real decision policy to perfect evidence and a perfect agent."""
    state = ClaimState(claim_id=UUID(int=0), policy_id=coverage.policy_id, description="oracle")
    state.photos["p1"] = PhotoRecord(
        photo_id="p1",
        sha256="oracle",
        filename="oracle",
        blob_name="oracle",
        status=PhotoStatus.ACCEPTED,
    )
    state.findings = list(findings)
    state.integrity_checked = True
    state.cost_estimate = estimate_cost(findings, card)
    state.coverage = coverage
    state.recommendation = AgentRecommendation(
        route_suggestion=Route.FAST_TRACK if findings else Route.ADJUSTER_REVIEW,
        confidence=Confidence.HIGH if findings else Confidence.LOW,
        rationale="oracle",
        citations=(),
        open_questions=() if findings else ("no labelled damage",),
    )
    return decide(state, config).route
