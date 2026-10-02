"""MCP `vision` server: our damage and part models, plus the photo quality gate."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from claimlens.data.taxonomy import PartGroups
from claimlens.mcp.base import AuditSink, ScopedServer, tool_errors
from claimlens.mcp.guard import safe_photo_path
from claimlens.mcp.profiles import Profile
from claimlens.mcp.schemas import DamageReport, FindingOut, PartOut, PartsReport, QualityReport
from claimlens.pricing import RateCard, severity_of
from claimlens.quality import QualityConfig, check_quality
from claimlens.vision.base import Detector
from claimlens.vision.instances import Segmenter


@dataclass
class VisionDeps:
    damage: Callable[[], Detector]
    parts: Callable[[], Segmenter]
    groups: PartGroups
    rate_card: RateCard
    quality: QualityConfig
    roots: Sequence[Path]


def build_vision(profile: Profile, deps: VisionDeps, audit: AuditSink) -> ScopedServer:
    server = ScopedServer("vision", profile, audit)
    damage_cache: list[Detector] = []
    parts_cache: list[Segmenter] = []

    def damage_model() -> Detector:
        if not damage_cache:  # load lazily: the first call pays the model start-up
            damage_cache.append(deps.damage())
        return damage_cache[0]

    def parts_model() -> Segmenter:
        if not parts_cache:
            parts_cache.append(deps.parts())
        return parts_cache[0]

    def segment_damage(photo: str) -> DamageReport:
        with tool_errors():
            path = safe_photo_path(photo, deps.roots)
            model = damage_model()
            findings = model.detect(path, "photo")
            return DamageReport(
                model_version=model.model_version,
                findings=[
                    FindingOut(
                        type=f.damage_type.value,
                        confidence=round(f.confidence, 3),
                        part=f.part,
                        part_area_ratio=f.part_area_ratio,
                        severity=severity_of(f, deps.rate_card).value,
                    )
                    for f in findings
                ],
            )

    def segment_parts(photo: str) -> PartsReport:
        with tool_errors():
            path = safe_photo_path(photo, deps.roots)
            model = parts_model()
            result = model.segment(path)
            return PartsReport(
                model_version=model.model_version,
                parts=[
                    PartOut(
                        label=p.label,
                        group=deps.groups.group_of(p.label),
                        confidence=round(p.confidence, 3),
                    )
                    for p in result.instances
                ],
            )

    def assess_quality(photo: str) -> QualityReport:
        with tool_errors():
            result = check_quality(safe_photo_path(photo, deps.roots), deps.quality)
            return QualityReport(accepted=result.ok, reason=result.reason)

    server.register(
        segment_damage, "segment_damage", "Find damage in a car photo (type, part, severity)."
    )
    server.register(segment_parts, "segment_parts", "Find car parts in a photo.")
    server.register(
        assess_quality, "assess_quality", "Check whether a photo is usable for a claim."
    )
    return server
