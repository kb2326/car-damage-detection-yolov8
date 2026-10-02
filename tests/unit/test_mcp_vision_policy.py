from pathlib import Path

from claimlens.data.taxonomy import load_part_groups
from claimlens.mcp.policy_admin import build_policy_admin
from claimlens.mcp.profiles import load_profiles
from claimlens.mcp.vision import VisionDeps, build_vision
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card
from claimlens.quality import QualityConfig
from claimlens.vision.instances import SegInstance, Segmentation
from tests.fakes import FakeDetector
from tests.mcp_helpers import MemoryAudit, call, error_text, tool_names

ROOT = Path(__file__).resolve().parents[2]
PROFILES = load_profiles(ROOT / "config" / "agents.toml")
PHOTO = "tests/fixtures/images/dent_1.jpg"


class _Parts:
    model_version = "fake-parts"

    def segment(self, image_path: Path) -> Segmentation:
        door = SegInstance(
            label="front_left_door",
            confidence=0.9,
            box_xyxy=(0, 0, 1, 1),
            polygon_xyn=(0.1, 0.1, 0.5, 0.1, 0.5, 0.5),
        )
        return Segmentation(instances=(door,), width=10, height=10)


def _deps() -> VisionDeps:
    return VisionDeps(
        damage=FakeDetector,
        parts=_Parts,
        groups=load_part_groups(ROOT / "config" / "taxonomy.toml"),
        rate_card=load_rate_card(ROOT / "config" / "rate_card.toml"),
        quality=QualityConfig(),
        roots=(ROOT / "tests" / "fixtures", ROOT / "data"),
    )


def test_profiles_see_only_their_tools() -> None:
    assert tool_names(build_vision(PROFILES["intake"], _deps(), MemoryAudit())) == [
        "assess_quality"
    ]
    assert tool_names(build_vision(PROFILES["demo"], _deps(), MemoryAudit())) == [
        "assess_quality",
        "segment_damage",
        "segment_parts",
    ]


def test_segment_damage_returns_findings_and_audits(monkeypatch: object) -> None:
    import os

    os.chdir(ROOT)
    audit = MemoryAudit()
    result = call(
        build_vision(PROFILES["demo"], _deps(), audit), "segment_damage", {"photo": PHOTO}
    )
    assert not result.is_error
    report = result.structured_content
    assert report is not None
    assert report["model_version"] == "fake-detector-v1"
    assert report["findings"][0]["type"] == "dent"
    assert report["findings"][0]["severity"] == "minor"
    assert [(r.tool, r.outcome) for r in audit.records] == [("segment_damage", "ok")]


def test_segment_parts_maps_groups() -> None:
    import os

    os.chdir(ROOT)
    result = call(
        build_vision(PROFILES["demo"], _deps(), MemoryAudit()), "segment_parts", {"photo": PHOTO}
    )
    assert result.structured_content is not None
    assert result.structured_content["parts"][0]["group"] == "door"


def test_unadvertised_tool_is_refused_and_audited() -> None:
    audit = MemoryAudit()
    result = call(
        build_vision(PROFILES["intake"], _deps(), audit), "segment_damage", {"photo": PHOTO}
    )
    assert result.is_error
    assert "ScopeDenied" in error_text(result)
    assert [(r.tool, r.outcome) for r in audit.records] == [("segment_damage", "denied")]


def test_unsafe_photo_path_is_an_error_result() -> None:
    import os

    os.chdir(ROOT)
    audit = MemoryAudit()
    result = call(
        build_vision(PROFILES["demo"], _deps(), audit), "segment_damage", {"photo": ".env"}
    )
    assert result.is_error
    assert "PathRejected" in error_text(result)
    assert audit.records[-1].outcome == "error"


def test_policy_admin_lookups() -> None:
    policies = load_policies(ROOT / "config" / "policies.toml")
    server = build_policy_admin(PROFILES["demo"], policies, MemoryAudit())
    coverage = call(server, "get_coverage", {"policy_id": "P-1001"}).structured_content
    assert coverage == {"found": True, "active": True, "collision": True, "deductible": 250}
    missing = call(server, "get_policy", {"policy_id": "P-9999"}).structured_content
    assert missing is not None
    assert missing["found"] is False
    assert tool_names(build_policy_admin(PROFILES["intake"], policies, MemoryAudit())) == [
        "get_policy"
    ]
