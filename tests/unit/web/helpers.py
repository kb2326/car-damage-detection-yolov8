"""Event logs for web tests, built by running the real pipeline with fakes."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from PIL import Image

from claimlens.domain import BoundingBox, DamageFinding, DamageType
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.workflow import PipelineDeps, process_claim
from tests.fakes import FakeDetector, make_test_deps


def image(path: Path, color: tuple[int, int, int] = (200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)
    return path


def finding(photo_id: str = "p1") -> DamageFinding:
    return DamageFinding(
        photo_id=photo_id,
        damage_type=DamageType.DENT,
        confidence=0.9,
        bbox=BoundingBox(x1=64, y1=48, x2=320, y2=240),
        image_area_fraction=0.16,
        part="back_bumper",
        part_area_ratio=0.06,
    )


def deps(tmp_path: Path, store: SQLiteEventStore) -> PipelineDeps:
    detector = FakeDetector(findings={"p1": [finding("p1")]})
    return make_test_deps(tmp_path, store, detector)


def decided_claim(
    tmp_path: Path, store: SQLiteEventStore, color: tuple[int, int, int] = (200, 30, 30)
) -> UUID:
    d = deps(tmp_path, store)
    claim = submit_claim(
        store,
        d.blobs,
        policy_id="P-1001",
        description="Reversed into a bollard <b>slowly</b>",
        photo_paths=[image(tmp_path / "in" / f"{'-'.join(map(str, color))}.png", color)],
    )
    process_claim(claim, d)
    return claim
