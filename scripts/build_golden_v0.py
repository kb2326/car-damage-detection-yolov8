"""Build golden claims v0 from the Roboflow test split. Run from the repository root:

uv run python scripts/build_golden_v0.py
"""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from claimlens.decision import load_decision_config
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, PriorClaim, write_golden
from claimlens.evals.oracle import findings_from_yolo_label, oracle_route
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card

DATASET = Path("data/raw/legacy-course-subset")
OUT_DIR = Path("evals/golden/v0")
ACTIVE_POLICIES = ["P-1001", "P-1002", "P-1003", "P-1004", "P-1005", "P-1006"]
DESCRIPTION = "Damage reported after a low-speed collision."


def _make_assets(assets: Path) -> list[Path]:
    assets.mkdir(parents=True, exist_ok=True)
    tiny = assets / "tiny.png"
    Image.new("RGB", (100, 100), (120, 120, 120)).save(tiny, format="PNG")
    strip = assets / "thin_strip.png"
    Image.new("RGB", (1200, 200), (90, 90, 90)).save(strip, format="PNG")
    not_image = assets / "not_an_image.jpg"
    not_image.write_text("This file is text, not a photo.\n", encoding="utf-8")
    return [tiny, not_image, strip]


def main() -> int:
    data_yaml = (DATASET / "data.yaml").read_text(encoding="utf-8")
    names: list[str] = yaml.safe_load(data_yaml)["names"]
    images = sorted((DATASET / "test" / "images").glob("*.jpg"))
    if len(images) < 47:
        raise SystemExit(f"expected at least 47 test images in {DATASET}, found {len(images)}")
    policies = load_policies(Path("config/policies.toml"))
    card = load_rate_card(Path("config/rate_card.toml"))
    config = load_decision_config(Path("config/decision_policy.toml"))

    def expected_for(image: Path, policy_id: str) -> Route:
        label = DATASET / "test" / "labels" / f"{image.stem}.txt"
        with Image.open(image) as opened:
            width, height = opened.size
        text = label.read_text(encoding="utf-8")
        findings = findings_from_yolo_label(text, names, "p1", width, height)
        return oracle_route(findings, policies.get_coverage(policy_id), card, config)

    cases: list[GoldenClaim] = []

    def add(**fields: object) -> None:
        case_id = f"g{len(cases) + 1:03d}"
        cases.append(GoldenClaim.model_validate({"case_id": case_id, **fields}))

    for i in range(30):
        image, policy = images[i], ACTIVE_POLICIES[i % len(ACTIVE_POLICIES)]
        add(
            scenario="oracle_single_photo",
            policy_id=policy,
            description=DESCRIPTION,
            photos=[image.as_posix()],
            expected_route=expected_for(image, policy),
            label_source="oracle",
        )
    for image in images[30:35]:
        add(
            scenario="lapsed_policy",
            policy_id="P-2001",
            description=DESCRIPTION,
            photos=[image.as_posix()],
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        )
    for image in images[35:40]:
        add(
            scenario="no_collision_cover",
            policy_id="P-2002",
            description=DESCRIPTION,
            photos=[image.as_posix()],
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        )
    for image in images[40:45]:
        add(
            scenario="photo_reuse",
            policy_id="P-1002",
            description=DESCRIPTION,
            photos=[image.as_posix()],
            prior_claims=[PriorClaim(policy_id="P-1001", photos=(image.as_posix(),))],
            expected_route=Route.FRAUD_REVIEW,
            label_source="scenario",
        )
    for asset in _make_assets(OUT_DIR / "assets"):
        add(
            scenario="unusable_photo",
            policy_id="P-1001",
            description=DESCRIPTION,
            photos=[asset.as_posix()],
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        )
    for i, image in enumerate(images[45:47]):
        policy = ACTIVE_POLICIES[i]
        add(
            scenario="duplicate_in_claim",
            policy_id=policy,
            description=DESCRIPTION,
            photos=[image.as_posix(), image.as_posix()],
            expected_route=expected_for(image, policy),
            label_source="oracle",
        )

    write_golden(OUT_DIR / "claims.jsonl", cases)
    print(f"Wrote {len(cases)} golden claims to {OUT_DIR / 'claims.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
