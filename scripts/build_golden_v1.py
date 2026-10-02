"""Build golden claims v1: v0 without near-duplicate photos, plus 50 frozen CarDD test claims.

Run from the repository root after `dvc repro`:  uv run python scripts/build_golden_v1.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from claimlens.data.dedupe import find_clusters, hash_file
from claimlens.data.records import read_records
from claimlens.decision import load_decision_config
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, load_golden, write_golden
from claimlens.evals.oracle import findings_from_record, oracle_route
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card

V0 = Path("evals/golden/v0/claims.jsonl")
OUT = Path("evals/golden/v1/claims.jsonl")
INTERIM = Path("data/interim/damage-v1")
ACTIVE_POLICIES = ["P-1001", "P-1002", "P-1003", "P-1004", "P-1005", "P-1006"]
DESCRIPTION = "Damage reported after a low-speed collision."
SEED = 20261001
FAST_TRACK_CASES = 20
OTHER_CASES = 30


def _readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def dedupe_v0(cases: list[GoldenClaim]) -> tuple[list[GoldenClaim], list[str]]:
    """Drop oracle cases whose first photo repeats an earlier one in the same scenario.

    Scenario cases are left alone: their photos are deliberate test inputs (for example two
    blank "unusable" images that share a perceptual hash but test different failures).
    """
    hashes = [
        hash_file(c.case_id, Path(c.photos[0]))
        for c in cases
        if c.label_source == "oracle" and _readable(Path(c.photos[0]))
    ]
    clusters = find_clusters(hashes, max_distance=6)
    seen: set[tuple[str, int]] = set()
    kept: list[GoldenClaim] = []
    dropped: list[str] = []
    for case in cases:
        cluster = clusters.get(case.case_id)
        key = (case.scenario, cluster) if cluster is not None else None
        if key is not None and key in seen:
            dropped.append(case.case_id)
            continue
        if key is not None:
            seen.add(key)
        kept.append(case)
    return kept, dropped


def main() -> int:
    v0, dropped = dedupe_v0(load_golden(V0))
    records = read_records(INTERIM / "records.jsonl")
    splits: dict[str, str] = json.loads((INTERIM / "splits.json").read_text(encoding="utf-8"))
    test = sorted((r for r in records if splits[r.image_id] == "test"), key=lambda r: r.image_id)
    random.Random(SEED).shuffle(test)

    policies = load_policies(Path("config/policies.toml"))
    card = load_rate_card(Path("config/rate_card.toml"))
    config = load_decision_config(Path("config/decision_policy.toml"))
    fast: list[GoldenClaim] = []
    other: list[GoldenClaim] = []
    for i, record in enumerate(test):
        if len(fast) >= FAST_TRACK_CASES and len(other) >= OTHER_CASES:
            break
        policy = ACTIVE_POLICIES[i % len(ACTIVE_POLICIES)]
        coverage = policies.get_coverage(policy)
        route = oracle_route(findings_from_record(record), coverage, card, config)
        bucket = fast if route is Route.FAST_TRACK else other
        limit = FAST_TRACK_CASES if route is Route.FAST_TRACK else OTHER_CASES
        if len(bucket) < limit:
            bucket.append(
                GoldenClaim(
                    case_id="pending",
                    scenario="cardd_test_oracle",
                    policy_id=policy,
                    description=DESCRIPTION,
                    photos=(record.path,),
                    expected_route=route,
                    label_source="oracle",
                )
            )
    new = sorted(fast + other, key=lambda c: c.photos[0])
    numbered = [c.model_copy(update={"case_id": f"g{101 + i:03d}"}) for i, c in enumerate(new)]

    write_golden(OUT, v0 + numbered)
    print(f"Kept {len(v0)} v0 cases (dropped duplicates: {', '.join(dropped) or 'none'})")
    print(f"Added {len(fast)} fast-track and {len(other)} other CarDD test cases -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
