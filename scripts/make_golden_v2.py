"""Build evals/golden/v2/claims.jsonl: golden v1 unchanged, then 53 narrative cases.

Each narrative case reuses one photo from a golden v1 case that passes rules R1-R6 with the fused
detector, so only the claimant's story decides the route. `photo_kind` in narratives.toml picks a
photo whose damage fits the story. Deterministic: run it again and the file is identical.

    uv run python scripts/make_golden_v2.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, check_golden_citations, load_golden, write_golden
from claimlens.knowledge.clauses import load_wordings
from claimlens.policy import load_policies

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "evals" / "golden" / "v1" / "claims.jsonl"
V2 = ROOT / "evals" / "golden" / "v2" / "claims.jsonl"
NARRATIVES = ROOT / "evals" / "golden" / "v2" / "narratives.toml"

# Golden v1 cases the stub fast-tracks with the fused detector (2026-10-03), by what the detector
# finds on them. g049 has two copies of one photo; only the first is used.
PHOTO_SOURCES: dict[str, tuple[str, ...]] = {
    "dent": ("g028",),
    "lamp": ("g049",),
    "tyre": ("g103", "g108", "g115"),
    "glass": ("g104", "g107", "g146", "g147"),
}


def main() -> int:
    v1 = load_golden(V1)
    by_id = {c.case_id: c for c in v1}
    narratives = tomllib.loads(NARRATIVES.read_text(encoding="utf-8"))["case"]
    used = dict.fromkeys(PHOTO_SOURCES, 0)
    new: list[GoldenClaim] = []
    for n, item in enumerate(narratives, start=1):
        kind = item["photo_kind"]
        sources = PHOTO_SOURCES[kind]
        source = by_id[sources[used[kind] % len(sources)]]
        used[kind] += 1
        benign = item["scenario"] == "benign_distractor"
        new.append(
            GoldenClaim(
                case_id=f"n{n:03d}",
                scenario=item["scenario"],
                policy_id=item["policy_id"],
                description=item["description"],
                photos=(source.photos[0],),
                expected_route=Route.FAST_TRACK if benign else Route.ADJUSTER_REVIEW,
                label_source="ai_authored_narrative",
                notes=f"photo from {source.case_id} ({kind})",
                narrative=True,
                expected_citations=tuple(item.get("expected_citations", ())),
                must_not_fast_track_reason="" if benign else item["reason"],
            )
        )
    policies = load_policies(ROOT / "config" / "policies.toml")
    wording_of_policy = {}
    for case in new:
        record = policies.get_record(case.policy_id)
        if record is None or not policies.get_coverage(case.policy_id).confirmed:
            print(f"{case.case_id}: policy {case.policy_id} lacks active collision cover")
            return 1
        wording_of_policy[case.policy_id] = record.wording
    clauses = {c.clause_id: c.wording for c in load_wordings(ROOT / "knowledge" / "policies")}
    problems = check_golden_citations(new, wording_of_policy, clauses.get)
    if problems:
        print("\n".join(problems))
        return 1
    v1_text = V1.read_text(encoding="utf-8")
    write_golden(V2, new)
    V2.write_text(v1_text + V2.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    print(f"wrote {97 + len(new)} cases to {V2.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
