# Golden claims v2

150 cases: the 97 cases of [golden v1](../v1/README.md), unchanged, then 53 **narrative cases**
(`n001`–`n053`) where the claimant's story, not the photo, decides the route.

## Why

Golden v1 tests the vision models and the rules. Its descriptions are generic ("Damage reported
after a low-speed collision"), so it cannot show what the LLM triage agent adds. The narrative
cases can: a story that triggers a policy exclusion must go to a person, and a harmless story with
words that sound like an exclusion must not.

## How the cases are built

- Every narrative case reuses **one photo from a golden v1 case that passes rules R1–R6** with the
  fused detector (no new images). Only 9 such photos exist: 1 dent, 1 broken lamp, 3 flat tyres
  and 4 shattered glass. `photo_kind` in `narratives.toml` picks a photo whose damage fits the
  story, so only the `story_mismatch` cases mismatch on purpose.
- Every policy used has active collision cover, so the rules cannot decide on cover alone.
- `expected_citations`: clause ids a good answer should cite (any one is enough), checked against
  the claimant's own wording.

| Scenario | Cases | Expected route |
|---|---|---|
| `exclusion_commercial` (delivery, ride-hailing, rental) | 7 | ADJUSTER_REVIEW |
| `exclusion_racing` (track day, street race, timed event) | 5 | ADJUSTER_REVIEW |
| `exclusion_driver` (no licence, alcohol, drugs) | 5 | ADJUSTER_REVIEW |
| `exclusion_intentional_or_wear` | 5 | ADJUSTER_REVIEW |
| `late_report` (more than 30 days) | 5 | ADJUSTER_REVIEW |
| `story_mismatch` (hail, rollover, flood vs one small damage) | 6 | ADJUSTER_REVIEW |
| `cover_missing_basic` (theft or vandalism on the basic wording) | 5 | ADJUSTER_REVIEW |
| `prompt_injection` (instructions in the description) | 5 | ADJUSTER_REVIEW |
| `benign_distractor` ("I deliver my kids to school", "racing to the shop") | 10 | FAST_TRACK |

## Labels

`label_source = "ai_authored_narrative"`: the stories were **written by Claude from the fictional
policy wordings and are not yet reviewed by a person** (`reviewed: false`). Reports say so.

## Regenerate

`uv run python scripts/make_golden_v2.py` (deterministic). Edit `narratives.toml`, never the
`.jsonl` directly; `tests/unit/test_golden_v2.py` checks the result.
