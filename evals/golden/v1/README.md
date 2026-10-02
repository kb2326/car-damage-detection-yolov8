# Golden claims v1

Built by `scripts/build_golden_v1.py`: 97 cases (30 FAST_TRACK, 62 ADJUSTER_REVIEW, 5 FRAUD_REVIEW).

- **47 v0 cases** (ids `g001`-`g050`) with near-duplicate photos removed: when two
  oracle-labelled cases in the same scenario use near-identical first photos (perceptual hash
  distance at most 6), only the first is kept. This dropped `g007` and `g008` (Roboflow copies of
  `crashed454`, same as `g006`) and `g020` (a copy of `dent1336`, same as `g018`). Scenario cases
  are never deduplicated: their photos are deliberate test inputs.
- **50 new cases** (ids `g101`-`g150`) from the frozen CarDD test split of `damage-v1`, which is
  never used for training: 20 whose oracle route is FAST_TRACK and 30 others, chosen by a seeded
  shuffle. Expected routes come from the decision policy applied to ground-truth masks
  (`label_source: oracle`).

`reviewed: true` marks cases a person checked in FiftyOne (`claimlens review launch golden`).
Reviewer corrections are recorded in each case's `notes`.
