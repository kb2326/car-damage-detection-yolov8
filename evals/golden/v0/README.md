# Golden claims v0

50 end-to-end triage cases built by `scripts/build_golden_v0.py` from the Roboflow car-damage v6
test split (CC BY 4.0). Image paths point into `data/raw/`, which is not in git, so running this
set needs the dataset locally.

| Label source | Meaning |
|---|---|
| `oracle` | Decision policy applied to the ground-truth annotations with a perfect agent: the route a correct pipeline should produce |
| `scenario` | Route fixed by the scenario's design (lapsed policy, missing cover, reused photo, unusable photo) |

`reviewed: true` marks cases a human has checked by looking at the photo and the expected route.

| Scenario | Cases | Expected routes |
|---|---|---|
| `oracle_single_photo` | 30 | 9 FAST_TRACK, 21 ADJUSTER_REVIEW |
| `lapsed_policy` | 5 | ADJUSTER_REVIEW |
| `no_collision_cover` | 5 | ADJUSTER_REVIEW |
| `photo_reuse` | 5 | FRAUD_REVIEW |
| `unusable_photo` | 3 | ADJUSTER_REVIEW |
| `duplicate_in_claim` | 2 | FAST_TRACK |

Most oracle cases go to an adjuster because many test photos show heavily crashed cars, whose
damage boxes cover a large share of the image and price above the $3,000 fast-track limit.

Known limitations: one photo per claim for most cases; severity uses the box area of each
annotation, not the true damaged area; the rate card is fictional.
