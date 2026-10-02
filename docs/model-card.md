# Model card: ClaimLens damage and part models (v1)

Following *Model Cards for Model Reporting* (Mitchell et al., 2019). Every number comes from
`evals/reports/2026-10-02-*.md` and `reports/models/*.json`.

## Model details

| | Damage model | Part model |
|---|---|---|
| Name (MLflow) | `claimlens-damage` v2, run `damage-yolo11s-v1` | `claimlens-parts` v1, run `parts-yolo11n-v1` |
| Architecture | YOLO11s-seg (instance segmentation) | YOLO11n-seg |
| Classes | 6: crack, dent, glass_shatter, lamp_broken, scratch, tire_flat | 22 car parts, mapped to 9 groups for pricing |
| Training | 92 epochs (best 72), 640 px, seed 20261002, Kaggle T4 | 63 epochs (best 43), same settings |
| Code / data | commit `378aca2`, `damage-v1` md5 `cc56ac51…` | commit `7b97f54`, `parts-v1` md5 `9764aa90…` |
| Calibration | temperature T = 0.80 | none |
| Owner, date | ClaimLens project, 2026-10-02 | same |

The champion of each task was chosen on **validation** mask mAP50. The damage `n` model
(`damage-yolo11n-v1`, val 0.719, test 0.735) lost on validation to `s` (val 0.738).

## Intended use

- **Intended:** measuring damage on car photos inside ClaimLens, a non-commercial learning and
  portfolio project. The models measure; fixed rules decide the route; people decide outcomes.
- **Out of scope:**
  - real insurance decisions or any commercial use;
  - denying a claim (the system cannot deny);
  - photos of other vehicle types;
  - redistributing the weights.

## Training data

- **Damage:** `damage-v1`, the 4,000 CarDD images (Wang et al., 2023). Deduplicated, with the
  frozen official test split. See [`data-card.md`](data-card.md).
- **Parts:** `parts-v1`, 3,833 images from Roboflow `car-seg` (CC BY 4.0).

## Metrics (test split)

**Damage model, test mask mAP50 0.733** (target ≥ 0.50). Mask mAP50-95 0.549; box mAP50 0.743.

| Class | Mask mAP50 |
|---|---|
| glass_shatter | 0.971 |
| tire_flat | 0.892 |
| lamp_broken | 0.883 |
| dent | 0.643 |
| scratch | 0.566 |
| crack | 0.441 |

**Part model, test mask mAP50 0.746** (22 classes). Mask mAP50-95 0.600.
- **Strongest:** front_bumper 0.982, front_glass 0.977, hood 0.967, back_bumper 0.944.
- **Weakest:** the left/right lights and doors, for example back_left_light 0.444 and
  front_right_light 0.479.
- Per-class numbers are in `evals/reports/2026-10-02-parts-model-v1.md`.

## Calibration (damage model, validation split)

| | Value |
|---|---|
| Predictions matched to truth | 3,083 (1,258 correct) on 814 photos |
| Temperature | 0.80, so the raw scores were slightly under-confident |
| ECE before / after | 0.059 / 0.040 |
| Recommended R6 threshold | 0.65 (82% of kept predictions correct) — **not applied** |

On golden v1, the recommended 0.65 keeps escalation recall at 1.00 but gives 9 instead of 11
correct fast-tracks. The owner decides whether to adopt it, in a separate change.

## System results (golden v1, 97 claims)

| Detector | Route accuracy | Escalation recall | Correct fast-tracks |
|---|---|---|---|
| Legacy course model | 0.73 | 1.00 | 4 of 30 |
| Damage model only | 0.80 | 1.00 | 11 of 30 |
| Damage + parts + fusion + calibration | 0.80 | 1.00 | 11 of 30 |

With fusion:
- **88%** of findings on golden photos get a part (351 of 400).
- Severity changes for 112 of them: 77 move up and 35 move down. A dent is now judged against the
  size of the part it sits on.

## Performance on a laptop CPU (median ms per photo, 20 test photos)

| Model | `.pt` | `.onnx` |
|---|---|---|
| Damage (s) | 357 | 674 |
| Parts (n) | 193 | 460 |

ONNX is slower on this machine, so the pipeline uses `.pt`. The ONNX files exist for portability.

## Limitations

- **Stock photos, not claim photos.** CarDD comes from Flickr and Shutterstock; real driver photos
  will look different.
- **Close-ups confuse the part model.** It agrees with only 61 of 133 reviewed parts (46%) on
  CarDD close-ups. Lights (9 of 35) and doors (4 of 10) are weakest.
- **Left/right confusion.** Pricing ignores side, so this matters less than the scores suggest.
- **No "smash" class.** CarDD has no class for a crushed panel. Golden case g014 (labelled smash)
  is seen as two dents.
- **Crack and scratch are the weakest damage classes** (0.441, 0.566).
- **Sizing:** price bands use the damage **box** as a share of the photo, or the damage **mask** as
  a share of its part. Sizing by the mask share of the photo was tried and broke safety (golden
  g121, g014), so it is not used.

## Ethical considerations

- A model finding never denies a claim. Uncertain, expensive or fraud-flagged claims go to a
  person, and the gate for every model change is escalation recall = 1.00 on the golden set.
- Every decision records the detector version (for example
  `fused:damage-yolo11s-v1+parts-yolo11n-v1+T0.80`) in the hash-chained claim log.

## Licence

**Private.** The weights are trained on CarDD, which allows non-commercial research and education
use without redistribution. They are not released on Hugging Face, Kaggle or git; they live only in
the local DVC remote, the local MLflow logbook and private Kaggle outputs (ADR 0004, ADR 0008).
