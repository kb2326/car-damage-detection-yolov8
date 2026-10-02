# M2 retrospective: data engine

- **Dates:** 2026-10-01 to 2026-10-02
- **Plans:** [`m2a-data-pipeline`](../plans/2026-10-01-m2a-data-pipeline.md),
  [`m2b-review-autolabel`](../plans/2026-10-01-m2b-review-autolabel.md)

## What we shipped

- **M2a:** `damage-v1` (4,000 CarDD images) and `parts-v1` (3,833 images), built by DVC stages with
  a name-based taxonomy, a data contract, exact and perceptual dedupe, and splits that keep whole
  duplicate clusters together and never alter the frozen test split (ADR 0004, ADR 0005).
- **M2b:**
  - Grounding DINO + SAM 2 part proposals on 100 frozen test images, run on a Kaggle GPU
    (ADR 0006).
  - Approve/reject review stored as versioned data in git, applied by pure functions.
  - `fusion-eval-v1`: 133 approved part masks on 77 images (AI-reviewed, not human-verified).
  - Golden claims v1: 97 cases, 3 duplicate photos removed and 50 CarDD test cases added; 30 are
    expected fast-tracks, up from 11.
  - FiftyOne review UI and `claimlens review` commands.
- 192 tests (3 skipped integration tests), 98% line coverage, strict typing.

## Numbers

| Metric | v0 (50 cases) | v1 (97 cases) |
|---|---|---|
| Route accuracy | 0.78 (39/50) | 0.73 (71/97) |
| Escalation recall | 1.00 (39/39) | 1.00 (67/67) |
| Correct fast-tracks | 0 of 11 | 4 of 30 |

Accuracy fell because v1 has almost three times as many expected fast-tracks, which is exactly
where the legacy model fails. The system is still safe: no claim that needed a person was
fast-tracked. Of the 26 missed fast-tracks, most go to review because the legacy model over-prices
the damage (R5) or is unsure (R6). That is the job of the M3 models.

Auto-labelling approval rate: 133 of 455 proposals (29%). Wheels, lights, glass and mirrors work.
Doors, hoods and bumpers mostly do not.

## What went well

- Dedupe found real leaks: the same dent photo was in train and valid under two CarDD ids.
- Keeping review decisions as data means a later human review replaces the AI review with no code
  changes.
- Moving the GPU job to Kaggle took one script and one metadata file. 100 images took about a minute.

## What was harder than expected

- Roboflow v6 was CarDD re-uploaded under a different licence, so CarDD's non-commercial terms apply.
- The laptop overheated on the CPU labelling run; heavy model work now runs on Kaggle or Colab.
- Blank golden "unusable photo" assets share one perceptual hash, so golden dedupe had to be
  limited to oracle cases.
- Reviewing 455 polygons by hand was too slow for the owner, so the review was delegated to an AI
  reviewer. That trades trust for speed and is recorded as such.
- Syncing a different dependency group removed one of two OpenCV packages that share the `cv2`
  folder and broke it; reinstalling `opencv-python-headless` fixed it.

## What we will change in M3

- Train our own part model on `parts-v1` rather than relying on open-vocabulary prompts for panels.
- Train a damage segmentation model and measure severity on masks, aiming to fix the R5 and R6
  misses.
- Get a human spot-check of `fusion-eval-v1` and a human review of the golden claims before
  reporting final numbers.
