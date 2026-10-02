# ADR 0006: Verified auto-labelling for car-part masks

- **Status:** Accepted
- **Date:** 2026-10-02

## Context

The damage images (`damage-v1`) have damage masks but no part masks, so we cannot yet say *which
part* a dent is on. Labelling parts by hand on thousands of photos is too slow for this project.
Open-vocabulary foundation models can propose part masks, but their output is not trustworthy on
its own.

## Decision

- **Propose with foundation models:** Grounding DINO tiny finds part boxes from text prompts, and
  SAM 2.1 tiny turns each box into a mask (`claimlens.autolabel`, behind the `PartLabeller`
  protocol so the models can be swapped).
- **Map phrases exactly:** a detection is kept only if its phrase is exactly one prompt; merged
  phrases such as "door hood" are dropped as `ambiguous_phrase`. Parts are coarse groups
  (`part_groups` in `config/taxonomy.toml`), at most two per group per image.
- **Verify before use:** every proposal is approved or rejected in a review step. Decisions are
  stored as versioned JSON in git (`reviews/<job>.json`), applied by pure functions, and an image is
  used only when every proposal on it was decided.
- **Run on a free GPU:** the job runs as a private Kaggle kernel on a T4 (`training/kaggle/autolabel`).
  A laptop CPU took about 20 s per image and overheated; the GPU took about 1 minute for 100 images.
  Results come back with `kaggle kernels output` and are recorded with `dvc commit`. The kernel
  checks out a pinned commit (`fusion-eval-v1` was produced by `0396bfd`), and the DVC stage is
  frozen so `dvc repro` never reruns it on a laptop.
- **Tie decisions to proposals:** keys are positional (`image_id#index`), so a review stores the
  SHA-256 of the proposals file and `review apply` refuses a mismatch. A later review pass is
  pre-loaded with the existing decisions and merged on export, so a spot-check never erases them.

## Results (`fusion-eval-v1`, 100 frozen CarDD test images)

| Part group | Approved / proposed |
|---|---|
| wheel | 40 / 84 |
| light | 35 / 89 |
| glass | 25 / 75 |
| mirror | 12 / 44 |
| hood | 11 / 48 |
| door | 10 / 106 |
| front_bumper | 0 / 5 |
| trunk | 0 / 4 |
| rear_bumper | 0 / 0 |
| **Total** | **133 / 455 (29%)** |

A further 162 proposals were dropped as merged phrases and 54 as duplicates before review.
77 images are usable; 23 had every proposal rejected.

Observed failure modes:

- **"door" is a catch-all.** The model labels most large panels (fenders, quarter panels, bumpers)
  and often two doors at once as one "door".
- **Bumpers are almost never found**, even though most CarDD damage is on bumpers. Close-up photos
  rarely show a whole bumper, and the prompt "bumper" competes with "door" and "hood".
- **Wrong vehicle:** parts of cars in the background are proposed.
- **Small, distinct parts work best:** wheels, headlights, tail lights, mirrors and windscreens.

## Reviewer

The review of `fusion-eval-v1` was done by Claude (an AI assistant), not by a person, at the
project owner's request; the reviewer field in `reviews/fusion-eval-v1.json` says so. The set is
therefore **AI-reviewed, not human-verified**. A human spot-check is still recommended before the
set is used to report results.

## Consequences

- Off-the-shelf open-vocabulary models are good enough to bootstrap small parts, not panels.
  Bumpers, doors and fenders need a trained part model (`parts-v1`, M3) or human labelling.
- The approval rate per part group is itself a useful metric for later labelling jobs.
- Because decisions are data, a later human review can replace the AI review without code changes.
