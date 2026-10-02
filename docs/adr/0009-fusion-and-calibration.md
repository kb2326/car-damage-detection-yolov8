# ADR 0009: Damage-to-part fusion, part-ratio severity and calibration

- **Status:** Accepted
- **Date:** 2026-10-02

## Context

- After M3a, severity came from how much of the *photo* the damage box covers. A close-up of a
  small dent looked severe.
- The R6 confidence threshold (0.40) was a guess made for the legacy model.
- M3b adds a part model (`parts-yolo11n-v1`, test mask mAP50 0.746) and asks two questions: which
  part each damage is on, and whether the confidence numbers are honest.

## Decision

- **Fusion by intersection over damage area.**
  - Damage and part outlines are rasterised on a 256 × 256 grid. That is about 0.4% of the image
    side, enough for assignment.
  - Each damage goes to the part covering the largest share of it, with at least 10% coverage. A
    tie goes to the smaller part.
  - Parts map to 9 coarse groups.
  - `FusedDetector` implements the existing `Detector` protocol, so the workflow and the event log
    do not change.
- **Part-ratio severity.**
  - The new rate card version `rate-card-2026.10-v1` adds bands on damage mask area ÷ part mask
    area: minor below 0.10, moderate below 0.30.
  - When no part is found, the old bands apply to the damage **box** as a share of the photo,
    which is what the rate card defines.
  - The mask-share-of-photo size was tried twice (M3a, and the first fused run) and both times
    fast-tracked golden claims that needed a person. It is not used.
- **One-temperature calibration on validation.**
  - Predictions are matched to ground truth (same class, mask IoU ≥ 0.5, greedy by confidence).
  - A single temperature is fitted by minimum negative log-likelihood on a fixed grid.
  - On the damage champion: T = 0.80, ECE from 0.059 to 0.040.
- **Threshold recommended, not applied.**
  - The recommended threshold is the lowest one on a 0.05 grid where at least 80% of the kept
    calibrated predictions are correct: 0.65.
  - The owner decided that calibration only recommends. `decision_policy.toml` stays at 0.40, and a
    change needs its own reviewed PR.
- **ONNX as an export, not the default.** Both models are exported, but on this CPU ONNX was slower
  (674 vs 357 ms and 460 vs 193 ms), so the pipeline keeps `.pt`.

## Consequences

- **Gate:** golden v1 with the fused detector gives route accuracy 0.80, escalation recall 1.00,
  and 11 of 30 correct fast-tracks. That is the same as damage-only, with severity now relative to
  the part for 88% of findings.
- **Weak spots:** the part model agrees with only 46% of reviewed parts on CarDD close-ups. The
  box fallback keeps those cases safe, but part-relative pricing is only as good as the part model
  on close-ups.
- **Next for the owner:** the recommended 0.65 would cost 2 golden fast-tracks (9 instead of 11)
  for more trustworthy findings. That decision stays open.
