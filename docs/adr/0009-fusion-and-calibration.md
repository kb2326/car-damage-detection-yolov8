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

## Update (2026-10-02): threshold adopted

The owner approved the recommended threshold. `config/decision_policy.toml` is now
`decision-policy-v1` with `min_finding_confidence = 0.65`. Golden v1 with the fused detector:
route accuracy 0.78, escalation recall 1.00, 9 of 30 correct fast-tracks
(`evals/reports/2026-10-02-triage-baseline-v1-fused-r6-065.md`). This trades two fast-tracks for
findings that are correct at least 80% of the time on validation.

## Update (2026-10-03): plausible parts and a capped part ratio

The M5a triage agent found impossible fusion output on golden v1: a flat tyre "on the front
bumper", glass "on the hood", and damage covering 108% to 298% of a part. Two causes, both fixed:

- **The ratio** was damage area / part area, even when only part of the damage overlapped the
  part. It is now **overlap / part area**, the share of the part the damage covers, never above 1.
- **Any damage type could go to any part.** `config/taxonomy.toml` `[part_groups.damage_parts]`
  now limits three types: a flat tyre only to a wheel, a broken lamp only to a light, glass shatter
  to glass, a light or a mirror. Dents, scratches, cracks and smashes can be on any part. With no
  plausible part the finding has no part, and pricing uses the box share as before.

Golden v1 with the fused detector and the stub agent is unchanged: route accuracy 0.78,
escalation recall 1.00, 9 of 30 fast-tracks
(`evals/reports/2026-10-03-triage-baseline-v1-fused-fusionfix.md`). Two claims move from R5 to R6
because their estimates fell. With the LLM agent: 0.70, 1.00, 1 of 30, $0.85
(`evals/reports/2026-10-03-triage-agent-v1-golden-v1-fusionfix.md`). The agent no longer reports
impossible output, but it still holds back the same claims for vague stories, a flat tyre that
could be a puncture, or glass that a low-speed knock rarely shatters. That is the agent's
calibration, which M5b measures.
