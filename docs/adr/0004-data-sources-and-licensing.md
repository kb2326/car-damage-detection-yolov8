# ADR 0004: Data sources and licensing

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

The damage model needs thousands of segmentation-labelled images. The candidate Roboflow dataset
(`auto-industry/car-damage-detection-vyhvw` v6) is labelled CC BY 4.0, but its 4,000 images, its
2,816 / 810 / 374 split, its six classes and the six-digit ids in its file names match CarDD
exactly. CarDD itself is distributed through a licence form for non-commercial research and
education only. A re-uploader cannot grant rights they do not hold.

## Decision

- Use the Roboflow copy for convenience, but **apply CarDD's terms**: non-commercial research and
  education only; do not redistribute images; label any published weights as trained on CarDD
  for non-commercial use.
- Submit the official CarDD licence form, and reconcile the official copy with the Roboflow copy
  (deduplicate and record the result) once access is granted.
- Use the 339-image `legacy-course-subset` only for golden claims, never for training.
- Use Roboflow `car-seg` (CC BY 4.0, via Ultralytics) for the part model.
- Record that the Ultralytics library is AGPL-3.0. Whether that is acceptable for the public demo
  is decided in an ADR in M3.

## Consequences

- No images are committed or published; only pointers, code and statistics.
- The data card states the terms for each source.
- A future commercial use would need new data or a different licence.
