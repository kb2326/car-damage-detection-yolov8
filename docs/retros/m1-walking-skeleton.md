# M1 retrospective: walking skeleton

- **Dates:** 2026-10-01
- **Plan:** [`docs/plans/2026-10-01-m1-walking-skeleton.md`](../plans/2026-10-01-m1-walking-skeleton.md)

## What we shipped

- An append-only, hash-chained SQLite event log; claim state rebuilt by folding events (ADR 0003).
- A resumable workflow: quality gate, perception with the legacy YOLOv8n model (labels now read
  from the model file), photo-reuse check, rate-card pricing, coverage lookup, a stub agent, and
  the R1-R9 decision policy. No route can deny a claim.
- `claimlens run | show | verify | eval-triage`.
- 50 golden claims (32 oracle-labelled, 18 scenario-labelled) and a Markdown evaluation report.
- 102 tests, 98% line coverage, strict typing, CI with no data or API keys.

## Baseline numbers

From [`evals/reports/2026-10-01-triage-baseline-v0.md`](../../evals/reports/2026-10-01-triage-baseline-v0.md):

| Metric | Value |
|---|---|
| Route accuracy | 0.78 (39/50) |
| Escalation recall | 1.00 (39/39) |
| Correct fast-tracks | 0 of 11 |

The pipeline is safe but not yet useful: it never fast-tracks a claim that needed a human, and it
never fast-tracks one that did not. All 11 misses are claims that should have been fast-tracked:

| Rule that sent it to review | Cases | Cause |
|---|---|---|
| R5 cost above limit | 6 | The model over-sizes damage or labels it as `smash`, so prices jump |
| R6 low confidence | 4 | The model is unsure (confidence < 0.40) |
| R7 nothing detected | 1 | The model missed the damage entirely |

Every scenario that does not depend on the model (lapsed policy, no cover, reused photo, unusable
photo) is 100% correct. The remaining errors are all perception.

## What went well

- Writing the plan with full code made execution mostly transcription plus testing.
- The event log made idempotency and crash-resume easy to test.
- Fixing the label bug at the boundary (`normalize_class_name` raises on unknown names) turns the
  original silent mislabelling into a loud error.

## What was harder than expected

- The newest ruff also formats Python code blocks inside Markdown, which changed the plan on commit.
- A shell heredoc tripped over quoting; writing files with the editor tool was more reliable.
- Many golden photos show heavily crashed cars, so most oracle cases correctly go to an adjuster.
  The set needs more minor-damage cases to measure fast-tracking well.

## What we will change in M2

- Get more and better data (CarDD, deduplicated, grouped splits) before training anything.
- Add minor-damage cases to the golden set so fast-track accuracy has more than 11 cases.
- Measure severity on masks rather than boxes once segmentation arrives in M3.
