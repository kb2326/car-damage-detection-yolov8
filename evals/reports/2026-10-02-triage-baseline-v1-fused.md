# Triage evaluation: evals/golden/v1/claims.jsonl

- Date: 2026-10-02
- Detector: `fused:damage-yolo11s-v1+parts-yolo11n-v1+T0.80`
- Agent: `stub-v0`
- Decision policy: `decision-policy-v0`
- Cases: 97 (0 human-reviewed)

| Metric | Value |
|---|---|
| Route accuracy | 0.80 (78/97) |
| Escalation recall | 1.00 (67/67) |

## Confusion matrix (rows: expected, columns: predicted)

| expected \ predicted | FAST_TRACK | ADJUSTER_REVIEW | FRAUD_REVIEW | ERROR |
|---|---|---|---|---|
| FAST_TRACK | 11 | 19 | 0 | 0 |
| ADJUSTER_REVIEW | 0 | 62 | 0 | 0 |
| FRAUD_REVIEW | 0 | 0 | 5 | 0 |

## Accuracy by scenario

| Scenario | Correct | Total |
|---|---|---|
| cardd_test_oracle | 38 | 50 |
| duplicate_in_claim | 1 | 2 |
| lapsed_policy | 5 | 5 |
| no_collision_cover | 5 | 5 |
| oracle_single_photo | 21 | 27 |
| photo_reuse | 5 | 5 |
| unusable_photo | 3 | 3 |

## Mismatches

| Case | Scenario | Expected | Predicted | Rule | Reason |
|---|---|---|---|---|---|
| g013 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g018 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g019 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g026 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g027 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g030 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g050 | duplicate_in_claim | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g105 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g114 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g119 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g120 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g122 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g124 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g129 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 3 finding(s) below confidence 0.40. |
| g130 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g139 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g142 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g144 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 3 finding(s) below confidence 0.40. |
| g148 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |

## What if: other confidence thresholds (R6)

Policy unchanged; each claim is re-decided from its final state.

| Min confidence | Route accuracy | Escalation recall | Correct fast-tracks |
|---|---|---|---|
| 0.25 | 0.79 | 0.97 | 12 of 30 |
| 0.40 | 0.80 | 1.00 | 11 of 30 |
| 0.55 | 0.78 | 1.00 | 9 of 30 |
| 0.65 | 0.78 | 1.00 | 9 of 30 |
