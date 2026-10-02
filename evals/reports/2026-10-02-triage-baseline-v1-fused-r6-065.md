# Triage evaluation: evals/golden/v1/claims.jsonl

- Date: 2026-10-02
- Detector: `fused:damage-yolo11s-v1+parts-yolo11n-v1+T0.80`
- Agent: `stub-v0`
- Decision policy: `decision-policy-v1`
- Cases: 97 (0 human-reviewed)

| Metric | Value |
|---|---|
| Route accuracy | 0.78 (76/97) |
| Escalation recall | 1.00 (67/67) |

## Confusion matrix (rows: expected, columns: predicted)

| expected \ predicted | FAST_TRACK | ADJUSTER_REVIEW | FRAUD_REVIEW | ERROR |
|---|---|---|---|---|
| FAST_TRACK | 9 | 21 | 0 | 0 |
| ADJUSTER_REVIEW | 0 | 62 | 0 | 0 |
| FRAUD_REVIEW | 0 | 0 | 5 | 0 |

## Accuracy by scenario

| Scenario | Correct | Total |
|---|---|---|
| cardd_test_oracle | 37 | 50 |
| duplicate_in_claim | 1 | 2 |
| lapsed_policy | 5 | 5 |
| no_collision_cover | 5 | 5 |
| oracle_single_photo | 20 | 27 |
| photo_reuse | 5 | 5 |
| unusable_photo | 3 | 3 |

## Mismatches

| Case | Scenario | Expected | Predicted | Rule | Reason |
|---|---|---|---|---|---|
| g012 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g013 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g018 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g019 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g026 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g027 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g030 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g050 | duplicate_in_claim | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g105 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g112 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g114 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g119 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g120 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g122 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g124 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g129 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 5 finding(s) below confidence 0.65. |
| g130 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g139 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g142 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g144 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 3 finding(s) below confidence 0.65. |
| g148 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
