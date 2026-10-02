# Triage evaluation: evals/golden/v1/claims.jsonl

- Date: 2026-10-02
- Detector: `legacy-yolov8n:yolov8n-cardamage-v6.pt`
- Agent: `stub-v0`
- Decision policy: `decision-policy-v0`
- Cases: 97 (0 human-reviewed)

| Metric | Value |
|---|---|
| Route accuracy | 0.73 (71/97) |
| Escalation recall | 1.00 (67/67) |

## Confusion matrix (rows: expected, columns: predicted)

| expected \ predicted | FAST_TRACK | ADJUSTER_REVIEW | FRAUD_REVIEW | ERROR |
|---|---|---|---|---|
| FAST_TRACK | 4 | 26 | 0 | 0 |
| ADJUSTER_REVIEW | 0 | 62 | 0 | 0 |
| FRAUD_REVIEW | 0 | 0 | 5 | 0 |

## Accuracy by scenario

| Scenario | Correct | Total |
|---|---|---|
| cardd_test_oracle | 34 | 50 |
| duplicate_in_claim | 0 | 2 |
| lapsed_policy | 5 | 5 |
| no_collision_cover | 5 | 5 |
| oracle_single_photo | 19 | 27 |
| photo_reuse | 5 | 5 |
| unusable_photo | 3 | 3 |

## Mismatches

| Case | Scenario | Expected | Predicted | Rule | Reason |
|---|---|---|---|---|---|
| g012 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g013 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g018 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g019 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g026 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g027 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g028 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g030 | oracle_single_photo | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g049 | duplicate_in_claim | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g050 | duplicate_in_claim | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g103 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g105 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g107 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g108 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g112 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 3 finding(s) below confidence 0.40. |
| g114 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g115 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g119 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 2 finding(s) below confidence 0.40. |
| g120 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 2 finding(s) below confidence 0.40. |
| g122 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g124 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g130 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g139 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g142 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.40. |
| g144 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 2 finding(s) below confidence 0.40. |
| g148 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 2 finding(s) below confidence 0.40. |
