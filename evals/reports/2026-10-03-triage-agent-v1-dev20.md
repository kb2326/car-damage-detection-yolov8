# Triage evaluation: evals/golden/v1/claims.jsonl

- Date: 2026-10-03
- Detector: `fused:damage-yolo11s-v1+parts-yolo11n-v1+T0.80`
- Agent: `triage-agent-v1+triage/v1`
- Decision policy: `decision-policy-v1`
- Cases: 20 (0 human-reviewed)

| Metric | Value |
|---|---|
| Route accuracy | 0.70 (14/20) |
| Escalation recall | 1.00 (14/14) |

## Confusion matrix (rows: expected, columns: predicted)

| expected \ predicted | FAST_TRACK | ADJUSTER_REVIEW | FRAUD_REVIEW | ERROR |
|---|---|---|---|---|
| FAST_TRACK | 0 | 6 | 0 | 0 |
| ADJUSTER_REVIEW | 0 | 12 | 0 | 0 |
| FRAUD_REVIEW | 0 | 0 | 2 | 0 |

## Accuracy by scenario

| Scenario | Correct | Total |
|---|---|---|
| cardd_test_oracle | 5 | 10 |
| duplicate_in_claim | 0 | 1 |
| lapsed_policy | 2 | 2 |
| no_collision_cover | 1 | 1 |
| oracle_single_photo | 3 | 3 |
| photo_reuse | 2 | 2 |
| unusable_photo | 1 | 1 |

## Mismatches

| Case | Scenario | Expected | Predicted | Rule | Reason |
|---|---|---|---|---|---|
| g049 | duplicate_in_claim | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g103 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g112 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g120 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g130 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g144 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 3 finding(s) below confidence 0.65. |

## Agent

| Measure | Value |
|---|---|
| Model calls per claim (mean) | 1.9 |
| Tool calls per claim (mean) | 0.9 |
| Cost per claim (mean / max) | $0.010 / $0.012 |
| Total cost of this run | $0.20 |
| Agent failures (sent to a person) | none |
