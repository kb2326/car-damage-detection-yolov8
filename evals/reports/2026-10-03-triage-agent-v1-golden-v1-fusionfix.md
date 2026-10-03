# Triage evaluation: evals/golden/v1/claims.jsonl

- Date: 2026-10-03
- Detector: `fused:damage-yolo11s-v1+parts-yolo11n-v1+T0.80`
- Agent: `triage-agent-v1+triage/v1`
- Decision policy: `decision-policy-v1`
- Cases: 97 (0 human-reviewed)

| Metric | Value |
|---|---|
| Route accuracy | 0.70 (68/97) |
| Escalation recall | 1.00 (67/67) |

## Confusion matrix (rows: expected, columns: predicted)

| expected \ predicted | FAST_TRACK | ADJUSTER_REVIEW | FRAUD_REVIEW | ERROR |
|---|---|---|---|---|
| FAST_TRACK | 1 | 29 | 0 | 0 |
| ADJUSTER_REVIEW | 0 | 62 | 0 | 0 |
| FRAUD_REVIEW | 0 | 0 | 5 | 0 |

## Accuracy by scenario

| Scenario | Correct | Total |
|---|---|---|
| cardd_test_oracle | 30 | 50 |
| duplicate_in_claim | 0 | 2 |
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
| g049 | duplicate_in_claim | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g050 | duplicate_in_claim | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g103 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g104 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g105 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g107 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g108 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g112 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g114 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g115 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g119 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 2 finding(s) below confidence 0.65. |
| g120 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g122 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g124 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g129 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 5 finding(s) below confidence 0.65. |
| g130 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g139 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 5 finding(s) below confidence 0.65. |
| g142 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g144 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 3 finding(s) below confidence 0.65. |
| g146 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g147 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R7 | The triage agent is unsure or has open questions. |
| g148 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |

## Agent

| Measure | Value |
|---|---|
| Model calls per claim (mean) | 2.0 |
| Tool calls per claim (mean) | 0.7 |
| Cost per claim (mean / max) | $0.009 / $0.017 |
| Total cost of this run | $0.85 |
| Agent failures (sent to a person) | none |
| Decided by rule | R1: 5, R3: 3, R4: 10, R5: 49, R6: 19, R7: 10, R9: 1 |

## Claims the agent held back (R7, R8)

| Case | Expected | Rule | Agent said | Rationale | Open questions |
|---|---|---|---|---|---|
| g004 | ADJUSTER_REVIEW | R7 | ADJUSTER_REVIEW, low | The claimant reports a low-speed collision, but the vision models found no damage and estimate $0. The single photo may not show the damaged area, or there may be no damage. Even if damage exists, a repair under $1,000 would fall below the deductible. A person should check the photo before any decision. | Does the photo show the whole vehicle and the area of impact, as BAS-9.2 asks? Can the claimant send more photos or say where the damage is? Is there any damage at all, or is the repair cost below the $1,000 deductible? |
| g027 | FAST_TRACK | R7 | ADJUSTER_REVIEW, medium | The claimant reports a low-speed collision, but the vision models found no damage in the single photo and the estimate is $0. The story cannot be confirmed from the evidence. The photo may be unclear or may not show the damaged area, so an adjuster should ask for more photos or an inspection. | Does the single photo show the whole vehicle and the damaged area? Is there any real damage that needs further photos or an inspection? Was a police report filed or anyone injured? |
| g049 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The claim is covered, but the evidence is thin. One photo was accepted and one was rejected, and the only finding is a broken lamp (0.71 confidence) on an unidentified part. The claimant's description is generic. The $900–$1,800 estimate looks high for a lamp alone, so a person should check it. | Why was the second photo rejected, and can a clearer one be obtained? Which part is the broken lamp on, and does the damage fit a low-speed collision? Does the wide $900–$1,800 estimate reflect more damage than the lamp alone? |
| g103 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The only finding is a flat tire (E1), and the description is generic ("low-speed collision") with no impact point or other object named. A flat alone does not clearly show collision damage and could be a puncture or wear (STD-8.3), so the story is not confirmed by the damage. The $200–600 estimate straddles the $500 deductible (STD-3.1), so the payout may be small or nil. | Is there any body, wheel or suspension damage consistent with a collision, beyond the flat tire? What was hit, and where on the vehicle? Could the flat be a puncture or wear instead? Only one photo was submitted. Should more be requested? |
| g104 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | Collision cover is active and glass damage is handled as a collision claim with the full deductible (BAS-4.1). The description is vague, though. The only finding is a shattered glass part (E1), and one photo does not show whether other collision damage exists. A low-speed collision that breaks glass about 92% needs checking. The payout after the $1,000 deductible would be small, roughly $200 to $1,500. | Does the damage match a low-speed collision, or could it be non-collision glass breakage such as vandalism or a stone strike? Can the claimant supply whole-vehicle and licence plate photos, as BAS-9.2 asks? Which glass part is damaged, and is there any other collision damage? |
| g107 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The description is vague ("low-speed collision") and the only finding is glass shatter on an unknown part, from one photo. It is unclear whether this is a collision or a glass-only/vandalism loss. That matters because the glass deductible ($100, STD-4.2) differs from the collision deductible ($500, STD-2.1). No integrity signals were raised. | Which glass part is damaged and does it fit a low-speed collision? Which deductible applies: glass ($100) or collision ($500)? Is a $1,200–$2,500 estimate plausible for glass alone, and are more photos needed? |
| g108 | FAST_TRACK | R7 | ADJUSTER_REVIEW, medium | The only finding is a flat tire (E1), and the claimant's description is generic with no details of what was hit. A flat alone does not clearly show collision damage and could be a puncture or wear (PRM-8.3), which is not covered. The $200–$600 estimate is close to the $500 deductible, so the payout may be small or zero. | What did the vehicle collide with, and is there any body or wheel-rim damage consistent with an impact? Is the flat a puncture or worn tire rather than collision damage? Is a claim worthwhile given the estimate of $200–$600 against the $500 deductible? |
| g115 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The only damage found is a flat tire (E1). A flat on its own does not clearly show a collision, because it could be a puncture or wear and tear (STD-8.3). The description is generic and names no impact point or object. The estimate of $200 to $600 is close to the $250 deductible, so a collision payout may be small or nil (STD-2.1). | What did the vehicle hit, and where was the impact? Is there any other damage, such as rim, suspension or body damage, that matches a collision? Was the tire worn or punctured before the incident? |
| g146 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The description is vague ("low-speed collision") and the only finding is glass shatter on an unknown part, from one photo. A low-speed collision does not clearly explain shattered glass, and a $1,200–$2,500 estimate is high for glass alone. The glass deductible ($100, STD-4.2) may differ from the collision deductible ($250, STD-2.1), so an adjuster should confirm which applies. | Which glass part is damaged, and was it broken by the collision or by something else, such as vandalism or a break-in? Does the estimate cover more than glass, given that only one photo was provided? Should the $100 glass deductible or the $250 collision deductible apply? |
| g147 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The only damage found is a fully shattered glass part (E1), and the claimant says only "low-speed collision" without naming the glass or the impact. A low-speed impact rarely shatters glass completely, so the story may not match the damage. The $1,200–$2,500 estimate looks high for glass alone, and it is unclear whether the $100 glass deductible (STD-4.2) or the $500 collision deductible applies. | Which glass is damaged, and how did a low-speed collision shatter it completely? Is there other collision damage that the single photo does not show? Was this a collision or vandalism, and which deductible applies? Why is the estimate so high for glass replacement alone? |
