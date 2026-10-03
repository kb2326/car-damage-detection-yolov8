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
| g119 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g120 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g122 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 1 finding(s) below confidence 0.65. |
| g124 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g129 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R6 | 5 finding(s) below confidence 0.65. |
| g130 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
| g139 | cardd_test_oracle | FAST_TRACK | ADJUSTER_REVIEW | R5 | Estimate missing or above the $3,000 limit. |
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
| Cost per claim (mean / max) | $0.009 / $0.013 |
| Total cost of this run | $0.91 |
| Agent failures (sent to a person) | none |

## Claims the agent held back (R7, R8)

| Case | Expected | Rule | Agent said | Rationale | Open questions |
|---|---|---|---|---|---|
| g004 | ADJUSTER_REVIEW | R7 | ADJUSTER_REVIEW, medium | The claimant reports a low-speed collision, but the vision models found no damage and the repair estimate is $0. The description is vague and one photo does not show damage. Even if damage exists, repair costs under the $1,000 deductible would result in no payout (BAS-3.1, BAS-2.2). An adjuster should check whether the photo is adequate (BAS-9.2). | Does the single photo show the whole vehicle and licence plate, or is there damage it missed? Can the claimant provide more photos or say where the damage is? |
| g027 | FAST_TRACK | R7 | ADJUSTER_REVIEW, medium | The claimant reports a low-speed collision, but the vision models found no damage and the estimate is $0. The description does not match the photo evidence. This may be a poor or incomplete photo, or damage that is not visible. An adjuster should decide whether to ask for more photos or an inspection. | Does the single photo show the whole vehicle and the licence plate, and is it clear enough to show damage? Should the claimant be asked for more photos or an inspection? |
| g049 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | Collision is covered (STD-2.1) and nothing is excluded. But the description is vague, and only one of two photos was accepted. The only finding is a broken trunk lamp at 0.71 confidence. The $900–$1,800 estimate looks high for a lamp alone and is a wide range, so a person should check it. | Why was the second photo rejected, and can the claimant send a replacement? Does the estimate of $900–$1,800 fit a lamp alone, or is there other damage not seen in the photo? What were the circumstances of the collision (what was hit, where, when)? |
| g103 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The finding in E1 is a flat tire located on the "trunk". A tire is not part of the trunk, and a coverage of 158% of the part is not possible. The vision output is inconsistent, so it can't be matched to a low-speed collision. A flat tire can also come from wear or a puncture rather than a collision (STD-8.3). The estimate of $200 to $600 is close to the $500 deductible. | Which part is actually damaged: the tire or the trunk? Please check the photo. Does the damage fit a low-speed collision, or does it look like a puncture or wear? Is the payout above the $500 deductible once the estimate is checked? |
| g104 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | Collision cover is active and glass is treated as a collision claim (BAS-4.1), but the description is vague. A low-speed collision does not obviously explain a glass shatter covering 96% of the part (E1). Only one photo was accepted, so there is no whole-vehicle or plate photo (BAS-9.2). The $1,200–$2,500 estimate is close to the $1,000 deductible (BAS-3.1), so the net payout may be small. | Which piece of glass is damaged (windscreen, side or rear), and does that fit a low-speed collision? Can the claimant provide whole-vehicle and licence plate photos? Was there any other damage, such as impact marks, that the single photo does not show? |
| g107 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The vision finding (E1) doesn't hold up. It reports glass shatter on a front bumper, which is not a glass part, and a damage area of 298% of the part, which is impossible. The claimant's description is generic and gives no detail to check it against. Collision cover (STD-2.1) would apply if the loss is genuine, but the damage evidence is unreliable and should be checked by a person, possibly with new photos. | Is the damage glass (headlight or windscreen) or the bumper itself? The model's label may be wrong. Why does the finding show 298% of the part? Is it a model error? Does the $1,200-$2,500 estimate fit the damage actually shown in the photo? Can the claimant give more detail on the collision and upload more photos? |
| g108 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The finding doesn't fit the description. The vision model reports a flat tire on the front bumper, but a tire is not part of a bumper, and a flat doesn't usually come from a low-speed collision. The description is also very vague. The estimate of $200 to $600 is close to the $500 deductible, so the payout may be small or zero. A flat could be puncture or wear, which PRM-8.3 excludes. | Does the photo show bumper damage from the collision, or only a flat tire? Was the flat caused by the collision, or by a puncture or wear? What did the collision involve (other vehicle or object), and where did it happen? |
| g115 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | The finding does not make sense. A flat tire is reported on the front bumper, and it covers 108% of the part, which is impossible because a tire is not part of a bumper. The description is vague ("low-speed collision") and does not explain a flat tire. Collision cover applies in principle (STD-2.1), but the damage evidence looks unreliable, so a person should check the photo. | Does the photo show a flat tire, bumper damage, or both? Is a flat tire consistent with a low-speed collision, or is it unrelated wear or a puncture? Is the $200-$600 estimate based on the right part? |
| g146 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | Collision is covered (STD-2.1), but the finding doesn't fit: a hood is normally sheet metal, so "glass_shatter on hood" (E1) is odd, and 108% of the part is not a valid proportion. The description is generic and only one photo was accepted, so the vehicle and plate photos STD-9.2 asks for may be missing. The finding may be mislabelled, for example windscreen damage or a dented hood, which would change the deductible (STD-4.2 versus the $250 collision deductible). | Does the photo actually show glass on the hood, or is the part or damage type mislabelled (windscreen vs hood)? Are whole-vehicle and licence plate photos available? Does a low-speed collision plausibly explain a $1,200-$2,500 repair? |
| g147 | FAST_TRACK | R7 | ADJUSTER_REVIEW, low | Collision is covered (STD-2.1), but the description is vague. The only finding (E1) reports 146% of the part damaged, which is not possible and suggests a measurement error. A single photo shows only shattered glass, so I can't confirm it matches a low-speed collision or the $1,200–$2,500 estimate, which is high for glass alone. The glass deductible may be $100 or the $500 collision deductible, depending on how the loss is classified. | Does the 146% damage figure reflect a vision-model error, and does the photo show other damage consistent with a collision? Was only the glass damaged, and what caused it (impact or something else)? Which deductible applies: $100 glass (STD-4.2) or $500 collision? Why is the estimate so high for a glass-only finding? |
