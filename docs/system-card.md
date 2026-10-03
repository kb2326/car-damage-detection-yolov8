# ClaimLens system card

- **Version:** M7, 3 October 2026
- **Owner:** kb2326 (non-commercial learning and portfolio project)
- **Related documents:** [model card](model-card.md) · [data card](data-card.md) ·
  [design](specs/2026-10-01-claimlens-design.md) · ADRs in [`adr/`](adr/)

## 1. What it is for

ClaimLens triages car-damage insurance claims. It takes a claim (a chat or a form, plus photos),
measures the damage, checks cover and the claim's integrity, and routes the claim to one of three
places: **fast-track**, **adjuster review** or **fraud review**. It explains why, with the rule
that fired, the evidence and cited policy clauses.

**It is not for:**
- **denying claims.** No model, agent or rule can deny; only a person can, in the review step;
- **paying claims.** A payment needs a single-use token signed by a person;
- **real customers or real policies.** All policies, insurers and claims are fictional;
- **commercial use.** The damage model is trained on CarDD, which allows non-commercial research only.

## 2. How a claim flows

```
customer ─▶ intake chat (Claude Haiku) ─▶ claim log (append-only, hash-chained)
          ─▶ photo quality gate ─▶ damage model + part model (YOLO11, own training) ─▶ fusion
          ─▶ integrity (exact and near-copy photo reuse) ─▶ pricing (rate card) ─▶ coverage
          ─▶ triage agent (Claude Sonnet, read-only tools, approved procedures, claim memory)
          ─▶ rules R1-R9 decide the route ─▶ a person reviews (approve, ask, change, deny)
```

**Models measure, the agent reasons, rules decide.** The agent's recommendation is one input to
the rules. A fraud signal, a failure, no usable photo, missing cover, a large estimate or a
low-confidence finding (rules R1 to R6) all send the claim to a person **whatever the agent says**.
One thing only the agent checks: an exclusion that appears only in the customer's story (business
use, racing, an unlicensed driver). No rule reads the story, so for those claims the agent's own
judgement is the control (rules R7 and R8 act on its doubt); see section 6.

## 3. Components

| Component | What it is | Key number |
|---|---|---|
| Damage model | YOLO11s-seg, 6 damage types | test mask mAP50 0.733 |
| Part model | YOLO11n-seg, 22 car parts | test mask mAP50 0.746 |
| Calibration | Temperature 0.80; rule R6 needs confidence 0.65 | ECE 0.059 → 0.040 |
| Triage agent | LangGraph agent, Claude Sonnet 5.5 via our gateway; read-only MCP tools; citations checked in code | $0.011 per claim |
| Intake agent | LangGraph chat, Claude Haiku 4.5; pauses and resumes; never promises, prices or denies | 9 to 17 turns |
| Policy search | Hybrid search over 56 fictional clauses (LanceDB) | recall@5 1.00 (small corpus) |
| Claim memory | Earlier claims: route, review, damage, photo fingerprints; written only by the workflow | near-copy threshold 10 bits |
| Rules | R1 fraud signal ≥ 0.5 … R9 fast-track; versioned config | escalation recall 1.00 |
| Web prototype | FastAPI app: chat, claims list, claim page, review | local only |

## 4. Evaluation

| What | How | Result |
|---|---|---|
| Routing safety | Golden v2: 150 claims with known right routes, incl. 53 story cases | **Escalation recall 1.00**; route accuracy 0.77 |
| Story cases | Stories that change the right route (business use, racing, unlicensed drivers…) | 43 of 43 sent to a person with the right clause |
| Harmless stories | Stories that should be fast-tracked | 5 of 10 (the agent is cautious) |
| Citations | Every clause the agent cites must exist in the customer's own wording | validity 1.00 |
| Intake | 10 simulated customers × 4 tries (vague, over-sharing, story-changing, bad photos) | pass^4 0.80 (target 0.70) |
| Red team, hijacked | 34 attacks over the OWASP agentic top 10; the fake model obeys the attacker | **33 of 33 run held**, plus a dependency audit in CI |
| Red team, live | 15 risky stories with prompt injections, real Sonnet agent | 15 of 15 sent to a person; the agent was fooled 0 times |
| CI gate | Every PR: tests, types, lint, dependency audit, eval gate on a committed scorecard | safety metrics must stay 1.00 |

Reports: `evals/reports/` (golden v2, simulated customers, red team, live red team).

## 5. Safeguards

- **Rules decide**, after the models and the agent; no route can deny.
- **Least privilege:** each agent gets a named list of read-only tools; payments are outside
  every agent's reach and need a person's signed, single-use token.
- **Data, not instructions:** customer text, tool results and memory reach the models inside
  marked data blocks; the intake agent learns only whether a policy exists.
- **Limits:** per-claim and per-day spending caps, step, time and token limits; any failure goes to
  a person (rule R2).
- **Audit:** every step is an event on a hash-chained log (`claimlens verify`); model, prompt and
  agent versions are recorded.
- **Memory governance:** written only by the workflow, with provenance; no customer words; can be
  forgotten (`claimlens memory forget`), which also removes old copies.
- **Fraud:** exact and near-copy photo reuse across claims raise a fraud signal (R1).
- **Human review:** the claim page shows the evidence, confidence, the agent's reasons, similar
  claims and the rule; reviews are logged; a reviewer's route change counts everywhere.
- **Web prototype:** local only, Host and Origin checks, escaped output, checked uploads.

## 6. Limitations and open risks

- **Exclusions in the story rest on the agent.** For business use, racing or an unlicensed driver
  mentioned only in the customer's words, no rule is a backstop: a fully fooled agent could let
  such a claim through to R9. The live red team measured this (15 of 15 injected stories sent to a
  person, the agent fooled 0 times), but it is measured, not guaranteed. A code check of the story
  (keyword or classifier, reviewed by people) is the planned control.
- **Data:** CarDD and course photos only; no real claim photos, no night or rain variety.
- **Weak findings on undamaged cars:** the damage model reports low-confidence damage
  (0.08–0.39) on clean cars, which inflates estimates; R6 keeps such claims from a fast-track.
- **Cautious agent:** half of harmless stories still go to a person (cost, not risk).
- **Test cases written by Claude:** golden stories and simulated customers were not all checked by
  a person; the LLM judge is not validated against human labels.
- **Near-copies:** crops heavier than about 2% are not caught; no AI-generated-image or photo
  metadata checks.
- **Corrected answers:** a customer who corrects a date may keep the first one (intake prompt v2).
- **Single provider:** Anthropic only; no fallback provider.
- **Not covered by the red team:** ASI07 (no agent-to-agent channel exists).

## 7. Privacy and licences

- Secrets live only in a git-ignored `.env`. The event log stores facts and a fingerprint of the
  chat transcript, not the transcript.
- CarDD images and the trained weights are never redistributed. Published images use only
  CC BY 4.0 `car-seg` photos, credited.
- The public demo (M8b) is a read-only showcase: pre-filed sample claims, no visitor input, no
  LLM calls.

## 8. Regulation note

Motor claims triage is not listed in Annex III of the EU AI Act (life and health insurance pricing
and risk assessment are), so ClaimLens would not be a high-risk system there. It applies
high-risk-style controls voluntarily: risk management (this card, the red team), data governance
(data card, licences), logging (hash-chained events), human oversight (review, no auto-deny) and
accuracy reporting (the evaluations above).
