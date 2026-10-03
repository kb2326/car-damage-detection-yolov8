# ADR 0014: A committed scorecard gate, and an LLM judge checked against the owner

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

M5a gave ClaimLens an LLM triage agent. Two questions follow:

1. **Is it any good?** Golden v1 cannot say. Its descriptions are generic, so it tests the models
   and the rules, not the agent's reading of a story.
2. **How do we keep it good?** A prompt, rule or model change can quietly make it worse.

CI has no API key and no model weights, so it cannot re-run the evaluation itself.

## Decision

- **Golden v2 (150 cases):** golden v1 unchanged, plus 53 narrative cases written from the policy
  wordings: 43 must go to a person, 10 are harmless stories that should be fast-tracked. Each
  reuses a v1 photo that passes rules R1–R6 and fits the story, so the story decides the route.
  Labels are AI-authored and not yet reviewed by a person; reports say so.
- **Metrics computed by code:** citation validity, right clause cited, narrative cases caught,
  harmless stories fast-tracked, agent failure rate, and cost per claim **at list price from token
  counts**, so response-cache hits cannot make a run look cheaper.
- **An LLM judge** (`prompts/judge/v1.md`, the `strong` tier) answers three yes/no questions about
  each recommendation, each with a reason:
  - is every statement grounded in the evidence or a cited clause;
  - is each citation relevant;
  - could an adjuster act on it?
  The verdict is `pass` only if all three are yes, computed in code and never read from the
  model. The judge never sees the expected route.
- **The judge must agree with the owner.** `claimlens judge export` samples 50 recommendations
  (half the judge failed, at least 20 narrative) into a self-contained page. The owner marks each
  Good or Bad, and `claimlens judge agreement` reports agreement and Cohen's kappa.
  - The bar is agreement ≥ 0.80 and kappa ≥ 0.60 (`config/eval_gate.toml`).
  - Below the bar, the judge prompt gets a new version, at most twice. After that the judge is
    "not validated" and stays out of the gate.
- **The gate checks a committed scorecard** (`claimlens eval-gate`, a CI step).
  - `evals/scorecards/current.json` holds the metrics and a sha256 of every file that can change
    the result: prompts, agent and LLM config, rules, rate card, models registry, taxonomy,
    policies and wordings, and the golden file.
  - The gate fails when:
    - any of those files changed since the evaluation ran (**stale**: re-run it);
    - escalation recall or citation validity is below 1.00 (**safety**);
    - a metric falls more than its margin below `baseline.json` (**regression**: route accuracy
      0.02, narrative catch rate 0.05, benign pass rate 0.10, judge pass rate 0.05);
    - mean cost per claim rises more than 25% (**cost**).
  - Accepting a new baseline is a deliberate commit with a reason in the pull request.

## Consequences

- **Results** (golden v2, fused detector, `triage-agent-v1+triage/v1`, three runs agree):

  | Metric | Value |
  |---|---|
  | Escalation recall | 1.00 (110 of 110) |
  | Route accuracy | 0.76 |
  | Narrative cases caught | 43 of 43 (1.00) |
  | Right clause cited, citation validity | 1.00, 1.00 |
  | Harmless stories fast-tracked | **3 of 10 (0.30)** |
  | Agent failures | 0 |
  | Cost per claim (list price) | $0.010 |
  | Judge pass rate | 0.77 (116 of 150), $0.59 to judge 150 |

- **The agent catches every story that must go to a person, but it is too cautious with harmless
  ones.** The benign pass rate (0.30) is the number a prompt change should raise, and the gate
  will hold escalation recall and the catch rate while it does.
- **The gate proved itself:** changing one character of the triage prompt made `eval-gate` fail
  as stale; restoring it passed.
- **The scorecard must be complete:** it must be for golden v2 and cover all 150 cases
  (`golden`, `min_cases` in `config/eval_gate.toml`). `eval-triage --cases` cannot write one, and a
  metric the baseline has but the new run lacks fails the gate.
- **Limits:**
  - The gate trusts the committed scorecard, so a person could commit one that no real run
    produced. Pull-request review is the control for that.
  - Source code is not fingerprinted (only config, prompts, wordings and the golden file). A change
    to the rules code or the agent code passes the gate unless it changes those files; reviewers
    must ask for a re-run.
  - The cost metric is list price, not actual spend; actual spend is in each report.
