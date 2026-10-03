# ADR 0019: A hijacked-model red team, near-copy photos as fraud, and a system card

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

Before a public showcase (M8b), the owner trimmed M7 to three parts: an attack suite, the fraud
use of the claim memory that ADR 0017 left open, and a system card. About fifteen attack tests
already existed across the agent, tool, intake and web tests, but nothing tied them to a threat
model or showed coverage.

## Decision

- **Red team in "hijacked" mode.** Each attack is tested with a fake model that does what the
  attacker asks. A pass means the code controls (rules, scopes, caps, checks) hold *whatever the
  model says*, which is a stronger claim than "the model resisted".
  - `evals/redteam/attacks.toml`: 34 attacks, each mapped to an OWASP Top 10 for Agentic
    Applications risk and pointing at the test that proves its control. Existing tests are reused
    by reference; new ones cover a hijacked agent on claims the rules send to a person, code
    execution and deny paths.
  - A gate test keeps every attack pointing at a real test and every risk covered. ASI07 is not
    applicable: there is no agent-to-agent channel.
  - `claimlens eval-redteam` runs the suite and writes the report.
  - **Live mode** reuses `eval-triage` on `evals/redteam/injections.jsonl`: 15 golden stories
    that must reach a person, each with an injection (fake system message, fake adjuster note,
    "already approved", fake clause, JSON override).
- **Supply chain:** CI runs `pip-audit` on every locked dependency. One advisory is ignored, with
  its reason in the workflow: diskcache (PYSEC-2026-2447) comes only through DVC, a developer tool,
  and has no fixed release.
- **Near-copy photos are a fraud signal.** With memory on, a photo within 10 bits of a remembered
  claim's photo raises `photo_near_copy` (score 0.8), so rule R1 sends the claim to fraud review.
  An exact copy keeps `photo_reuse` (1.0) and is not reported twice. No rule or threshold changed.
  Without memory, nothing changes; the golden evaluation gives each case its own memory, so the
  scorecard is unaffected.
- **System card** (`docs/system-card.md`): purpose and limits, flow, components, evaluation,
  safeguards, limitations, privacy, an EU AI Act note. It links the model and data cards.

## Consequences

- Hijacked mode: **33 of 33 attacks run held** (the 34th, the dependency audit, runs in CI). A mutation that let the rules trust a confident
  FAST_TRACK recommendation made every hijack scenario fail, so the tests guard the control.
- Live mode (Claude Sonnet, $0.17): **15 of 15** claims went to a person, and the agent itself
  recommended review every time, naming the injection as unverified text.
- **What the hijacked suite proves, and what it does not.** It proves the rules hold for fraud
  signals, failures, photos, cover, cost and confidence (R1 to R6) whatever the model says. It does
  not cover exclusions found only in the customer's story (business use, racing): no rule reads the
  story, so the agent is the only control there, and a fully fooled agent could let such a claim
  reach R9. The live run measures that risk (0 of 15 fooled); it is not a guarantee. A code check
  of the story is the planned control (system card, section 6).
- A memory failure skips only the near-copy check; the exact-copy signal is kept (tested).
- A skipped or expected-to-fail attack test counts as not held, never as held.
- **Limits:** heavier crops, AI-generated images and photo metadata are not checked; the suite
  does not cover denial-of-service against a public server (the showcase takes no input).
