# ADR 0017: Claim memory, Agent Skills and simulated customers

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

By M6a ClaimLens could take a claim by chat, decide it and send it to a person. Three things were
missing:
- a memory of earlier claims (a reused photo, a policy that claims often, how similar claims
  ended);
- a way to give the triage agent written procedures without bloating its prompt;
- an honest measure of the intake chat against customers who are vague, over-share or correct
  themselves.

## Decision

- **Claim memory** (`claimlens.memory`, LanceDB in `var/memory`):
  - one record per decided claim: route, the latest human review, damage and parts, the
    estimate, yes/no intake facts, the photos' perceptual hashes, a one-line summary and its
    embedding. It holds **no customer words**.
  - **Written only by the workflow:** after `RouteDecided`, and when `record_review` runs. Each
    write appends `MemoryWritten` (fields and source event numbers) to the claim's hash-chained
    log.
  - A failed write is recorded as `MemoryWriteFailed`, which the fold ignores, so it never reaches
    the rules (a `StageFailed` would have sent a re-decided claim to R2). A re-run fills in a
    missed write.
  - `claimlens memory rebuild | show | forget`. Forgetting appends `MemoryForgotten` **first**,
    for any known claim, even one that was never in memory; then it deletes the row and the old
    LanceDB versions that still held it, so the data is gone from disk. A forgotten claim is
    never written again: not by a re-run, a review or a rebuild.
  - `rebuild` skips a claim it cannot read (for example a missing photo), reports it and carries
    on.
  - **Opt-in** with `--memory`, so tests and CI never load the embedding model.
- **Near-copy photos** use the perceptual hash the data pipeline already had, not a new image
  model. The threshold is **10 bits**, from measurements on golden photos:
  - a re-save or resize moved the hash at most 2 bits, and a 2% crop at most 10;
  - two different photos were never closer than 18 (780 pairs).
  Heavier crops are not caught; that is left for M7.
- **`find_similar_claims`** (triage profile, read-only) now answers from memory. Each item gives
  a reason:
  - a near-copy photo, with its distance;
  - the same policy;
  - similar damage (vector search plus a shared damage type).
  It also gives the earlier claim's route, review, damage and cost.
  **Memory informs; it does not decide.** No rule reads it; whether a near-copy should route to
  fraud review is M7's call.
- **Agent Skills** (`skills/<name>/SKILL.md`, front matter plus a procedure):
  - three skills: `glass-claims`, `flat-tyre-claims` and `exclusion-review`;
  - only skills with `approved_by` and an `approved_on` date load (`null`, `~` and empty quotes
    count as no approval); **the owner approved all three on 2026-10-03**; a skipped skill is
    logged with the reason;
  - the triage agent (prompt `triage/v2`) sees the approved list and calls `load_skill(name)`;
  - the recommendation records `skills_used` (`name@vN`);
  - an unknown name is a plain answer, not a tool error, so a typo cannot block a fast-track.
- **The eval gate** now also fingerprints `skills/*/SKILL.md` and `config/intake.toml`.
- **Simulated customers** (`claimlens eval-intake`):
  - 10 personas with a hidden truth and a style: cooperative, vague, over-sharer, story-changer
    and photo-trouble;
  - Haiku plays the customer, and the harness answers photo requests (a dark photo first for
    photo-trouble);
  - a trial passes when intake finishes on its own with the policy, date, work use, other party,
    injuries, a police report when needed, and all three photos;
  - **pass^k** counts a persona only when all k trials pass;
  - each trial gets its own response cache, so the k trials are independent.

## Consequences

- **Golden v2 with `triage/v2` and skills** (memory switched on, but each golden case runs in its
  own empty memory so cases cannot leak into each other; `find_similar_claims` therefore found
  nothing, and cross-claim memory is covered by unit tests, not by this run):

  | Metric | Before (v1) | Now |
  |---|---|---|
  | Escalation recall | 1.00 | **1.00** |
  | Stories caught / right clause / citation validity | 1.00 / 1.00 / 1.00 | **1.00 / 1.00 / 1.00** |
  | Harmless stories fast-tracked | 0.30 | **0.50** |
  | Route accuracy | 0.76 | **0.77** |
  | Cost per claim | $0.010 | $0.011 |

  - Skills were loaded on 86 of 150 claims: glass 55, exclusion review 32, flat tyre 7.
  - The run cost $1.60, and the new scorecard is the baseline.
- **Simulated customers:** pass@1 **0.85**, pass^4 **0.80** (target 0.70); the run cost $4.35.
  - Eight of ten personas pass every trial, including the vague ones, the over-sharer with an
    injury and a police report, and the dark-photo ones.
  - **Weak spot:** a customer who corrects the date ("sorry, it was 5 days ago") often keeps the
    first date. An intake prompt v2 should record a corrected answer again.
  - One long conversation reached the per-session spending cap and was handed to a person, as
    designed.
- **Limits:**
  - the near-copy check misses heavier crops;
  - the personas are written by Claude, and none reports a claim late (more than 30 days), which
    the spec listed;
  - the stdio MCP demo server does not use memory.
