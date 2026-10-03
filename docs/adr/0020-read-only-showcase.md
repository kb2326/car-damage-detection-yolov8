# ADR 0020: A read-only public showcase on Hugging Face Spaces

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

The project needs a public face: something a reviewer can open in a browser. The owner's aim is
proof of work, not a service: "let them see what the product would look like in Hugging Face and
how it's built via GitHub". A live public demo would need sign-in, budget protection against
strangers' LLM spend, the CarDD-trained weights on a public server, and abuse handling.

## Decision

- **Showcase mode of the real app** (`claimlens serve --showcase --data showcase`), not a separate
  mock-up:
  - one middleware refuses every state-changing request (403), whatever route it reaches;
  - the event store opens read-only (SQLite `mode=ro`);
  - the chat page replays one recorded conversation; the claim page hides review and resume;
  - a banner links to the repository, the system card, the model card and the photo credits;
  - standard headers on every page: `nosniff`, a same-origin `Content-Security-Policy` (inline
    style attributes allowed for the damage boxes; not on `/docs`, whose Swagger UI uses a CDN),
    `Referrer-Policy: same-origin`;
  - it may listen on `0.0.0.0` only in this mode; the Host check allows `*.hf.space` and local
    names (`CLAIMLENS_ALLOWED_HOSTS`).
- **Recorded sample claims** (`showcase/`, 1.8 MB, committed), built once by
  `scripts/build_showcase.py` with the real fused vision models and LLM agents:
  - photos: Wikimedia Commons images of damaged cars under CC BY / CC BY-SA, credited in
    `showcase/CREDITS.md`, readable number plates blurred. `car-seg` photos were screened first
    but show undamaged cars, so they would only have shown the model's weak false positives;
  - five claims: a shattered mirror (the glass procedure; the agent questioned a high estimate,
    R8), a junction crash (a weak finding, R6), a heavy front crash (over $3,000, R5), a re-saved
    copy of the mirror photo on another policy (near-copy signal, fraud review, R1; a person then
    denied it), and a parked van damaged by falling bricks, filed through a recorded 12-turn chat
    (R5; a reviewer asked for more information);
  - every claim's hash chain verifies (tested).
- **No weights, no key, no LLM calls** in the image or the Space: the showcase only reads recorded
  events. The image is `python:3.12-slim` with the web and agent groups (no PyTorch), runs as a
  non-root user on port 7860.
- **Deployment:** `scripts/deploy_space.py` stages only the app, its config and the showcase data
  (tested: no data, weights, tests or secrets) and uploads them with the Hugging Face Hub API. The
  `deploy-space` workflow runs it by hand or on a release, with the `HF_TOKEN` secret and the
  `HF_SPACE` variable. CI is unchanged.
- **Dropped from M8:** drift monitoring (nothing live to monitor), sign-in (nothing to protect).

## Consequences

- Anyone can see the product and its reasoning; nobody can use or abuse it, and it costs nothing
  to run (free CPU Space; it sleeps after about 48 hours without visitors).
- What the showcase shows is honest: of five claims none is fast-tracked, because the agent and
  the rules are cautious on these photos; the reasons are on each claim page.
- Building the samples surfaced an intake weakness: when a customer cannot send a requested photo,
  the agent keeps asking until the chat's spending cap hands the claim over, instead of recording
  the gap. Listed as an open item.
- The image is 1.15 GB, mostly the agent libraries, which showcase mode imports but never calls;
  a lighter image would need the web layer to import them lazily.
