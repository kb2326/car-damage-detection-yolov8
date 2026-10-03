# M8b: The Public Showcase (design)

- **Status:** Approved by the owner (2026-10-03)
- **Date:** 2026-10-03
- **Parent design:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) (section 20, M8)
- **Builds on:** ADR 0018 (web prototype), ADR 0019 (trust), the system card

## 1. Purpose

Let anyone see what ClaimLens looks like, without being able to use it. The owner's words: "this is
only for project proof … let them see what the product would look like in Hugging Face and how it's
built via GitHub". So the public demo is a **read-only showcase**: the real web app, serving
pre-recorded sample claims, with links to the repository, the system card and the model card.

## 2. Decisions taken with the owner (2026-10-03)

- **Hugging Face Spaces**, Docker SDK, free CPU tier.
- **Read-only:** no chat, no filing, no review, no LLM calls, no visitor input stored.
- **No model weights and no API key in the Space.** The showcase reads recorded events only; the
  CarDD-trained weights never leave the owner's machine (ADR 0004).
- **Dropped:** drift monitoring (nothing live to monitor) and sign-in (nothing to protect).

## 3. Showcase mode

`claimlens serve --showcase --data <folder>` (also `CLAIMLENS_SHOWCASE=1` for the container):
- every state-changing endpoint returns **403** "This is a read-only showcase. Run ClaimLens
  locally to file and review claims." (one middleware, so no endpoint can be forgotten);
- the chat page shows **one recorded conversation** (a transcript saved when the samples were built)
  instead of the live chat, with the same look;
- the claim page hides the review form and the Resume button, and shows the sample's recorded review;
- a banner on every page: "Showcase: sample claims recorded on 3 October 2026 with the real models
  and agents. Policies are fictional." with links to GitHub, the system card and the model card;
- the store is opened read-only (SQLite `mode=ro`) and memory is off;
- it may bind `0.0.0.0` inside the container (no writes are possible); the Host check allows only
  the configured public host names (`CLAIMLENS_ALLOWED_HOSTS`, the Space's `*.hf.space` address);
- standard headers: `X-Content-Type-Options: nosniff`, a restrictive Content-Security-Policy
  (self only), `Referrer-Policy: same-origin`.

## 4. Sample claims

`scripts/build_showcase.py` (run once on the owner's machine with the real models and agents, about
$0.15) builds `showcase/` from openly licensed Wikimedia Commons photos of genuinely damaged cars
(CC BY / CC BY-SA, credited; readable number plates blurred). `car-seg` photos show undamaged
cars, which would only demonstrate the model's weak false positives.
1. a shattered side mirror (glass): the clean case, expected fast-track with the glass procedure;
2. a front-end collision with some weak findings: a person reviews (R6);
3. a heavy front crash with a large estimate: a person reviews;
4. a re-saved copy of claim 1's photo: near-copy fraud signal, fraud review (R1); a person then
   denies it, showing that only a person can deny;
5. a claim filed through a recorded chat (scripted customer) and approved by a reviewer.
`showcase/claims.db`, `showcase/blobs/`, `showcase/transcript.json`, `showcase/CREDITS.md` are
committed (a few MB). `claimlens verify` passes on every claim.

## 5. Container and deployment

- `Dockerfile` (python:3.12-slim, uv, `--group web --group agent`; no vision, knowledge or training
  groups, so no PyTorch). Runs as a non-root user on port 7860.
- `hf-space/README.md`: Hugging Face front matter (`sdk: docker`, `app_port: 7860`, title, emoji,
  licence note), what the showcase is, credits, links.
- `.github/workflows/deploy-space.yml`: on a GitHub release (or manual dispatch), push the app,
  the showcase data and the Space README to `huggingface.co/spaces/<owner>/claimlens` with the
  `HF_TOKEN` secret. CI stays as it is.
- The owner creates the Hugging Face account and token once; Claude walks through it.

## 6. Testing

- Showcase mode: every POST is 403 (parametrized over all write routes); GET pages render;
  the chat page shows the transcript and no input; the review form is absent; the banner and links
  are present; a write to the store is impossible (read-only connection).
- Security headers on every response; a foreign Host is refused; `0.0.0.0` is allowed only with
  `--showcase`.
- The committed `showcase/` data: every claim verifies; every photo is from `car-seg` (listed in
  CREDITS with its source name); no file under `showcase/` comes from `data/raw/cardd*` or the
  course dataset.
- Docker: built and smoke-tested locally (`/claims` returns 200) before the first deploy.

## 7. Done means

- The Space is live and shows the five sample claims; the README links to it.
- Final README polish and project summary; the system card links the showcase.
- A LinkedIn post draft for the owner; the ClaimLens Explained page updated (when the owner is in
  that account).

## 8. Out of scope

Live chat or LLM calls in public, sign-in, monitoring, a paid host, a custom domain.
