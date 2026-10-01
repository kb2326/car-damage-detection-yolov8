# M0 retrospective: foundations

- **Dates:** 2026-10-01
- **Pull requests:** #1 (restructure, spec, roadmap), #2 (PR/FAQ)

## What we shipped

- Audit of the course project: the app mapped a 7-class model onto a hard-coded 17-class list, so
  every displayed label was wrong; validation mAP50 was 0.137.
- Standard repository layout (`src/`, `tests/`, `docs/`, `data/`, `models/`, `legacy/`), uv,
  ruff, mypy `--strict`, pytest, pre-commit and GitHub Actions CI.
- Design spec, plain-language roadmap, PR/FAQ, ADR 0001-0002, AGENTS.md.
- Repository renamed to `kb2326/claimlens` with description and topics.

## What went well

- Moving files instead of deleting them: 339 dataset images and the model weights existed only on
  local disk and were preserved.
- Writing the PR/FAQ first gave every later decision a reference point.

## What was harder than expected

- GitHub authentication: a `GITHUB_TOKEN` for another account overrode the repo owner's login, and
  pushing a CI workflow needed the `workflow` scope.
- Windows line endings needed a `.gitattributes` file.

## What we will change

- Keep data and weights backed up outside this laptop until DVC is set up in M2.
