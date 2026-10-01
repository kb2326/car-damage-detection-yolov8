# AGENTS.md

Guidance for AI coding agents working in this repository.

- Package code lives in `src/claimlens`. Do not import from or modify `legacy/`.
- Use `uv run …` for all commands. Checks: `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy`, `uv run pytest`.
- Never commit anything under `data/` or `models/`, or any `.env` file.
- Follow the design in `docs/specs/` and record significant decisions as ADRs in `docs/adr/`.
- Principles: models measure, the agent reasons, rules decide; the system never auto-denies a claim.
