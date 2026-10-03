"""`claimlens eval-intake`: run the simulated customers against the real intake agent."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any


def add_eval_intake_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ev = sub.add_parser("eval-intake", help="score the intake agent on simulated customers")
    ev.add_argument("--personas", type=Path, default=Path("evals/intake/personas.jsonl"))
    ev.add_argument("-k", type=int, default=4, help="trials per customer (pass^k)")
    ev.add_argument("--report", type=Path, required=True)
    ev.add_argument("--only", nargs="*", default=None, help="persona ids to run")
    ev.add_argument("--llm-daily-cap", type=float, default=None)


def _spent(log: Path, start: int) -> float:
    if not log.is_file():
        return 0.0
    lines = log.read_text(encoding="utf-8").splitlines()[start:]
    return sum(float(json.loads(line)["cost_usd"]) for line in lines if line.strip())


def _simulated(name: str) -> Callable[[Any], str]:
    """Files nothing: a simulated claim only needs an id."""

    def submit(state: Any) -> str:
        return f"simulated-{name}"

    return submit


def run_eval_intake(args: argparse.Namespace) -> int:
    from claimlens.evals.intake_sim import (
        customer_model,
        load_personas,
        pass_at_k,
        render_intake_report,
        run_trial,
        write_fixture_photos,
    )
    from claimlens.intake_agent.session import build_intake
    from claimlens.llm.factory import build_gateway
    from claimlens.llm.prompts import load_prompt
    from claimlens.llm.types import LLMError

    personas = load_personas(args.personas)
    if args.only:
        personas = [p for p in personas if p.persona_id in set(args.only)]
    today = date.today()
    template = load_prompt(Path("prompts"), "customer", "v1").text
    log = Path("var/llm-calls.jsonl")
    start = len(log.read_text(encoding="utf-8").splitlines()) if log.is_file() else 0
    trials = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        photos = write_fixture_photos(work / "photos")
        for persona in personas:
            for n in range(args.k):
                name = f"{persona.persona_id}-{n}"
                try:
                    # A fresh response cache per trial, so the k trials are independent.
                    gateway = build_gateway(
                        args.config,
                        Path.cwd(),
                        per_day_usd=args.llm_daily_cap,
                        cache_path=work / f"cache-{name}.sqlite",
                    )
                except LLMError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
                sessions = build_intake(
                    args.config,
                    Path.cwd(),
                    work / f"intake-{name}.sqlite",
                    lambda: None,  # type: ignore[arg-type,return-value]
                    gateway=gateway,
                    process=False,
                    submit=_simulated(name),
                )
                trial = run_trial(
                    persona,
                    sessions,
                    customer_model(gateway, template, persona, today),
                    photos,
                    today=today,
                )
                trials.append(trial)
                outcome = "pass" if trial.passed else "FAIL " + "; ".join(trial.failures)
                print(f"{name}: {outcome} ({trial.turns} turns)", flush=True)
    cost = _spent(log, start)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_intake_report(trials, k=args.k, cost_usd=cost, generated_on=today),
        encoding="utf-8",
    )
    first, every = pass_at_k(trials)
    print(f"pass@1 {first:.2f}  pass^{args.k} {every:.2f}  cost ${cost:.2f}  -> {args.report}")
    return 0
