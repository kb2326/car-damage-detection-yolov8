"""`claimlens judge run | export | agreement`: the LLM judge and its check against the owner."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Callable
from pathlib import Path

from claimlens.evals.agreement import compute_agreement
from claimlens.evals.judge import Judgement, judge_item
from claimlens.evals.judge_export import (
    load_labels,
    load_run,
    render_label_page,
    sample_for_labelling,
)
from claimlens.evals.scorecard import Scorecard
from claimlens.llm.gateway import Gateway
from claimlens.llm.prompts import load_prompt

# per_day_usd -> gateway; the default builds the real gateway (needs ANTHROPIC_API_KEY).
JudgeGatewayFactory = Callable[[float | None], Gateway]


def add_judge_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    judge = sub.add_parser("judge", help="LLM judge: run it, export items to label, agreement")
    jsub = judge.add_subparsers(dest="judge_command", required=True)
    run = jsub.add_parser("run", help="judge every saved recommendation in a run")
    run.add_argument("--run-dir", type=Path, required=True, help="from eval-triage --save-run")
    run.add_argument("--out", type=Path, required=True, help="judgements .jsonl to write")
    run.add_argument("--prompt", default="judge/v1", help="prompt name/version")
    run.add_argument("--llm-daily-cap", type=float, default=None)
    export = jsub.add_parser("export", help="write the labelling page for the owner")
    export.add_argument("--judgements", type=Path, required=True)
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True, help="HTML page to write")
    export.add_argument("-n", type=int, default=50, help="items to label")
    agree = jsub.add_parser("agreement", help="compare the judge with the owner's labels")
    agree.add_argument("--judgements", type=Path, required=True)
    agree.add_argument("--labels", type=Path, required=True, help="judge-labels.json")
    agree.add_argument(
        "--scorecard", type=Path, default=None, help="add the judge pass rate if validated"
    )


def _load_judgements(path: Path) -> list[Judgement]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Judgement.model_validate_json(line) for line in lines if line.strip()]


def _sample_path(judgements: Path) -> Path:
    return judgements.parent / "sample.json"


def run_judge_command(args: argparse.Namespace, gateway_factory: JudgeGatewayFactory) -> int:
    if args.judge_command == "run":
        name, _, version = args.prompt.partition("/")
        prompt = load_prompt(Path("prompts"), name, version)
        gateway = gateway_factory(args.llm_daily_cap)
        judgements = [judge_item(gateway, prompt, item) for item in load_run(args.run_dir)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "".join(j.model_dump_json() + "\n" for j in judgements), encoding="utf-8"
        )
        passed = sum(j.verdict == "pass" for j in judgements)
        rate = passed / len(judgements) if judgements else 0.0
        cost = sum(j.cost_usd for j in judgements)
        print(f"Judge pass rate: {rate:.2f} ({passed}/{len(judgements)})  cost ${cost:.2f}")
        return 0
    if args.judge_command == "export":
        judgements = _load_judgements(args.judgements)
        # Narrative golden cases have ids n001..n053.
        narrative = {j.case_id for j in judgements if j.case_id.startswith("n")}
        sample = sample_for_labelling(judgements, narrative, n=args.n)
        items = {item.case_id: item for item in load_run(args.run_dir)}
        _sample_path(args.judgements).write_text(json.dumps(sample, indent=1), encoding="utf-8")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_label_page([items[i] for i in sample]), encoding="utf-8")
        print(f"Wrote {len(sample)} items to {args.out}")
        return 0
    loaded = _load_judgements(args.judgements)
    verdicts = {j.case_id: j.verdict for j in loaded}
    sample = json.loads(_sample_path(args.judgements).read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    result = compute_agreement({i: verdicts[i] for i in sample}, labels)
    bar = tomllib.loads(Path("config/eval_gate.toml").read_text(encoding="utf-8"))
    min_agreement, min_kappa = bar["judge_min_agreement"], bar["judge_min_kappa"]
    print(
        f"Agreement: {result.agreement:.2f}, kappa {result.kappa:.2f} (n={result.n})\n"
        f"Both good {result.both_pass}, both bad {result.both_fail}, "
        f"judge good but you bad {result.judge_pass_human_fail}, "
        f"judge bad but you good {result.judge_fail_human_pass}"
    )
    if result.agreement < min_agreement or result.kappa < min_kappa:
        print(f"Judge is below the bar (agreement {min_agreement:.2f}, kappa {min_kappa:.2f})")
        return 1
    print("Judge is validated")
    if args.scorecard is not None:
        card = Scorecard.model_validate_json(args.scorecard.read_text(encoding="utf-8"))
        rate = sum(j.verdict == "pass" for j in loaded) / len(loaded)
        card = card.model_copy(
            update={
                "metrics": {**card.metrics, "judge_pass_rate": rate},
                "versions": {**card.versions, "judge": loaded[0].judge_version},
            }
        )
        args.scorecard.write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        print(f"Judge pass rate {rate:.2f} added to {args.scorecard}")
    return 0
