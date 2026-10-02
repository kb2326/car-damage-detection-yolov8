"""`claimlens knowledge build | search | eval`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from claimlens.knowledge.clauses import load_wordings
from claimlens.knowledge.embed import Embedder
from claimlens.knowledge.index import IndexMissingError, PolicyIndex, build_index
from claimlens.policy import load_policies

EmbedderFactory = Callable[[], Embedder]


def add_knowledge_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    knowledge = sub.add_parser("knowledge", help="policy wordings: build, search, evaluate")
    ksub = knowledge.add_subparsers(dest="knowledge_command", required=True)
    for name in ("build", "search", "eval"):
        p = ksub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path("."))
        p.add_argument("--index", type=Path, default=Path("var/lancedb"))
        if name == "search":
            p.add_argument("query")
            p.add_argument("--policy", default=None, help="policy id, e.g. P-1001")
            p.add_argument("-k", type=int, default=5)
        if name == "eval":
            p.add_argument("--out", type=Path, default=None)


def recall_at_k(
    index: PolicyIndex,
    questions: Sequence[dict[str, str]],
    wording_of: Callable[[str], str],
    k: int,
) -> tuple[float, list[tuple[str, str, list[str]]]]:
    rows: list[tuple[str, str, list[str]]] = []
    for q in questions:
        hits = index.search(q["question"], wording=wording_of(q["policy_id"]), k=k)
        rows.append((q["question"], q["expected"], [h.clause_id for h in hits]))
    found = sum(expected in ids for _, expected, ids in rows)
    return found / max(len(rows), 1), rows


def run_knowledge_command(args: argparse.Namespace, *, embedder_factory: EmbedderFactory) -> int:
    policies = load_policies(args.config / "policies.toml")

    def wording_of(policy_id: str) -> str:
        record = policies.get_record(policy_id)
        if record is None:
            raise ValueError(f"unknown policy {policy_id}")
        return record.wording

    try:
        if args.knowledge_command == "build":
            clauses = load_wordings(args.root / "knowledge" / "policies")
            count = build_index(clauses, embedder_factory(), args.index)
            print(f"Indexed {count} clauses into {args.index}")
            return 0
        index = PolicyIndex.open(args.index, embedder_factory())
        if args.knowledge_command == "search":
            wording = wording_of(args.policy) if args.policy else None
            for hit in index.search(args.query, wording=wording, k=args.k):
                print(f"{hit.clause_id}  {hit.title}  ({hit.score:.3f})\n    {hit.text}")
            return 0
        lines = (args.root / "evals" / "knowledge" / "questions.jsonl").read_text(encoding="utf-8")
        questions = [json.loads(line) for line in lines.splitlines() if line.strip()]
        recall, rows = recall_at_k(index, questions, wording_of, 5)
        report = [
            "# Policy search v1",
            "",
            f"- Date: {date.today().isoformat()}",
            f"- Questions: {len(rows)}",
            f"- **recall@5: {recall:.2f}** (target ≥ 0.90)",
            "",
            "| Question | Expected | Top 5 | Found |",
            "|---|---|---|---|",
            *[
                f"| {q} | {e} | {', '.join(ids)} | {'yes' if e in ids else 'no'} |"
                for q, e, ids in rows
            ],
        ]
        out = (
            args.out
            or args.root / "evals" / "reports" / f"{date.today().isoformat()}-policy-search-v1.md"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"recall@5 = {recall:.2f}; report written to {out}")
        return 0
    except (IndexMissingError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
