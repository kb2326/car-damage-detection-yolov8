"""Split a policy wording into citable clauses (`### STD-6.1 Rental vehicle`)."""

from __future__ import annotations

import re
from pathlib import Path

from claimlens.domain import Frozen

PREFIXES = {"basic": "BAS", "standard": "STD", "premium": "PRM"}
_HEADING = re.compile(r"^###\s+([A-Z]{3}-\d+(?:\.\d+)?)\s+(.+?)\s*$")


class Clause(Frozen):
    clause_id: str
    wording: str
    title: str
    text: str


def parse_policy(path: Path) -> list[Clause]:
    wording = path.stem
    if wording not in PREFIXES:
        raise ValueError(f"unknown wording file {path.name}; expected one of {sorted(PREFIXES)}")
    clauses: list[Clause] = []
    current: tuple[str, str] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            clauses.append(
                Clause(
                    clause_id=current[0],
                    wording=wording,
                    title=current[1],
                    text=" ".join(body).strip(),
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            clause_id, title = match.groups()
            if not clause_id.startswith(PREFIXES[wording] + "-"):
                raise ValueError(
                    f"{path.name}: {clause_id} does not use the {PREFIXES[wording]} prefix"
                )
            current, body = (clause_id, title), []
        elif current is not None and line.strip() and not line.startswith("#"):
            body.append(line.strip())
    flush()
    ids = [c.clause_id for c in clauses]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicate clause ids {duplicates}")
    return clauses


def load_wordings(root: Path) -> list[Clause]:
    return [clause for path in sorted(root.glob("*.md")) for clause in parse_policy(path)]
