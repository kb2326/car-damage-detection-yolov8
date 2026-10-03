"""Agent Skills: short adjuster procedures in skills/<name>/SKILL.md, loaded only once approved.

Skills are written by people (or by Claude and approved by the owner), never by an agent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from claimlens.domain import Frozen

MIN_BODY = 50


class Skill(Frozen):
    name: str
    description: str
    version: int
    approved_by: str
    approved_on: str
    body: str


def _value(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def parse_skill(path: Path) -> Skill:
    """Front matter between two `---` lines (`key: value` pairs), then the procedure."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"{path}: a skill starts with front matter between --- lines")
    front, body = text[4:].split("\n---\n", 1)
    fields: dict[str, str] = {}
    for line in front.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, raw = line.partition(":")
            fields[key.strip()] = _value(raw.split(" #")[0])
    for required in ("name", "description"):
        if not fields.get(required):
            raise ValueError(f"{path}: front matter needs a {required}")
    if fields["name"] != path.parent.name:
        raise ValueError(f"{path}: name {fields['name']!r} must match its folder")
    try:
        version = int(fields.get("version", "1"))
    except ValueError:
        raise ValueError(f"{path}: version must be a whole number") from None
    if len(body.strip()) < MIN_BODY:
        raise ValueError(f"{path}: the procedure is too short")
    return Skill(
        name=fields["name"],
        description=fields["description"],
        version=version,
        approved_by=fields.get("approved_by", ""),
        approved_on=fields.get("approved_on", ""),
        body=body.strip(),
    )


def _approval_problem(skill: Skill) -> str | None:
    if not skill.approved_by.strip():
        return "not approved by the owner yet"
    try:
        date.fromisoformat(skill.approved_on)
    except ValueError:
        return "approved_on must be a date (YYYY-MM-DD)"
    return None


def load_skills_report(root: Path) -> tuple[dict[str, Skill], list[str]]:
    """Approved skills by name, and why each other skill was skipped."""
    loaded: dict[str, Skill] = {}
    skipped: list[str] = []
    for path in sorted(root.glob("*/SKILL.md")):
        try:
            skill = parse_skill(path)
        except ValueError as exc:
            skipped.append(f"{path.parent.name}: {exc}")
            continue
        problem = _approval_problem(skill)
        if problem is None:
            loaded[skill.name] = skill
        else:
            skipped.append(f"{skill.name}: {problem}")
    return loaded, skipped


def load_skills(root: Path) -> dict[str, Skill]:
    return load_skills_report(root)[0]
