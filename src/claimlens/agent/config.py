"""Settings of the LLM triage agent."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field, field_validator

from claimlens.domain import Frozen

WRITE_TOOLS = frozenset({"add_note", "assign_queue", "issue_payment"})


class AgentConfig(Frozen):
    version: str
    tier: str
    prompt_name: str
    prompt_version: str
    profile: str
    tools: tuple[str, ...]
    max_steps: int = Field(ge=1, le=20)
    max_seconds: float = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_repairs: int = Field(ge=0, le=3)
    skills_dir: str = ""  # empty: no skills

    @field_validator("tools")
    @classmethod
    def _read_only(cls, tools: tuple[str, ...]) -> tuple[str, ...]:
        if WRITE_TOOLS & set(tools):
            raise ValueError("the triage agent is read-only: remove write tools from [tools]")
        return tools


def load_agent_config(path: Path) -> AgentConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    name, _, version = str(data["prompt"]).partition("/")
    return AgentConfig(
        version=data["version"],
        tier=data["tier"],
        prompt_name=name,
        prompt_version=version,
        profile=data["profile"],
        tools=tuple(data["tools"]),
        skills_dir=data.get("skills_dir", ""),
        **data["limits"],
    )
