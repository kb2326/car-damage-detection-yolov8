"""Settings of the intake agent."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from claimlens.domain import Frozen


class IntakeConfig(Frozen):
    version: str
    tier: str
    prompt_name: str
    prompt_version: str
    profile: str
    photo_kinds: tuple[str, ...]
    max_turns: int = Field(ge=1)
    max_steps_per_turn: int = Field(ge=1)
    max_retakes: int = Field(ge=0)
    max_tokens: int = Field(gt=0)
    min_brightness: float = Field(ge=0, le=255)
    min_edge_variance: float = Field(ge=0)


def load_intake_config(path: Path) -> IntakeConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    name, _, version = str(data["prompt"]).partition("/")
    return IntakeConfig(
        version=data["version"],
        tier=data["tier"],
        prompt_name=name,
        prompt_version=version,
        profile=data["profile"],
        photo_kinds=tuple(data["photo_kinds"]),
        **data["limits"],
        **data["photo_coaching"],
    )
