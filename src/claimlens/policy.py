"""Mock policy administration system: coverage lookup for fictional policies."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from claimlens.domain import Coverage, Frozen


class PolicyStatus(StrEnum):
    ACTIVE = "active"
    LAPSED = "lapsed"


class PolicyRecord(Frozen):
    policy_id: str
    status: PolicyStatus
    collision: bool
    deductible: int = Field(ge=0)
    wording: Literal["basic", "standard", "premium"] = "standard"


class PolicyRepository:
    def __init__(self, records: Iterable[PolicyRecord]) -> None:
        self._by_id: dict[str, PolicyRecord] = {}
        for record in records:
            if record.policy_id in self._by_id:
                raise ValueError(f"duplicate policy id {record.policy_id}")
            self._by_id[record.policy_id] = record

    def get_record(self, policy_id: str) -> PolicyRecord | None:
        return self._by_id.get(policy_id)

    def get_coverage(self, policy_id: str) -> Coverage:
        record = self._by_id.get(policy_id)
        if record is None:
            return Coverage(
                policy_id=policy_id, found=False, active=False, collision=False, deductible=0
            )
        return Coverage(
            policy_id=policy_id,
            found=True,
            active=record.status is PolicyStatus.ACTIVE,
            collision=record.collision,
            deductible=record.deductible,
        )


def load_policies(path: Path) -> PolicyRepository:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return PolicyRepository(PolicyRecord.model_validate(item) for item in data.get("policy", []))
