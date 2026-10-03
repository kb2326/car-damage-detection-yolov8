"""Label taxonomies: map every source's label names onto one canonical class list."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import model_validator

from claimlens.domain import Frozen


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


class UnknownLabelError(ValueError):
    def __init__(self, taxonomy: str, label: str) -> None:
        super().__init__(f"label {label!r} is not in the {taxonomy} taxonomy")
        self.label = label


class Taxonomy(Frozen):
    name: str
    classes: tuple[str, ...]
    ignore: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("taxonomy classes must be unique")
        overlap = set(self.classes) & set(self.ignore)
        if overlap:
            raise ValueError(f"labels listed as both class and ignored: {sorted(overlap)}")
        return self

    def canonical(self, label: str) -> str | None:
        """Return the canonical class, None for an ignored label, or raise for an unknown one."""
        key = normalize_label(label)
        if key in self.classes:
            return key
        if key in self.ignore:
            return None
        raise UnknownLabelError(self.name, label)

    def index(self, label: str) -> int:
        return self.classes.index(label)


def load_taxonomies(path: Path) -> dict[str, Taxonomy]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        name: Taxonomy.model_validate({"name": name, **body})
        for name, body in data.items()
        if isinstance(body, dict) and name != "part_groups"
    }


class PartGroups(Frozen):
    """Coarse part groups, each a set of fine-grained part classes."""

    classes: tuple[str, ...]
    members: dict[str, tuple[str, ...]]
    # Damage type -> the only groups it can be on (e.g. a flat tyre only on a wheel).
    # Damage types not listed can be on any part.
    damage_parts: dict[str, tuple[str, ...]] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.members) != set(self.classes):
            raise ValueError("part group members must be defined for exactly the group classes")
        for damage, groups in self.damage_parts.items():
            unknown = sorted(set(groups) - set(self.classes))
            if unknown:
                raise ValueError(f"damage_parts.{damage} names unknown part group(s): {unknown}")
        seen: set[str] = set()
        for parts in self.members.values():
            for part in parts:
                if part in seen:
                    raise ValueError(f"part {part!r} is in more than one group")
                seen.add(part)
        return self

    def plausible(self, damage_type: str, group: str) -> bool:
        allowed = self.damage_parts.get(damage_type)
        return allowed is None or group in allowed

    def group_of(self, part: str) -> str:
        for group, parts in self.members.items():
            if part in parts:
                return group
        raise UnknownLabelError("part_groups", part)

    def as_taxonomy(self) -> Taxonomy:
        return Taxonomy(name="part_groups", classes=self.classes)


def load_part_groups(path: Path) -> PartGroups:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return PartGroups.model_validate(data["part_groups"])
