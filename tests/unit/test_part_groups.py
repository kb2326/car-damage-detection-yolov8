from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.records import Annotation
from claimlens.data.taxonomy import (
    PartGroups,
    UnknownLabelError,
    load_part_groups,
    load_taxonomies,
)

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_FILE = ROOT / "config" / "taxonomy.toml"


def test_every_part_class_belongs_to_exactly_one_group() -> None:
    groups = load_part_groups(TAXONOMY_FILE)
    parts = load_taxonomies(TAXONOMY_FILE)["parts"]
    members = [part for group in groups.members.values() for part in group]
    assert sorted(members) == sorted(parts.classes)
    assert groups.group_of("front_left_door") == "door"
    assert groups.as_taxonomy().classes == groups.classes


def test_unknown_part_raises() -> None:
    with pytest.raises(UnknownLabelError, match="fender"):
        load_part_groups(TAXONOMY_FILE).group_of("fender")


def test_groups_must_match_members() -> None:
    with pytest.raises(ValidationError, match="members"):
        PartGroups(classes=("door",), members={"hood": ("hood",)})
    with pytest.raises(ValidationError, match="more than one group"):
        PartGroups(classes=("a", "b"), members={"a": ("x",), "b": ("x",)})


def test_taxonomies_ignore_the_part_groups_section() -> None:
    assert "part_groups" not in load_taxonomies(TAXONOMY_FILE)


def test_annotation_score_is_optional() -> None:
    assert Annotation(label="door", polygon=(0, 0, 1, 0, 1, 1)).score is None
    assert Annotation(label="door", polygon=(0, 0, 1, 0, 1, 1), score=0.5).score == 0.5
