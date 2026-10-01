from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.taxonomy import Taxonomy, UnknownLabelError, load_taxonomies, normalize_label

ROOT = Path(__file__).resolve().parents[2]
DAMAGE = Taxonomy(name="damage", classes=("crack", "glass_shatter"), ignore=("smash",))


def test_normalize_label_handles_spaces_hyphens_and_case() -> None:
    assert normalize_label(" Glass-Shatter ") == "glass_shatter"
    assert normalize_label("tire flat") == "tire_flat"


def test_canonical_maps_by_name() -> None:
    assert DAMAGE.canonical("glass shatter") == "glass_shatter"
    assert DAMAGE.index("glass_shatter") == 1


def test_ignored_label_returns_none() -> None:
    assert DAMAGE.canonical("smash") is None


def test_unknown_label_raises() -> None:
    with pytest.raises(UnknownLabelError, match="Front-Windscreen-Damage"):
        DAMAGE.canonical("Front-Windscreen-Damage")


def test_duplicate_or_overlapping_classes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Taxonomy(name="x", classes=("dent", "dent"))
    with pytest.raises(ValidationError, match="both"):
        Taxonomy(name="x", classes=("dent",), ignore=("dent",))


def test_repo_taxonomies_load() -> None:
    taxonomies = load_taxonomies(ROOT / "config" / "taxonomy.toml")
    assert taxonomies["damage"].classes[0] == "crack"
    assert len(taxonomies["damage"].classes) == 6
    assert len(taxonomies["parts"].classes) == 22
    assert taxonomies["parts"].canonical("object") is None
