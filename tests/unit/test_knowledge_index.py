from pathlib import Path

import pytest

from claimlens.knowledge.clauses import Clause
from claimlens.knowledge.embed import FakeEmbedder
from claimlens.knowledge.index import IndexMissingError, PolicyIndex, build_index

pytest.importorskip("lancedb")

CLAUSES = [
    Clause(
        clause_id="STD-4.1",
        wording="standard",
        title="Glass damage",
        text="Windscreen repair has no deductible.",
    ),
    Clause(
        clause_id="STD-6.1",
        wording="standard",
        title="Rental vehicle",
        text="A rental car is covered for up to 10 days after a collision.",
    ),
    Clause(
        clause_id="BAS-6.1",
        wording="basic",
        title="Rental vehicle",
        text="Rental cars are not covered.",
    ),
    Clause(
        clause_id="STD-8.1",
        wording="standard",
        title="Exclusions",
        text="Damage while racing is excluded.",
    ),
]


def _index(tmp_path: Path) -> PolicyIndex:
    embedder = FakeEmbedder()
    assert build_index(CLAUSES, embedder, tmp_path / "lancedb") == 4
    return PolicyIndex.open(tmp_path / "lancedb", embedder)


def test_keyword_query_finds_the_clause(tmp_path: Path) -> None:
    hits = _index(tmp_path).search("racing exclusion")
    assert hits[0].clause_id == "STD-8.1"


def test_wording_filter_limits_results(tmp_path: Path) -> None:
    hits = _index(tmp_path).search("rental car", wording="basic")
    assert [h.clause_id for h in hits] == ["BAS-6.1"]


def test_hybrid_ranks_the_best_match_first(tmp_path: Path) -> None:
    hits = _index(tmp_path).search(
        "is a rental car covered after a collision", wording="standard", k=3
    )
    assert hits[0].clause_id == "STD-6.1"
    assert hits[0].score >= hits[-1].score


def test_citations_are_verified(tmp_path: Path) -> None:
    index = _index(tmp_path)
    assert index.verify_citations(["STD-6.1", "BAS-6.1"]) == []
    assert index.verify_citations(["STD-6.1", "STD-99.9"]) == ["STD-99.9"]
    clause = index.get("STD-4.1")
    assert clause is not None
    assert clause.title == "Glass damage"


def test_missing_index_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(IndexMissingError, match="knowledge build"):
        PolicyIndex.open(tmp_path / "nowhere", FakeEmbedder())


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    a, b = FakeEmbedder().embed(["rental car", "rental car"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_unknown_wording_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown wording"):
        _index(tmp_path).search("rental", wording="gold' OR 1=1 --")
