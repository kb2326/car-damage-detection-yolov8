"""LanceDB table of policy clauses with hybrid (BM25 + vector, RRF) search."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from claimlens.domain import Frozen
from claimlens.knowledge.clauses import Clause
from claimlens.knowledge.embed import Embedder

TABLE = "policy_clauses"
WORDINGS = frozenset({"basic", "standard", "premium"})


class IndexMissingError(Exception):
    pass


class ClauseHit(Frozen):
    clause_id: str
    wording: str
    title: str
    text: str
    score: float


def _lancedb() -> Any:
    import lancedb

    return lancedb


def build_index(clauses: Sequence[Clause], embedder: Embedder, path: Path) -> int:
    vectors = embedder.embed([f"{c.title}. {c.text}" for c in clauses])
    rows = [
        {
            "clause_id": c.clause_id,
            "wording": c.wording,
            "title": c.title,
            "text": c.text,
            "content": f"{c.title}. {c.text}",
            "vector": vector,
        }
        for c, vector in zip(clauses, vectors, strict=True)
    ]
    db = _lancedb().connect(str(path))
    table = db.create_table(TABLE, data=rows, mode="overwrite")
    table.create_fts_index("content", replace=True)
    return len(rows)


class PolicyIndex:
    def __init__(self, table: Any, embedder: Embedder) -> None:
        self._table = table
        self._embedder = embedder

    @classmethod
    def open(cls, path: Path, embedder: Embedder) -> PolicyIndex:
        if not path.exists():
            raise IndexMissingError(f"no policy index at {path}: run `claimlens knowledge build`")
        db = _lancedb().connect(str(path))
        try:
            table = db.open_table(TABLE)
        except (FileNotFoundError, ValueError) as exc:
            raise IndexMissingError(
                f"no policy index at {path}: run `claimlens knowledge build`"
            ) from exc
        return cls(table, embedder)

    def search(self, query: str, *, wording: str | None = None, k: int = 5) -> list[ClauseHit]:
        from lancedb.rerankers import RRFReranker

        if wording is not None and wording not in WORDINGS:
            raise ValueError(f"unknown wording {wording!r}")  # also keeps the filter injection-safe
        vector = self._embedder.embed([query])[0]
        builder = self._table.search(query_type="hybrid").vector(vector).text(query)
        if wording is not None:
            builder = builder.where(f"wording = '{wording}'", prefilter=True)
        rows = builder.rerank(RRFReranker()).limit(k).to_list()
        return [
            ClauseHit(
                clause_id=r["clause_id"],
                wording=r["wording"],
                title=r["title"],
                text=r["text"],
                score=float(r.get("_relevance_score", 0.0)),
            )
            for r in rows
        ]

    def _all(self) -> dict[str, Clause]:
        rows = self._table.to_arrow().select(["clause_id", "wording", "title", "text"]).to_pylist()
        return {r["clause_id"]: Clause.model_validate(r) for r in rows}

    def get(self, clause_id: str) -> Clause | None:
        return self._all().get(clause_id)

    def verify_citations(self, clause_ids: Sequence[str]) -> list[str]:
        known = self._all()
        return [c for c in clause_ids if c not in known]
