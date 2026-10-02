"""bge-small-en-v1.5 through fastembed (ONNX on CPU; downloads ~130 MB once)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class FastEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model: Any = TextEmbedding(model_name=model)
        self.name = model
        self.dim = 384

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._model.embed(list(texts))]
