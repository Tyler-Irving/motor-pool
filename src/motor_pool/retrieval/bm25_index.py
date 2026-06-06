"""Lexical index backed by bm25s.

The tokenizer preserves alphanumeric identifiers (paragraph numbers like
"2-104", STE/ICE "test" tokens) so a query for "para 2-104" still matches. bm25s
is imported lazily so the package imports without the retrieve extra installed.
The index plus a row->chunk_id map save to a small directory under indexes/bm25/.
"""

from __future__ import annotations

import json
from pathlib import Path

from motor_pool.schemas import Chunk

# Keep tokens that start with an alphanumeric and may contain hyphens/dots, so
# "2-104.1" and "psi" survive but bare punctuation does not.
_TOKEN_PATTERN = r"(?u)\b[a-z0-9][a-z0-9.\-]*\b"


class Bm25sIndex:
    """Bm25Index implementation over chunk ids."""

    def __init__(self, retriever, chunk_ids: list[str]) -> None:
        self._retriever = retriever
        self._chunk_ids = chunk_ids

    @staticmethod
    def _tokenize(texts: list[str]):
        import bm25s

        return bm25s.tokenize(
            texts, lower=True, token_pattern=_TOKEN_PATTERN, show_progress=False
        )

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "Bm25sIndex":
        import bm25s

        retriever = bm25s.BM25()
        retriever.index(cls._tokenize([c.text for c in chunks]), show_progress=False)
        return cls(retriever, [c.chunk_id for c in chunks])

    def search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        if not self._chunk_ids:
            return []
        tokens = self._tokenize([query])
        k = min(top_n, len(self._chunk_ids))
        results, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        return [
            (self._chunk_ids[int(row)], float(score))
            for row, score in zip(results[0], scores[0])
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(path))
        (path / "chunk_ids.json").write_text(
            json.dumps(self._chunk_ids), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Bm25sIndex":
        import bm25s

        retriever = bm25s.BM25.load(str(path))
        chunk_ids = json.loads((path / "chunk_ids.json").read_text(encoding="utf-8"))
        return cls(retriever, chunk_ids)
