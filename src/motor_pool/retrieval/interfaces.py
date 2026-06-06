"""Retrieval backend protocols and the public Retriever tool interface.

`Retriever` is the V1 deliverable that V2 wraps as one tool. It is intentionally
minimal: exactly `retrieve(query, *, top_k, filters) -> list[RetrievedChunk]`.
It is stateless, side-effect-free, and touches no model weights. That
statelessness is the testable boundary for "retrieval handles facts".

`ProcedureFetcher` is a separate seam, kept distinct so V2 can register
procedure fetch as its own deterministic tool rather than bolting it onto the
retrieval tool. HybridRetriever happens to implement both, but they are
independent surfaces.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..schemas import Chunk, RetrievalFilters, RetrievedChunk


class Embedder(Protocol):
    """A dense text embedder. The implementation owns its query/doc prefixes."""

    model_id: str
    dim: int

    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


class Bm25Index(Protocol):
    """A lexical index returning (chunk_id, score) pairs."""

    def search(self, query: str, top_n: int) -> list[tuple[str, float]]: ...


class VectorStore(Protocol):
    """A dense index over chunk embeddings, plus chunk lookup by id."""

    def search(self, query_vec: np.ndarray, top_n: int) -> list[tuple[str, float]]: ...

    def get_chunk(self, chunk_id: str) -> Chunk: ...


@runtime_checkable
class Retriever(Protocol):
    """The tool-callable retrieval interface. The V2 agent registers this."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class ProcedureFetcher(Protocol):
    """Fetch the sibling chunks under a parent section. A separate V2 tool seam."""

    def get_procedure(
        self, source_doc_id: str, parent_id: str
    ) -> list[Chunk] | None: ...
