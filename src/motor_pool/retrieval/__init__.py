"""Hybrid retrieval: BM25 + dense embeddings fused with reciprocal rank fusion.

The `Retriever` protocol in `interfaces` is the clean, tool-callable seam that
V2 will register as one tool among deterministic tools. In V1 it is implemented
only by `HybridRetriever`.
"""

from __future__ import annotations

from .hybrid_retriever import HybridRetriever, load_retriever
from .interfaces import Bm25Index, Embedder, ProcedureFetcher, Retriever, VectorStore
from .rrf import reciprocal_rank_fusion

__all__ = [
    "Bm25Index",
    "Embedder",
    "HybridRetriever",
    "ProcedureFetcher",
    "Retriever",
    "VectorStore",
    "load_retriever",
    "reciprocal_rank_fusion",
]
