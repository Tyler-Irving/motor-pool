"""HybridRetriever: BM25 + dense, fused with RRF.

Implements both Retriever (the minimal tool surface) and ProcedureFetcher (the
separate procedure-fetch seam). Constructed by dependency injection so every
backend is swappable and stub-testable. `retrieve` is stateless and touches no
model weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from motor_pool.schemas import (
    Chunk,
    IndexManifest,
    RetrievalFilters,
    RetrievedChunk,
    RrfConfig,
)

from .bm25_index import Bm25sIndex
from .interfaces import Bm25Index, Embedder, VectorStore
from .rrf import reciprocal_rank_fusion
from .vector_store_numpy import NumpyVectorStore


def _passes(chunk: Chunk, filters: RetrievalFilters | None) -> bool:
    if filters is None:
        return True
    if filters.source_doc_id and chunk.citation.source_doc_id != filters.source_doc_id:
        return False
    if filters.content_type and chunk.content_type != filters.content_type:
        return False
    return True


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        bm25: Bm25Index,
        vectors: VectorStore,
        chunks: list[Chunk],
        cfg: RrfConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.bm25 = bm25
        self.vectors = vectors
        self.chunks = {c.chunk_id: c for c in chunks}
        self.cfg = cfg or RrfConfig()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        limit = self.cfg.top_k if top_k is None else top_k
        # With a filter, fan out over the whole corpus so a matching chunk that
        # ranks below top_n in both backends is not starved.
        fan = max(self.cfg.top_n, len(self.chunks)) if filters else self.cfg.top_n
        lexical = self.bm25.search(query, fan)
        dense = self.vectors.search(self.embedder.embed_query(query), fan)
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in lexical], [cid for cid, _ in dense]],
            k=self.cfg.k,
            weights=[self.cfg.lexical_weight, self.cfg.dense_weight],
        )
        out: list[RetrievedChunk] = []
        for chunk_id, score in fused:
            chunk = self.chunks.get(chunk_id)
            if chunk is None or not _passes(chunk, filters):
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                    citation=chunk.citation,
                    content_type=chunk.content_type,
                    parent_id=chunk.parent_id,
                    section_path=chunk.section_path,
                    test_reference=chunk.test_reference,
                    goto_targets=chunk.goto_targets,
                )
            )
            if len(out) >= limit:
                break
        return out

    def get_procedure(
        self, source_doc_id: str, parent_id: str
    ) -> list[Chunk] | None:
        siblings = [
            c
            for c in self.chunks.values()
            if c.parent_id == parent_id and c.citation.source_doc_id == source_doc_id
        ]
        siblings.sort(key=lambda c: (c.citation.pdf_page_index, c.chunk_id))
        return siblings or None


def load_retriever(
    indexes_dir: str | Path, embedder: Embedder, cfg: RrfConfig | None = None
) -> HybridRetriever:
    """Build a HybridRetriever from the artifacts written by build_indexes."""
    indexes_dir = Path(indexes_dir)
    lines = (indexes_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
    embeddings = np.load(indexes_dir / "dense" / "embeddings.npy")

    # The manifest is written last by build_indexes, so its presence signals a
    # complete build. Validate it against the embedder and the loaded chunks so a
    # config/model change without a rebuild fails loudly instead of mixing spaces.
    manifest = IndexManifest.model_validate_json(
        (indexes_dir / "dense" / "index_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.embedder_id != embedder.model_id:
        raise ValueError(
            f"index built with embedder {manifest.embedder_id!r}, querying with "
            f"{embedder.model_id!r}. Rebuild with `motor-pool index`."
        )
    if manifest.dims != embeddings.shape[1]:
        raise ValueError(
            f"index manifest dims {manifest.dims} != embeddings dim "
            f"{embeddings.shape[1]}. Rebuild with `motor-pool index`."
        )
    if not (manifest.chunk_count == embeddings.shape[0] == len(chunks)):
        raise ValueError(
            f"index out of sync: manifest {manifest.chunk_count} / "
            f"{embeddings.shape[0]} embeddings / {len(chunks)} chunks. "
            "Rebuild with `motor-pool index`."
        )
    vectors = NumpyVectorStore(embeddings, [c.chunk_id for c in chunks], chunks)
    bm25 = Bm25sIndex.load(indexes_dir / "bm25")
    return HybridRetriever(embedder, bm25, vectors, chunks, cfg)
