"""Dense vector store: brute-force cosine over a numpy array.

For a corpus of 10^3 to 10^4 chunks this is sub-millisecond, zero extra deps,
and fully inspectable. Embeddings are normalized at build time, so cosine
similarity is a plain dot product. sqlite-vec is the documented upgrade path.
"""

from __future__ import annotations

import numpy as np

from motor_pool.schemas import Chunk


class NumpyVectorStore:
    """VectorStore implementation over an in-memory float32 matrix."""

    def __init__(
        self, embeddings: np.ndarray, chunk_ids: list[str], chunks: list[Chunk]
    ) -> None:
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._chunk_ids = chunk_ids
        self._chunks = {c.chunk_id: c for c in chunks}

    def search(self, query_vec: np.ndarray, top_n: int) -> list[tuple[str, float]]:
        if self._embeddings.shape[0] == 0:
            return []
        sims = self._embeddings @ np.asarray(query_vec, dtype=np.float32)
        n = min(top_n, sims.shape[0])
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]
        return [(self._chunk_ids[int(i)], float(sims[int(i)])) for i in top]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]
