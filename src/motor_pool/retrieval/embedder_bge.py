"""Dense embedder backed by BAAI/bge-base-en-v1.5.

Implements the Embedder protocol and owns the asymmetric query/doc prefixes that
bge expects. sentence-transformers (and torch) are imported lazily so the
package imports without the retrieve extra installed.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BgeEmbedder:
    """Embedder implementation. The model is loaded lazily on first use."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en-v1.5",
        *,
        query_prefix: str = _DEFAULT_QUERY_PREFIX,
        doc_prefix: str = "",
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.dim = 768
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._normalize = normalize
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
            dim_fn = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            self.dim = dim_fn()
        return self._model

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._load().encode(
            [self._query_prefix + text], normalize_embeddings=self._normalize
        )
        return np.asarray(vec, dtype=np.float32)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._load().encode(
            [self._doc_prefix + t for t in texts],
            normalize_embeddings=self._normalize,
            batch_size=32,
        )
        return np.asarray(vecs, dtype=np.float32)
