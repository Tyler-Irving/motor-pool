"""Phase 2 gate: the embedder applies the asymmetric query/doc prefixes.

Stubs the model so the test runs without sentence-transformers / torch.
"""

from __future__ import annotations

import numpy as np

from motor_pool.retrieval.embedder_bge import BgeEmbedder


class _FakeModel:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        self.seen = list(texts)
        return np.ones((len(texts), 3), dtype=np.float32)


def test_query_and_doc_prefixes_are_asymmetric(monkeypatch) -> None:
    fake = _FakeModel()
    embedder = BgeEmbedder("stub", query_prefix="Q: ", doc_prefix="D: ")
    monkeypatch.setattr(embedder, "_load", lambda: fake)

    embedder.embed_query("hello")
    assert fake.seen == ["Q: hello"]

    embedder.embed_documents(["a", "b"])
    assert fake.seen == ["D: a", "D: b"]


def test_embed_query_returns_1d_float32(monkeypatch) -> None:
    embedder = BgeEmbedder("stub")
    monkeypatch.setattr(embedder, "_load", lambda: _FakeModel())
    vec = embedder.embed_query("x")
    assert vec.shape == (3,)
    assert vec.dtype == np.float32
