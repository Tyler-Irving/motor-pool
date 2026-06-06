"""Phase 2 gate: HybridRetriever conforms to the Retriever contract.

Uses stub backends so the contract is exercised without the retrieve extra
(bm25s / sentence-transformers / torch). The real end-to-end retrieval over the
ingested corpus is validated separately via `motor-pool query`.
"""

from __future__ import annotations

import numpy as np

from motor_pool.retrieval import ProcedureFetcher, Retriever
from motor_pool.retrieval.hybrid_retriever import HybridRetriever
from motor_pool.schemas import (
    Citation,
    Chunk,
    ParagraphLocator,
    RetrievalFilters,
    RrfConfig,
)


def _chunk(num: str, content_type: str = "procedure", parent: str = "sec") -> Chunk:
    return Chunk(
        chunk_id=f"TM:{num}",
        text=f"{num}. body about widgets",
        content_type=content_type,
        parent_id=parent,
        citation=Citation(
            source_doc_id="TM-9-2320-280-10",
            source_doc_title="T",
            edition_date="1996",
            locator=ParagraphLocator(chapter=num.split("-")[0], paragraph=num),
            tm_page_label=num,
            pdf_page_index=int(num.split("-")[1]),
            source_pdf_sha256="x",
            chunk_id=f"TM:{num}",
        ),
    )


CHUNKS = [_chunk("2-1"), _chunk("2-2", content_type="table"), _chunk("2-3")]


class _Embedder:
    model_id = "stub"
    dim = 3

    def embed_query(self, text: str) -> np.ndarray:
        return np.ones(3, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class _Bm25:
    def __init__(self, ranked):
        self._ranked = ranked

    def search(self, query: str, top_n: int):
        return self._ranked[:top_n]


class _Vectors:
    def __init__(self, ranked):
        self._ranked = ranked

    def search(self, query_vec, top_n: int):
        return self._ranked[:top_n]

    def get_chunk(self, chunk_id: str):
        return {c.chunk_id: c for c in CHUNKS}[chunk_id]


def _retriever() -> HybridRetriever:
    bm25 = _Bm25([("TM:2-1", 5.0), ("TM:2-2", 3.0), ("TM:2-3", 1.0)])
    vectors = _Vectors([("TM:2-3", 0.9), ("TM:2-1", 0.8), ("TM:2-2", 0.1)])
    return HybridRetriever(_Embedder(), bm25, vectors, CHUNKS, RrfConfig(top_n=10))


def test_satisfies_protocols() -> None:
    r = _retriever()
    assert isinstance(r, Retriever)
    assert isinstance(r, ProcedureFetcher)


def test_returns_retrieved_chunks_fused() -> None:
    results = _retriever().retrieve("widgets", top_k=3)
    assert [r.chunk_id for r in results] == ["TM:2-1", "TM:2-3", "TM:2-2"]
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))
    assert results[0].citation.locator.paragraph == "2-1"


def test_top_k_truncates() -> None:
    assert len(_retriever().retrieve("widgets", top_k=1)) == 1


def test_content_type_filter() -> None:
    results = _retriever().retrieve("widgets", top_k=5, filters=RetrievalFilters(content_type="table"))
    assert [r.chunk_id for r in results] == ["TM:2-2"]


def test_source_doc_filter_excludes_everything_when_unknown() -> None:
    results = _retriever().retrieve(
        "widgets", top_k=5, filters=RetrievalFilters(source_doc_id="TM-OTHER")
    )
    assert results == []


def test_default_top_k_comes_from_config() -> None:
    bm25 = _Bm25([("TM:2-1", 5.0), ("TM:2-2", 3.0), ("TM:2-3", 1.0)])
    vectors = _Vectors([("TM:2-1", 0.9), ("TM:2-2", 0.8), ("TM:2-3", 0.1)])
    r = HybridRetriever(_Embedder(), bm25, vectors, CHUNKS, RrfConfig(top_n=10, top_k=2))
    assert len(r.retrieve("widgets")) == 2  # no explicit top_k -> cfg.top_k


def test_empty_query_returns_empty() -> None:
    assert _retriever().retrieve("   ") == []


def test_filter_not_starved_by_top_n() -> None:
    # The lone table chunk ranks last in both backends and below top_n=1, yet a
    # content_type filter must still find it (fan-out over the full corpus).
    bm25 = _Bm25([("TM:2-1", 5.0), ("TM:2-3", 1.0), ("TM:2-2", 0.5)])
    vectors = _Vectors([("TM:2-1", 0.9), ("TM:2-3", 0.8), ("TM:2-2", 0.1)])
    r = HybridRetriever(_Embedder(), bm25, vectors, CHUNKS, RrfConfig(top_n=1))
    results = r.retrieve("widgets", filters=RetrievalFilters(content_type="table"))
    assert [x.chunk_id for x in results] == ["TM:2-2"]


def test_projects_v2_seam_fields() -> None:
    chunk = _chunk("2-9").model_copy(
        update={
            "test_reference": "TEST #89",
            "goto_targets": ["2-72"],
            "section_path": "Ch2 > 2-9",
        }
    )
    r = HybridRetriever(
        _Embedder(),
        _Bm25([("TM:2-9", 1.0)]),
        _Vectors([("TM:2-9", 0.9)]),
        [chunk],
        RrfConfig(top_n=5),
    )
    result = r.retrieve("widgets", top_k=1)[0]
    assert result.test_reference == "TEST #89"
    assert result.goto_targets == ["2-72"]
    assert result.section_path == "Ch2 > 2-9"


def test_get_procedure_returns_section_siblings() -> None:
    procedure = _retriever().get_procedure("TM-9-2320-280-10", "sec")
    assert procedure is not None
    assert [c.citation.locator.paragraph for c in procedure] == ["2-1", "2-2", "2-3"]
    assert _retriever().get_procedure("TM-9-2320-280-10", "nope") is None
