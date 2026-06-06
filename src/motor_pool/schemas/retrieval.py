"""Retrieval request-shaping schemas: filters and RRF configuration.

These are deliberately small. The Retriever tool surface is a plain
`retrieve(query, *, top_k, filters) -> list[RetrievedChunk]`; there is no
request/response envelope in V1 (that is a V2 serialization concern).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .chunk import ContentType


class RetrievalFilters(BaseModel):
    """Optional metadata filters. When present, the retriever fans out over the
    full corpus before filtering, so the filter is exhaustive (no starvation)."""

    model_config = ConfigDict(extra="forbid")

    source_doc_id: str | None = None
    content_type: ContentType | None = None


class RrfConfig(BaseModel):
    """Reciprocal rank fusion configuration.

    k=60 is the standard default (Cormack et al.). Weights stay 1.0/1.0 in V1;
    they are exposed so the eval harness can tune them later, not hand-tuned now.
    """

    model_config = ConfigDict(extra="forbid")

    k: int = 60
    top_n: int = Field(default=50, description="Per-retriever fan-out before fusion.")
    top_k: int = Field(
        default=8,
        description="Chunks returned when retrieve() is called without an explicit "
        "top_k. A retrieve(top_k=...) argument overrides it.",
    )
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
