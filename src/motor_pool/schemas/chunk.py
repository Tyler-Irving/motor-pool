"""Chunk schemas: the unit of ingestion, indexing, and retrieval.

A Chunk is one procedure-level unit of a TM (primarily a numbered paragraph,
or a table / warning / troubleshooting block) plus its Citation. A
RetrievedChunk is what the Retriever returns: a Chunk projection plus a fused
relevance score.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .citation import Citation

ContentType = Literal[
    "procedure",
    "table",
    "warning_block",
    "troubleshooting",
    "narrative",
]


class Chunk(BaseModel):
    """One indexed unit of a TM. Written one-per-row to indexes/chunks.jsonl."""

    chunk_id: str
    text: str
    content_type: ContentType
    citation: Citation

    parent_id: str | None = Field(
        default=None, description="Parent Section chunk id, for V2 procedure fetch."
    )
    section_path: str | None = Field(
        default=None, description='Breadcrumb, e.g. "Ch2 > Sec IV > 2-104.1".'
    )
    test_reference: str | None = Field(
        default=None,
        description='STE/ICE test reference, e.g. "TEST #89". The V2 DTC analog.',
    )
    goto_targets: list[str] = Field(
        default_factory=list,
        description="Troubleshooting STEP/page jump targets, for the V2 fault tree.",
    )


class RetrievedChunk(BaseModel):
    """A chunk returned by the Retriever, carrying its fused relevance score.

    Carries the V2 seam fields (parent_id, section_path, test_reference,
    goto_targets) so the future agent can reach procedure fetch, the STE/ICE
    lookup, and the fault tree directly from a retrieval hit.
    """

    chunk_id: str
    text: str
    score: float = Field(description="Fused RRF score. Higher is more relevant.")
    citation: Citation
    content_type: ContentType
    parent_id: str | None = None
    section_path: str | None = None
    test_reference: str | None = None
    goto_targets: list[str] = Field(default_factory=list)
