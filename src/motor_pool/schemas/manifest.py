"""Manifest schemas: the corpus download manifest and the build-time index manifest.

CorpusManifest is committed (corpus/manifest.yaml); it describes the public TMs
to download and pins each by sha256. IndexManifest is a gitignored build
artifact written next to the dense index; it records exactly what produced the
index so a run is reproducible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TmEntry(BaseModel):
    """One technical manual in the corpus manifest. No binary is committed."""

    tm_number: str = Field(description='e.g. "TM 9-2320-280-10".')
    title: str
    url: str = Field(description="Public download URL. Provenance, may rot.")
    sha256: str = Field(
        default="",
        description="Expected hash of the PDF. Empty until pinned at download time.",
    )
    page_count: int | None = None
    edition_date: str | None = None
    distribution_statement: str = "Distribution Statement A"


class CorpusManifest(BaseModel):
    """The set of TMs that make up the corpus."""

    tms: list[TmEntry]


class IndexManifest(BaseModel):
    """Records what produced an index build, for reproducibility."""

    embedder_id: str
    dims: int
    normalize: bool
    chunk_count: int
    corpus_sha256s: list[str] = Field(
        default_factory=list, description="sha256 of each source PDF in the build."
    )
    bm25_backend: str = "bm25s"
    built_at: str = Field(description="Build timestamp, set by the caller.")
