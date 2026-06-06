"""Ingestion pipeline: parse -> chunk (Phase 1), then embed + bm25 (Phase 2).

Phase 1 writes indexes/chunks.jsonl (one Chunk per row). The dense embeddings,
the bm25 index, and the IndexManifest are added in Phase 2.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from motor_pool.config import IngestionConfig
from motor_pool.retrieval.interfaces import Embedder
from motor_pool.schemas import Chunk, IndexManifest, TmEntry

from .chunker import chunk_document
from .pdf_text import extract_pages

_READ_BLOCK = 1 << 16


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    """Load chunks written by write_chunks_jsonl, preserving row order."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_document(pdf_path: Path, tm: TmEntry, config: IngestionConfig) -> list[Chunk]:
    """Parse and chunk one TM PDF into procedure-level Chunks.

    The on-disk file is hashed and checked against the manifest pin (when set),
    and the computed digest is stamped into every citation so provenance reflects
    the bytes actually read, not the manifest value.
    """
    digest = _sha256(pdf_path)
    if tm.sha256 and digest != tm.sha256:
        raise ValueError(
            f"sha256 mismatch for {tm.tm_number}: manifest pins {tm.sha256}, "
            f"on-disk file is {digest}. Re-download before ingesting."
        )
    pages = extract_pages(pdf_path)
    return chunk_document(
        pages,
        source_doc_id=tm.tm_number,
        source_doc_title=tm.title,
        edition_date=tm.edition_date or "",
        source_pdf_sha256=digest,
        max_tokens=config.chunk_max_tokens,
    )


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    """Write chunks one JSON object per line, as UTF-8, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    os.replace(tmp, path)


def ingest_corpus(
    manifest_path: Path,
    pdfs_dir: Path,
    out_path: Path,
    *,
    config: IngestionConfig,
    only: set[str] | None = None,
) -> list[Chunk]:
    """Ingest every downloaded TM in the manifest into one chunks.jsonl.

    Warns about any selected TM whose PDF is not on disk and refuses to write an
    empty index (which would clobber a good one). Raises if `only` names a TM not
    in the manifest.
    """
    from motor_pool.corpus import load_manifest, local_path

    manifest = load_manifest(manifest_path)
    known = {tm.tm_number for tm in manifest.tms}
    if only is not None:
        unknown = only - known
        if unknown:
            raise ValueError(f"--only names not in manifest: {sorted(unknown)}")

    chunks: list[Chunk] = []
    ingested = 0
    for tm in manifest.tms:
        if only is not None and tm.tm_number not in only:
            continue
        pdf_path = local_path(tm, pdfs_dir)
        if not pdf_path.exists():
            print(f"warning: skipping {tm.tm_number}: not downloaded ({pdf_path})", file=sys.stderr)
            continue
        chunks.extend(ingest_document(pdf_path, tm, config))
        ingested += 1

    if ingested == 0:
        raise FileNotFoundError(
            "no TMs ingested (none downloaded or none matched). Run `motor-pool download` first."
        )
    write_chunks_jsonl(chunks, out_path)
    return chunks


def build_indexes(
    chunks: list[Chunk],
    *,
    out_dir: Path,
    embedder: Embedder,
    normalize: bool = True,
    corpus_sha256s: list[str] | None = None,
    built_at: str = "",
) -> IndexManifest:
    """Write the dense embeddings, the bm25 index, and the IndexManifest.

    Embeddings are row-aligned to `chunks` (the same order as chunks.jsonl), so
    the loader can map rows back to chunks without a separate id file.
    """
    from motor_pool.retrieval.bm25_index import Bm25sIndex

    out_dir = Path(out_dir)
    dense_dir = out_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dense_dir / "index_manifest.json"

    # Invalidate the prior manifest first. The manifest is the completion marker
    # (load_retriever requires it), so an interrupted rebuild fails the loader
    # rather than serving an embeddings/bm25 pair that may be half-written.
    manifest_path.unlink(missing_ok=True)

    embeddings = np.asarray(embedder.embed_documents([c.text for c in chunks]), dtype=np.float32)
    tmp_emb = dense_dir / "embeddings.tmp.npy"
    np.save(tmp_emb, embeddings)
    os.replace(tmp_emb, dense_dir / "embeddings.npy")
    Bm25sIndex.build(chunks).save(out_dir / "bm25")

    manifest = IndexManifest(
        embedder_id=embedder.model_id,
        dims=int(embeddings.shape[1]),
        normalize=normalize,
        chunk_count=len(chunks),
        corpus_sha256s=corpus_sha256s or [],
        built_at=built_at,
    )
    # Written last: its presence signals a complete build.
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
