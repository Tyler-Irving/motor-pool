"""Corpus handling: load the manifest and download the TM PDFs.

Lives in the package so the `motor-pool download` console entrypoint can reach
it from an installed wheel. The manifest (corpus/manifest.yaml) and the PDFs
(corpus/pdfs/, gitignored) are repo data, not code.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import yaml

from .schemas import CorpusManifest, TmEntry

_USER_AGENT = "motor-pool/0.1 (corpus downloader)"
_CHUNK = 1 << 16


def load_manifest(path: str | Path) -> CorpusManifest:
    """Load and validate corpus/manifest.yaml."""
    return CorpusManifest.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


def local_path(tm: TmEntry, out_dir: Path) -> Path:
    """The on-disk path a TM downloads to (filename taken from its URL)."""
    return out_dir / Path(tm.url).name


def _download_one(url: str, dest: Path, expected: str | None) -> str:
    """Download to a temp file, verify the digest, then move into place.

    The download is all-or-nothing: a network failure or a hash mismatch leaves
    no file at `dest`, so a later run cannot mistake a truncated file for the
    corpus. Returns the sha256 hex digest.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    digest = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=180) as response, open(tmp, "wb") as f:
            while True:
                block = response.read(_CHUNK)
                if not block:
                    break
                f.write(block)
                digest.update(block)
        got = digest.hexdigest()
        if expected and got != expected:
            raise ValueError(f"sha256 mismatch for {url}: expected {expected}, got {got}")
        os.replace(tmp, dest)
        return got
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def download_corpus(
    manifest_path: Path,
    out_dir: Path,
    *,
    only: set[str] | None = None,
) -> CorpusManifest:
    """Download the manifest's TMs into out_dir and verify or report sha256.

    `only` restricts the download to the given tm_number values. When a manifest
    entry pins a sha256 the download is verified against it; otherwise the
    computed digest is returned in the result so it can be pinned into the
    manifest.
    """
    manifest = load_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    updated: list[TmEntry] = []
    for tm in manifest.tms:
        if only is not None and tm.tm_number not in only:
            updated.append(tm)
            continue
        dest = out_dir / Path(tm.url).name
        digest = _download_one(tm.url, dest, tm.sha256 or None)
        updated.append(tm.model_copy(update={"sha256": digest}))
    return manifest.model_copy(update={"tms": updated})
