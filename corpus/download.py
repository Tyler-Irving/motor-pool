"""Runnable wrapper to download the corpus.

The implementation lives in the package at motor_pool.corpus. Prefer
`motor-pool download`. This script is for running directly from a checkout after
`uv sync`:

    python corpus/download.py
"""

from __future__ import annotations

from pathlib import Path

from motor_pool.corpus import download_corpus

if __name__ == "__main__":
    download_corpus(Path("corpus/manifest.yaml"), Path("corpus/pdfs"))
