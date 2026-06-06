"""OCR fallback for scanned pages.

Wired for completeness and robustness. Expected to be a no-op for the -280 TMs,
which are born-digital (verified at corpus selection time).
"""

from __future__ import annotations

from pathlib import Path


def ocr_page(pdf_path: Path, page_index: int) -> str:
    """OCR a single physical page and return its text."""
    raise NotImplementedError("Phase 1: OCR fallback (ocrmypdf/pytesseract).")
