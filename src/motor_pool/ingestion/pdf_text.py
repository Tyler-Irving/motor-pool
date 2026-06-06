"""Born-digital PDF text extraction and printed-page-label detection.

The anchor -280 TMs are born-digital (selectable text), so extraction is the
primary path and OCR is a wired fallback. The PDF carries no internal page
labels, so the printed label (e.g. "2-72", "F-15") is recovered from page
geometry: it is the bottom-most label-shaped token, preferring one whose chapter
prefix matches the page's content (a body cross-reference to another chapter is
never the page's own number).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz

# A printed page label: chapter-or-appendix prefix, hyphen, number, optional
# change-page decimal. Examples: "1-1", "2-104.2", "F-15", "G-14".
PAGE_LABEL_RE = re.compile(r"^([1-9]\d{0,2}|[A-P])-\d+(?:\.\d+)?$")

# A token that can be part of a (possibly glyph-split) page label.
_GLYPH_RE = re.compile(r"^[0-9A-P.\-]+$")

# Page numbers sit in the lower portion of the page, below the body.
_LABEL_MIN_Y = 400.0
# Glyphs within one number are tightly packed; a larger horizontal gap means a
# separate field (e.g. the "Change 1" marker next to the page number).
_LABEL_GAP = 8.0


@dataclass(frozen=True)
class PdfPage:
    """One physical PDF page: its text and the page-label candidates found on it.

    label_candidates is a list of (y, token) for every label-shaped token below
    _LABEL_MIN_Y, sorted by y ascending (so the last is bottom-most).
    """

    index: int
    text: str
    label_candidates: tuple[tuple[float, str], ...]

    @property
    def page_label(self) -> str | None:
        """The bottom-most label-shaped token, chapter-agnostic."""
        return self.label_candidates[-1][1] if self.label_candidates else None

    def label_for_chapter(self, chapter: str | None) -> str | None:
        """The bottom-most label whose chapter prefix matches `chapter`.

        Falls back to the chapter-agnostic bottom-most label when none matches.
        """
        if chapter is not None:
            matched = [t for t in self.label_candidates if t[1].split("-")[0] == chapter]
            if matched:
                return matched[-1][1]
        return self.page_label


def _labels_from_words(
    words: list[tuple[float, float, float, float, str]],
) -> tuple[tuple[float, str], ...]:
    """Find page-label candidates, reconstructing labels split into glyph tokens.

    Some revision pages render the footer number as individual character tokens
    ("3", "-", "1", "3" instead of "3-13"). Words in the footer band are grouped
    into rows; consecutive label-alphabet tokens in a row are joined before being
    matched, so both whole-token and glyph-split footers are recovered.
    """
    rows: dict[int, list[tuple[float, float, str]]] = {}
    for x0, y0, x1, _y1, word in words:
        if y0 > _LABEL_MIN_Y:
            rows.setdefault(round(y0 / 4.0), []).append((x0, x1, word))
    cands: list[tuple[float, str]] = []
    for ykey, row in rows.items():
        row.sort(key=lambda t: t[0])
        run: list[str] = []
        prev_x1: float | None = None
        for x0, x1, word in row:
            glyph = bool(word) and bool(_GLYPH_RE.match(word))
            gap = prev_x1 is not None and (x0 - prev_x1) > _LABEL_GAP
            if run and (not glyph or gap):  # a word or a horizontal gap ends the run
                joined = "".join(run)
                if PAGE_LABEL_RE.match(joined):
                    cands.append((ykey * 4.0, joined))
                run = []
            if glyph:
                run.append(word)
                prev_x1 = x1
            else:
                prev_x1 = None
        if run and PAGE_LABEL_RE.match("".join(run)):
            cands.append((ykey * 4.0, "".join(run)))
    cands.sort(key=lambda t: t[0])
    return tuple(cands)


def _label_candidates(page: fitz.Page) -> tuple[tuple[float, str], ...]:
    words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
    return _labels_from_words(words)


def extract_pages(pdf_path: str | Path) -> list[PdfPage]:
    """Return per-page text and page-label candidates, index-aligned to the PDF."""
    doc = fitz.open(pdf_path)
    try:
        return [
            PdfPage(index=i, text=doc[i].get_text(), label_candidates=_label_candidates(doc[i]))
            for i in range(doc.page_count)
        ]
    finally:
        doc.close()


def is_scanned(page_text: str, min_chars: int = 10) -> bool:
    """True if a page has too little extractable text and would need OCR."""
    return len(page_text.strip()) < min_chars


def printable_ratio(text: str) -> float:
    """Fraction of characters that are normal printable text.

    The alphabetical index pages of some TMs use an embedded font with no
    Unicode map and extract as control-character garbage; those score low here.
    Such pages are not scanned (OCR would not help), so they are reported and
    excluded by content range rather than sent to OCR.
    """
    if not text:
        return 0.0
    good = sum(1 for c in text if c.isprintable() or c in "\n\t ")
    return good / len(text)
