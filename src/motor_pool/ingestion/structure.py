"""Structural parsing of TM text: headings, paragraph headers, line cleaning.

These TMs use chapter / section / paragraph numbering. The procedure-level unit
is the numbered paragraph header `N-NN. TITLE`. Section headings (`Section IV.`)
group paragraphs. Running headers, page numbers, change markers, and bare
illustration callout digits are noise and are stripped before chunking.
"""

from __future__ import annotations

from .pdf_text import PAGE_LABEL_RE, PdfPage

import re

SECTION_RE = re.compile(r"^Section\s+([IVXL]+)\b\.?\s*(.*)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+(\d+)\b\.?\s*(.*)$")
APPENDIX_RE = re.compile(r"^APPENDIX\s+([A-Z])\b", re.IGNORECASE)
# Numbered paragraph header, number and title on one line:
# "2-27. FIRE EXTINGUISHER OPERATION".
PARAGRAPH_RE = re.compile(r"^(\d+-\d+(?:\.\d+)?)\.\s+(\S.*)$")
# Split header: the number sits alone on its line ("1-13.") with the title on the
# following line(s). Common in the equipment-description paragraphs.
PARAGRAPH_NUM_RE = re.compile(r"^(\d+-\d+(?:\.\d+)?)\.$")
# Table-of-contents / list-of-illustrations rows use dot leaders.
_DOTLEADER_RE = re.compile(r"\.\s*\.\s*\.")
_CHANGE_RE = re.compile(r"^Change\s+\d+$", re.IGNORECASE)
_CALLOUT_RE = re.compile(r"^\d{1,2}$")  # bare illustration callout digit


def is_toc_line(line: str) -> bool:
    """True for table-of-contents / list rows (dot leaders), which are not content."""
    return bool(_DOTLEADER_RE.search(line))


def clean_lines(text: str, *, header_text: str) -> list[str]:
    """Strip running header, page numbers, change markers, and callout digits."""
    out: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s == header_text or _CHANGE_RE.match(s) or _CALLOUT_RE.match(s):
            continue
        if PAGE_LABEL_RE.match(s):  # a standalone page-number line
            continue
        out.append(s)
    return out


def detect_headings(pages: list[PdfPage]) -> list[tuple[int, str]]:
    """Return (pdf_page_index, heading_text) for chapter and section headings."""
    out: list[tuple[int, str]] = []
    for page in pages:
        for raw in page.text.splitlines():
            s = raw.strip()
            if is_toc_line(s):
                continue
            if CHAPTER_RE.match(s) or SECTION_RE.match(s):
                out.append((page.index, s))
    return out
