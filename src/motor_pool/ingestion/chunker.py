"""Procedure-level chunking with citation metadata.

The unit is the numbered paragraph. Each paragraph becomes one Chunk (or a few
size-bounded parts), carrying a Citation whose locator is the raw paragraph key
and whose page is the printed label of the page the paragraph starts on. Inline
WARNING text stays with the procedure it guards. Continuation headers ("(Cont'd)"
with a repeated number) merge back into the original paragraph. Front-matter and
appendix material are excluded; appendix and troubleshooting handling come later.
"""

from __future__ import annotations

import re

from motor_pool.schemas import Chunk, Citation, ContentType, ParagraphLocator

from .pdf_text import PdfPage
from .structure import (
    CHAPTER_RE,
    PARAGRAPH_NUM_RE,
    PARAGRAPH_RE,
    SECTION_RE,
    clean_lines,
    is_toc_line,
)

# Top-level subitem starts, used as preferred split boundaries for long paragraphs.
_SUBITEM_RE = re.compile(r"^(?:[a-z]\.|\(\d+\)|\([a-z]\))\s")


def _slug(doc_id: str) -> str:
    return re.sub(r"\s+", "-", doc_id.strip())


def _content_type(title: str) -> ContentType:
    return "table" if "TABULATED DATA" in title.upper() else "procedure"


def _split_segments(lines: list[str], max_words: int) -> list[list[str]]:
    """Split body lines into segments under max_words, breaking at subitems."""
    segments: list[list[str]] = []
    current: list[str] = []
    count = 0
    for line in lines:
        words = len(line.split())
        if current and count + words > max_words and (_SUBITEM_RE.match(line) or count >= max_words):
            segments.append(current)
            current, count = [], 0
        current.append(line)
        count += words
    if current:
        segments.append(current)
    return segments or [[]]


_INT_LABEL_RE = re.compile(r"^([0-9A-P]+)-(\d+)$")


def _increment_label(label: str, steps: int) -> str:
    """Step a non-decimal page label by `steps` (e.g. "1-15", 1 -> "1-16")."""
    match = _INT_LABEL_RE.match(label)
    if not match or steps == 0:
        return label
    return f"{match.group(1)}-{int(match.group(2)) + steps}"


def _page_labels(pages: list[PdfPage]) -> list[str | None]:
    """Printed page label per page, interpolating across footer-less pages.

    A full-page illustration may have no extractable footer at all. Reusing the
    previous page's label verbatim cites the paragraph to the wrong page, so a
    page with no detected label is interpolated as the last detected label
    stepped by the page-index offset (printed pages run +1 with PDF pages).
    """
    labels: list[str | None] = []
    last_label: str | None = None
    last_index = 0
    for page in pages:
        detected = page.page_label
        if detected:
            last_label, last_index = detected, page.index
            labels.append(detected)
        elif last_label is not None:
            labels.append(_increment_label(last_label, page.index - last_index))
        else:
            labels.append(None)
    return labels


def chunk_document(
    pages: list[PdfPage],
    *,
    source_doc_id: str,
    source_doc_title: str,
    edition_date: str,
    source_pdf_sha256: str,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Parse procedure-level Chunks from extracted pages.

    `source_doc_id` is the TM number as printed (also the running header text),
    e.g. "TM 9-2320-280-10". It is slugified for chunk ids and the citation.
    """
    slug = _slug(source_doc_id)
    max_words = max(1, int(max_tokens * 0.75))

    page_labels = _page_labels(pages)
    spans: list[dict] = []
    by_number: dict[str, dict] = {}
    current: dict | None = None
    awaiting_title: dict | None = None  # span whose title is on the next line
    section: str | None = None
    chapter: str | None = None
    in_appendix = False

    for page in pages:
        # Appendix pages are lettered (A-1, G-14). Once numbered content has
        # started, a lettered bottom-most page label marks the appendix boundary.
        # Use the chapter-agnostic bottom-most label: an appendix page carries
        # stray chapter cross-references, so a chapter-biased pick would miss it.
        bottom = page.page_label
        if spans and bottom and bottom.split("-")[0].isalpha():
            in_appendix, current = True, None
        for line in clean_lines(page.text, header_text=source_doc_id):
            if CHAPTER_RE.match(line):
                current, section, awaiting_title = None, None, None
                continue
            section_match = SECTION_RE.match(line)
            if section_match:
                section, current, awaiting_title = section_match.group(1).upper(), None, None
                continue
            para_match = PARAGRAPH_RE.match(line)
            num_only = None if para_match else PARAGRAPH_NUM_RE.match(line)
            header = para_match or num_only
            if header and not in_appendix and not is_toc_line(line):
                number = header.group(1)
                if not number.split("-")[0].isdigit():
                    continue
                if number in by_number:  # a "(Cont'd)" continuation
                    current, awaiting_title = by_number[number], None
                    continue
                chapter = number.split("-")[0]
                current = {
                    "number": number,
                    "title": para_match.group(2).strip() if para_match else "",
                    "chapter": chapter,
                    "section": section,
                    "pdf_index": page.index,
                    "page_label": page_labels[page.index] or "",
                    "body": [],
                }
                by_number[number] = current
                spans.append(current)
                # For a split header, the next line is the title.
                awaiting_title = current if num_only else None
                continue
            if awaiting_title is not None:
                awaiting_title["title"] = line
                awaiting_title = None
                continue
            if current is not None and not in_appendix and not is_toc_line(line):
                current["body"].append(line)

    return _spans_to_chunks(
        spans,
        slug=slug,
        source_doc_title=source_doc_title,
        edition_date=edition_date,
        source_pdf_sha256=source_pdf_sha256,
        max_words=max_words,
    )


def _spans_to_chunks(
    spans: list[dict],
    *,
    slug: str,
    source_doc_title: str,
    edition_date: str,
    source_pdf_sha256: str,
    max_words: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for span in spans:
        number = span["number"]
        title = span["title"]
        header = f"{number}. {title}"
        parent_id = f"{slug}#ch{span['chapter']}#sec{span['section'] or 'NA'}"
        path = f"Ch {span['chapter']}"
        if span["section"]:
            path += f" > Sec {span['section']}"
        path += f" > {number}"

        segments = _split_segments(span["body"], max_words)
        multi = len(segments) > 1
        for part, segment in enumerate(segments, start=1):
            chunk_id = f"{slug}:{number}" + (f"#{part}" if multi else "")
            text = header if not segment else f"{header}\n{' '.join(segment)}"
            citation = Citation(
                source_doc_id=slug,
                source_doc_title=source_doc_title,
                edition_date=edition_date,
                locator=ParagraphLocator(
                    chapter=span["chapter"], section=span["section"], paragraph=number
                ),
                tm_page_label=span["page_label"] or "",
                pdf_page_index=span["pdf_index"],
                source_pdf_sha256=source_pdf_sha256,
                chunk_id=chunk_id,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    content_type=_content_type(title),
                    citation=citation,
                    parent_id=parent_id,
                    section_path=path,
                )
            )
    return chunks
