"""Phase 1 gate: chunker and page-label logic, on a synthetic TM page (no real PDF)."""

from __future__ import annotations

import pytest

from motor_pool.ingestion.chunker import _page_labels, chunk_document
from motor_pool.ingestion.pdf_text import PdfPage, _labels_from_words
from motor_pool.schemas import Chunk

HEADER = "TM 9-2320-280-99"

# A synthetic content page exercising: a single-line header, an inline WARNING,
# a split header (number alone, title on the next line), a TABULATED DATA table,
# and a decimal change-page paragraph. The trailing "2-1" is the printed page label.
PAGE_BODY = "\n".join(
    [
        HEADER,
        "Section I.  OPERATION",
        "2-1. KNOW YOUR CONTROLS",
        "a. Pull the lever (1).",
        "WARNING",
        "Do not touch hot exhaust parts.",
        "b. Release the lever.",
        "2-2.",
        "SECONDARY SYSTEM",
        "Operate the secondary system.",
        "2-3. TABULATED DATA",
        "Engine: V8 diesel.",
        "2-3.1.",
        "ADDED DATA",
        "Extra spec row.",
        "2-1",
    ]
)
# An appendix page (lettered label) that must be excluded.
APPENDIX_BODY = "\n".join([HEADER, "A-1. REFERENCES", "FM 1-2 applies.", "A-1"])


def _page(index: int, text: str, label: str) -> PdfPage:
    return PdfPage(index=index, text=text, label_candidates=((720.0, label),))


def _chunks() -> list[Chunk]:
    pages = [_page(0, PAGE_BODY, "2-1"), _page(1, APPENDIX_BODY, "A-1")]
    return chunk_document(
        pages,
        source_doc_id=HEADER,
        source_doc_title="Test Manual",
        edition_date="2020",
        source_pdf_sha256="abc",
        max_tokens=512,
    )


def test_paragraph_keys_in_order_appendix_excluded() -> None:
    nums = [c.citation.locator.paragraph for c in _chunks()]
    assert nums == ["2-1", "2-2", "2-3", "2-3.1"]


def test_warning_is_kept_with_its_procedure() -> None:
    c = next(c for c in _chunks() if c.citation.locator.paragraph == "2-1")
    assert "WARNING" in c.text
    assert "hot exhaust" in c.text


def test_split_header_title_captured() -> None:
    c = next(c for c in _chunks() if c.citation.locator.paragraph == "2-2")
    assert c.text.startswith("2-2. SECONDARY SYSTEM")


def test_raw_decimal_key_preserved() -> None:
    assert any(c.citation.locator.paragraph == "2-3.1" for c in _chunks())


def test_tabulated_data_typed_as_table() -> None:
    c = next(c for c in _chunks() if c.citation.locator.paragraph == "2-3")
    assert c.content_type == "table"


def test_citation_fields() -> None:
    c = _chunks()[0]
    assert c.chunk_id == "TM-9-2320-280-99:2-1"
    assert c.citation.source_doc_id == "TM-9-2320-280-99"
    assert c.citation.tm_page_label == "2-1"
    assert c.citation.locator.section == "I"
    assert c.citation.pdf_page_index == 0


def test_all_outputs_are_valid_chunks() -> None:
    for c in _chunks():
        assert isinstance(c, Chunk)
        Chunk.model_validate(c.model_dump())


def test_page_label_prefers_matching_chapter() -> None:
    # A body cross-reference to an appendix (A-1) must not be read as the page number.
    page = PdfPage(index=0, text="", label_candidates=((500.0, "2-2"), (700.0, "A-1")))
    assert page.label_for_chapter("2") == "2-2"
    assert page.page_label == "A-1"  # bottom-most, chapter-agnostic


def _word(x0: float, y0: float, x1: float, text: str):
    return (x0, y0, x1, y0 + 12.0, text)


def test_reconstructs_glyph_split_footer() -> None:
    # The "Change 1" revision pages render the page number as separate glyphs.
    words = [
        _word(50, 720, 70, "Change"),
        _word(72, 720, 78, "1"),
        _word(300, 720, 306, "3"),
        _word(306, 720, 309, "-"),
        _word(309, 720, 315, "1"),
        _word(315, 720, 321, "0"),
    ]
    assert [t[1] for t in _labels_from_words(words)] == ["3-10"]


def test_change_marker_glyph_not_merged_into_page_number() -> None:
    # Change marker BEFORE the page number must not produce "13-21".
    words = [
        _word(50, 720, 70, "Change"),
        _word(72, 720, 78, "1"),
        _word(300, 720, 306, "3"),
        _word(306, 720, 309, "-"),
        _word(309, 720, 315, "2"),
        _word(315, 720, 321, "1"),
    ]
    assert [t[1] for t in _labels_from_words(words)] == ["3-21"]


def test_whole_token_footer_and_header_handling() -> None:
    words = [_word(100, 30, 200, "TM"), _word(300, 720, 330, "2-136")]
    assert [t[1] for t in _labels_from_words(words)] == ["2-136"]


def test_footerless_page_is_interpolated_not_carried() -> None:
    def page(index: int, label: str | None) -> PdfPage:
        cands = ((720.0, label),) if label else ()
        return PdfPage(index=index, text="", label_candidates=cands)

    pages = [page(0, "1-15"), page(1, None), page(2, "1-17")]
    assert _page_labels(pages) == ["1-15", "1-16", "1-17"]


def test_appendix_cross_reference_does_not_trigger_boundary() -> None:
    # A chapter page with a numeric footer but an appendix cross-ref in the body
    # must still capture its paragraph (bottom-most label is numeric).
    page = PdfPage(
        index=0,
        text="\n".join([HEADER, "3-1. ENGINE", "See Appendix D.", "3-22"]),
        label_candidates=((500.0, "A-5"), (720.0, "3-22")),
    )
    chunks = chunk_document(
        [page],
        source_doc_id=HEADER,
        source_doc_title="T",
        edition_date="2020",
        source_pdf_sha256="x",
        max_tokens=512,
    )
    assert [c.citation.locator.paragraph for c in chunks] == ["3-1"]
    assert chunks[0].citation.tm_page_label == "3-22"


def test_ingest_corpus_rejects_unknown_only(tmp_path) -> None:
    from motor_pool.config import IngestionConfig
    from motor_pool.ingestion.pipeline import ingest_corpus

    mf = tmp_path / "manifest.yaml"
    mf.write_text(
        "tms:\n  - tm_number: 'TM A'\n    title: 'T'\n    url: 'http://e/a.pdf'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        ingest_corpus(mf, tmp_path, tmp_path / "out.jsonl", config=IngestionConfig(), only={"NOPE"})


def test_ingest_corpus_raises_when_nothing_downloaded(tmp_path) -> None:
    from motor_pool.config import IngestionConfig
    from motor_pool.ingestion.pipeline import ingest_corpus

    mf = tmp_path / "manifest.yaml"
    mf.write_text(
        "tms:\n  - tm_number: 'TM A'\n    title: 'T'\n    url: 'http://e/a.pdf'\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    with pytest.raises(FileNotFoundError):
        ingest_corpus(mf, tmp_path, out, config=IngestionConfig())
    assert not out.exists()  # must not write an empty index
