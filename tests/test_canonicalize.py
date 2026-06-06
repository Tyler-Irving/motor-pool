"""Phase 0 gate: citation canonicalization ignores label differences.

Critical: this logic is shared by the data-gen validator and the eval scorer, so
a bug here corrupts both train-time acceptance and eval scoring together. The
appendix cases below guard the high-severity over-stripping class of bug.
"""

from __future__ import annotations

from motor_pool.data_gen.canonicalize import (
    canonicalize_citation,
    normalize_doc_id,
    normalize_page,
    normalize_token,
)
from motor_pool.schemas import Citation, ParagraphLocator


def test_label_prefix_is_ignored() -> None:
    assert normalize_token("Para 2-14") == "2-14"
    assert normalize_token("Section 2-14") == "2-14"
    assert normalize_token("Para 2-14") == normalize_token("Section 2-14")


def test_page_prefix_is_ignored() -> None:
    assert normalize_token("p. 2-72") == "2-72"
    assert normalize_token("page 2-72") == "2-72"


def test_paragraph_decimal_is_preserved() -> None:
    # The change-page decimal must not be stripped.
    assert normalize_token("2-104.1") == "2-104.1"


def test_appendix_identifiers_are_not_overstripped() -> None:
    # Appendix paragraphs (HMMWV -10 runs appendices A..P) must survive.
    assert normalize_token("P-1") == "p-1"
    assert normalize_token("PG-1") == "pg-1"
    assert normalize_token("SEC-1") == "sec-1"
    assert normalize_token("CHP-1") == "chp-1"


def test_bare_appendix_label_is_preserved() -> None:
    # A standalone appendix letter is an identifier, not a label. Do not drop it.
    assert normalize_token("P") == "p"
    assert normalize_token("PG") == "pg"


def test_label_without_separator_is_left_alone() -> None:
    # Conservative: only strip a label when a real separator follows.
    assert normalize_token("para2-14") == "para2-14"


def test_normalize_page_does_not_strip_labels() -> None:
    assert normalize_page("2-72") == "2-72"
    assert normalize_page("P-3") == "p-3"


def test_doc_id_spacing_and_hyphen_equivalence() -> None:
    assert normalize_doc_id("TM 9-2320-280-10") == normalize_doc_id("TM-9-2320-280-10")
    assert normalize_doc_id("tm 9-2320-280-10") == "TM-9-2320-280-10"


def _citation(paragraph: str | None = None, section: str | None = None) -> Citation:
    return Citation(
        source_doc_id="TM 9-2320-280-10",
        source_doc_title="Operator's Manual",
        edition_date="JANUARY 1996",
        locator=ParagraphLocator(chapter="2", section=section, paragraph=paragraph),
        tm_page_label="2-72",
        pdf_page_index=10,
        source_pdf_sha256="x",
        chunk_id="c1",
    )


def test_canonicalize_citation_tuple() -> None:
    canon = canonicalize_citation(_citation(paragraph="2-104.1"))
    assert canon == ("TM-9-2320-280-10", "2-104.1", "2-72")


def test_canonicalize_prefers_paragraph_then_section() -> None:
    assert canonicalize_citation(_citation(section="IV"))[1] == "iv"
