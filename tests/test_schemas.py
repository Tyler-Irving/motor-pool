"""Phase 0 gate: schema round-trips and the discriminated-union contract."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from motor_pool.schemas import (
    Citation,
    GroundedAnswer,
    ParagraphLocator,
    ProcedureStep,
    Refusal,
    TrainingTarget,
)


def make_citation() -> Citation:
    return Citation(
        source_doc_id="TM-9-2320-280-10",
        source_doc_title="Operator's Manual, HMMWV M998 series",
        edition_date="JANUARY 1996",
        locator=ParagraphLocator(chapter="2", section="IV", paragraph="2-104.1"),
        tm_page_label="2-72",
        pdf_page_index=120,
        source_pdf_sha256="deadbeef",
        chunk_id="c1",
    )


def test_citation_roundtrip() -> None:
    c = make_citation()
    assert Citation.model_validate(c.model_dump()) == c


def test_distribution_statement_default() -> None:
    assert make_citation().distribution_statement == "Distribution Statement A"


def test_paragraph_kept_as_raw_string() -> None:
    # Decimal change-page insertions must survive untouched.
    assert make_citation().locator.paragraph == "2-104.1"


def test_locator_default_type() -> None:
    assert ParagraphLocator(chapter="1").type == "paragraph"


def test_training_target_resolves_to_answer() -> None:
    obj = {
        "answer_type": "answer",
        "summary": "Set the parking brake before towing.",
        "citations": [make_citation().model_dump()],
    }
    target = TypeAdapter(TrainingTarget).validate_python(obj)
    assert isinstance(target, GroundedAnswer)


def test_training_target_resolves_to_refusal() -> None:
    obj = {
        "answer_type": "refusal",
        "reason": "not_in_corpus",
        "message": "The operator manual does not cover depot-level rebuilds.",
    }
    target = TypeAdapter(TrainingTarget).validate_python(obj)
    assert isinstance(target, Refusal)


def test_refusal_rejects_unknown_reason() -> None:
    with pytest.raises(ValidationError):
        Refusal(reason="because", message="no")


def test_procedure_step_requires_citations_field() -> None:
    step = ProcedureStep(order=1, text="Chock the wheels.", citations=[make_citation()])
    assert step.citations[0].chunk_id == "c1"
