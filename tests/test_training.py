"""Phase 6 gate: training-target construction (no torch/unsloth needed)."""

from __future__ import annotations

import json

from pydantic import TypeAdapter

from motor_pool.inference.prompt import resolve_model_output, to_model_output
from motor_pool.schemas import (
    Citation,
    GroundedAnswer,
    ModelOutput,
    ParagraphLocator,
    ProcedureStep,
    Provenance,
    Refusal,
    RetrievedChunk,
    TrainingRecord,
)
from motor_pool.training.dataset import to_messages


def _rc(para: str) -> RetrievedChunk:
    cit = Citation(
        source_doc_id="TM-9-2320-280-10", source_doc_title="O", edition_date="1996",
        locator=ParagraphLocator(chapter=para.split("-")[0], paragraph=para),
        tm_page_label=para, pdf_page_index=1, source_pdf_sha256="x", chunk_id=f"TM:{para}",
    )
    return RetrievedChunk(chunk_id=f"TM:{para}", text=f"{para} body text", score=1.0,
                          citation=cit, content_type="procedure")


def test_to_model_output_roundtrips() -> None:
    chunks = [_rc("2-27"), _rc("2-2")]
    target = GroundedAnswer(
        summary="remove the pin",
        steps=[ProcedureStep(order=1, text="pull the pin", citations=[chunks[0].citation])],
        citations=[chunks[0].citation, chunks[1].citation],
    )
    mo = to_model_output(target, chunks)
    assert mo.cited == [1, 2]
    assert mo.steps[0].cited == [1]
    assert resolve_model_output(mo, chunks) == target  # exact inverse


def test_refusal_passes_through() -> None:
    assert isinstance(to_model_output(Refusal(reason="not_in_corpus", message="no"), []), Refusal)


def test_to_messages_shape_and_valid_assistant() -> None:
    record = TrainingRecord(
        question="how do I use the extinguisher?",
        retrieved_chunks=[_rc("2-27")],
        target=GroundedAnswer(summary="x", citations=[_rc("2-27").citation]),
        provenance=Provenance(teacher_model="t", gen_timestamp="", bucket="answerable_procedural"),
    )
    messages = to_messages(record)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    # the assistant turn must be valid ModelOutput JSON (what the model learns to emit)
    TypeAdapter(ModelOutput).validate_python(json.loads(messages[2]["content"]))
    # the user turn carries the tagged context and the question
    assert "[C1]" in messages[1]["content"]
    assert "extinguisher" in messages[1]["content"]
