"""Phase 5 gate: the validation gate and four-bucket sampling (no LLM)."""

from __future__ import annotations

from motor_pool.data_gen.question_sampler import sample_questions
from motor_pool.data_gen.validate import validate_record
from motor_pool.schemas import (
    Chunk,
    Citation,
    GroundedAnswer,
    ParagraphLocator,
    ProcedureStep,
    Provenance,
    Refusal,
    RetrievedChunk,
    TrainingRecord,
)


def _cit(para: str = "2-27") -> Citation:
    return Citation(
        source_doc_id="TM-9-2320-280-10", source_doc_title="O", edition_date="1996",
        locator=ParagraphLocator(chapter=para.split("-")[0], paragraph=para),
        tm_page_label="2-136", pdf_page_index=1, source_pdf_sha256="x", chunk_id=f"TM:{para}",
    )


def _chunk(para: str, title: str, ctype: str = "procedure") -> Chunk:
    return Chunk(chunk_id=f"TM:{para}", text=f"{para}. {title}", content_type=ctype, citation=_cit(para))


def _record(target, *, bucket: str = "answerable_procedural",
            chunk_text: str = "remove the locking pin and direct the nozzle") -> TrainingRecord:
    rc = RetrievedChunk(chunk_id="TM:2-27", text=chunk_text, score=1.0, citation=_cit(), content_type="procedure")
    return TrainingRecord(
        question="q", retrieved_chunks=[rc], target=target,
        provenance=Provenance(teacher_model="t", gen_timestamp="", bucket=bucket),
    )


def _supports(claim: str, text: str) -> bool:
    content = {w for w in claim.lower().split() if len(w) > 3}
    return bool(content & set(text.lower().split()))


def test_supported_answer_passes() -> None:
    answer = GroundedAnswer(summary="remove the locking pin", citations=[_cit()])
    assert validate_record(_record(answer), _supports).passed


def test_unsupported_claim_rejected() -> None:
    answer = GroundedAnswer(summary="the vehicle has twelve cylinders", citations=[_cit()])
    result = validate_record(_record(answer), _supports)
    assert not result.passed and any("unsupported" in r for r in result.reasons)


def test_leaked_number_rejected() -> None:
    answer = GroundedAnswer(summary="torque the locking pin to 250 foot pounds", citations=[_cit()])
    result = validate_record(_record(answer), _supports)
    assert not result.passed and any("leaked number 250" in r for r in result.reasons)


def test_missing_step_citation_rejected() -> None:
    answer = GroundedAnswer(
        summary="remove the locking pin",
        steps=[ProcedureStep(order=1, text="remove the pin", citations=[])],
        citations=[_cit()],
    )
    result = validate_record(_record(answer), _supports)
    assert not result.passed and any("step 1" in r for r in result.reasons)


def test_refusal_on_refusal_bucket_passes() -> None:
    record = _record(Refusal(reason="out_of_scope", message="no"), bucket="out_of_scope")
    assert validate_record(record, _supports).passed


def test_over_refusal_on_answerable_rejected() -> None:
    record = _record(Refusal(reason="not_in_corpus", message="no"), bucket="answerable_procedural")
    result = validate_record(record, _supports)
    assert not result.passed and any("over-refusal" in r for r in result.reasons)


def test_sample_questions_covers_buckets_without_dupes() -> None:
    chunks = [_chunk("2-1", "CONTROLS"), _chunk("1-1", "SCOPE"), _chunk("2-2", "DATA", "table")]

    def question_fn(chunk, factual, variant):
        return f"Question about {chunk.citation.locator.paragraph} variant {variant}?"

    mix = {
        "answerable_procedural": 0.4, "answerable_factual": 0.2,
        "hard_negative": 0.2, "out_of_scope": 0.1, "ambiguous": 0.1,
    }
    samples = sample_questions(chunks, bucket_mix=mix, target_n=10, question_fn=question_fn)
    buckets = {b for _, b in samples}
    assert {"answerable_procedural", "hard_negative", "ambiguous"} <= buckets
    answerable = [q for q, b in samples if b.startswith("answerable")]
    assert len(answerable) == len(set(answerable))  # no duplicate questions
