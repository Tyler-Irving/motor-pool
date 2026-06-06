"""Phase 4 gate: eval-set builder, the [C#] resolver, the generator, the runner."""

from __future__ import annotations

import httpx

from motor_pool.eval.eval_set import build_eval_set
from motor_pool.eval.runner import build_corpus_text, run_system
from motor_pool.inference.prompt import generate_grounded, resolve_model_output
from motor_pool.llm import ChatClient
from motor_pool.schemas import (
    Chunk,
    Citation,
    GroundedAnswer,
    ModelAnswer,
    ModelStep,
    ParagraphLocator,
    Refusal,
    RetrievedChunk,
)


def _citation(para: str, page: str) -> Citation:
    return Citation(
        source_doc_id="TM-9-2320-280-10",
        source_doc_title="Operator",
        edition_date="1996",
        locator=ParagraphLocator(chapter=para.split("-")[0], paragraph=para),
        tm_page_label=page,
        pdf_page_index=1,
        source_pdf_sha256="x",
        chunk_id=f"TM-9-2320-280-10:{para}",
    )


def _chunk(para: str, title: str, ctype: str = "procedure") -> Chunk:
    return Chunk(
        chunk_id=f"TM-9-2320-280-10:{para}",
        text=f"{para}. {title}\nbody text about {title.lower()}",
        content_type=ctype,
        citation=_citation(para, para),
    )


def _retrieved(para: str) -> RetrievedChunk:
    c = _chunk(para, "FIRE EXTINGUISHER OPERATION")
    return RetrievedChunk(
        chunk_id=c.chunk_id, text=c.text, score=1.0, citation=c.citation,
        content_type=c.content_type,
    )


def test_build_eval_set_is_section_disjoint() -> None:
    chunks = [
        _chunk("1-1", "SCOPE"),
        _chunk("2-1", "CONTROLS"),
        _chunk("2-2", "TABULATED DATA", ctype="table"),
        _chunk("3-1", "LUBRICATION"),
    ]
    items, held = build_eval_set(chunks, max_answerable=4, questions_per_para=2)
    answerable = [i for i in items if i.label == "answerable"]
    refusals = [i for i in items if i.label == "refuse"]
    assert len(answerable) == 4 and len(refusals) > 0
    gold = {s for i in answerable for s in i.gold_sections}
    assert gold <= set(held)  # gold sections are held out
    pool = {"1-1", "2-1", "2-2", "3-1"} - set(held)
    assert gold.isdisjoint(pool)  # eval gold is disjoint from the data-gen pool


def test_resolve_drops_out_of_range_indices() -> None:
    chunks = [_retrieved("2-27")]
    out = ModelAnswer(summary="do it", steps=[ModelStep(order=1, text="pull pin", cited=[1, 9])], cited=[1])
    resolved = resolve_model_output(out, chunks)
    assert isinstance(resolved, GroundedAnswer)
    assert len(resolved.steps[0].citations) == 1  # index 9 dropped
    assert resolved.citations[0].chunk_id == "TM-9-2320-280-10:2-27"


def test_resolve_passes_refusal_through() -> None:
    assert isinstance(
        resolve_model_output(Refusal(reason="not_in_corpus", message="no"), []), Refusal
    )


def test_generate_grounded_with_mock_client() -> None:
    content = '{"answer_type":"answer","summary":"remove the pin","steps":[],"cited":[1]}'

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = ChatClient(model="m", transport=httpx.MockTransport(handler))
    result = generate_grounded("how to use extinguisher", [_retrieved("2-27")], client=client)
    assert isinstance(result, GroundedAnswer)
    assert result.citations[0].chunk_id == "TM-9-2320-280-10:2-27"


def test_run_system_holds_retriever_constant() -> None:
    class StubRetriever:
        def retrieve(self, query, *, top_k=None, filters=None):
            return [_retrieved("2-27")]

    items, _ = build_eval_set([_chunk("2-1", "CONTROLS")], max_answerable=2, questions_per_para=2)
    answer = GroundedAnswer(summary="x", citations=[_citation("2-27", "2-136")])
    preds, retrieved = run_system(items, StubRetriever(), lambda q, ch: answer, top_k=4)
    assert len(preds) == len(items) == len(retrieved)
    assert all(len(r) == 1 for r in retrieved)


def test_build_corpus_text_concatenates_parts() -> None:
    text = build_corpus_text([_chunk("2-27", "FIRE")])
    assert ("TM-9-2320-280-10", "2-27", "2-27") in text
