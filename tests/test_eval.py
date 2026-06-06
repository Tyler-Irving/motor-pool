"""Phase 4 gate: metric computation and the results-table emitter."""

from __future__ import annotations

import json

from motor_pool.eval.metrics import score_system
from motor_pool.eval.report import emit_table
from motor_pool.schemas import (
    Citation,
    EvalItem,
    GroundedAnswer,
    ParagraphLocator,
    Refusal,
    RetrievedChunk,
)

CANON = ("TM-9-2320-280-10", "2-27", "2-136")
CORPUS = {CANON: "use the fire extinguisher remove locking pin direct nozzle"}


def _supports(claim: str, text: str) -> bool:
    content = {w for w in claim.lower().split() if len(w) > 3}
    return bool(content & {w for w in text.lower().split() if len(w) > 3})


def _citation() -> Citation:
    return Citation(
        source_doc_id="TM-9-2320-280-10",
        source_doc_title="Operator",
        edition_date="1996",
        locator=ParagraphLocator(chapter="2", paragraph="2-27"),
        tm_page_label="2-136",
        pdf_page_index=200,
        source_pdf_sha256="x",
        chunk_id="TM-9-2320-280-10:2-27",
    )


def _retrieved() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="TM-9-2320-280-10:2-27",
            text=CORPUS[CANON],
            score=1.0,
            citation=_citation(),
            content_type="procedure",
        )
    ]


def _items_and_predictions():
    items = [
        EvalItem(id="a1", question="how to use extinguisher", bucket="answerable_procedural",
                 label="answerable", gold_sections=["2-27"]),
        EvalItem(id="r1", question="depot rebuild", bucket="out_of_scope", label="refuse",
                 refuse_reason="out_of_scope"),
        EvalItem(id="r2", question="overhaul engine", bucket="out_of_scope", label="refuse",
                 refuse_reason="out_of_scope"),
        EvalItem(id="a2", question="check tire pressure", bucket="answerable_factual",
                 label="answerable", gold_sections=["2-5"]),
    ]
    predictions = [
        GroundedAnswer(summary="remove locking pin", citations=[_citation()]),  # correct answer
        Refusal(reason="not_in_corpus", message="not covered"),                 # correct refusal
        GroundedAnswer(summary="the depot rebuild procedure is complex", citations=[]),  # wrong: answered+unsupported
        Refusal(reason="out_of_scope", message="no"),                           # over-refusal
    ]
    retrieved = [_retrieved(), _retrieved(), _retrieved(), _retrieved()]
    return items, predictions, retrieved


def _score(bootstrap_n=0):
    items, preds, retr = _items_and_predictions()
    return score_system("base+RAG", items, preds, retr, CORPUS, _supports, bootstrap_n=bootstrap_n)


def test_refusal_metrics() -> None:
    s = _score()
    assert s.refusal_precision == 0.5  # tp=1 (r1), fp=1 (a2) -> 1/2
    assert s.refusal_recall == 0.5     # tp=1, fn=1 (r2 answered) -> 1/2
    assert s.refusal_f1 == 0.5
    assert s.over_refusal_rate == 0.5  # a2 refused, 1 of 2 answerable


def test_hallucination_split() -> None:
    s = _score()
    assert s.hallucination_answerable == 0.0   # a1 answered and supported
    assert s.hallucination_should_refuse == 1.0  # r2 answered with unsupported claim


def test_citation_and_faithfulness() -> None:
    s = _score()
    assert s.citation_exists_rate == 1.0      # only a1's citation, which exists
    assert s.valid_citation_rate == 1.0
    assert s.schema_valid_rate == 1.0
    assert s.faithfulness == 0.5              # a1 -> 1.0, r2 -> 0.0


def test_bootstrap_produces_cis() -> None:
    s = _score(bootstrap_n=200)
    assert set(s.ci) >= {"refusal_f1", "faithfulness", "citation_exists_rate"}
    for low, high in s.ci.values():
        assert low <= high


def test_emit_table(tmp_path) -> None:
    s = _score()
    md_path = emit_table([s], out_dir=tmp_path)
    md = open(md_path).read()
    assert "base+RAG" in md and "Faithfulness" in md and "Refuse-F1" in md
    data = json.loads((tmp_path / "results.json").read_text())
    assert data[0]["system"] == "base+RAG" and data[0]["n"] == 4
