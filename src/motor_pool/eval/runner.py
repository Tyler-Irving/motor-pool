"""Eval runner: run a system over the frozen eval set and score it.

A "system" is a generate(question, chunks) -> TrainingTarget | None callable.
base+RAG and finetuned+RAG differ only in the generator; the retriever, top_k,
and per-item chunk set are held identical, so the comparison isolates generation.
"""

from __future__ import annotations

from typing import Callable

from motor_pool.data_gen.canonicalize import canonicalize_citation
from motor_pool.schemas import Chunk, EvalItem, MetricScores, RetrievedChunk, TrainingTarget

from .metrics import CorpusText, Supports, score_system

Generate = Callable[[str, list[RetrievedChunk]], "TrainingTarget | None"]


def build_corpus_text(chunks: list[Chunk]) -> CorpusText:
    """Map each canonical citation to the full paragraph text (parts concatenated)."""
    text: CorpusText = {}
    for chunk in chunks:
        canon = canonicalize_citation(chunk.citation)
        text[canon] = (text.get(canon, "") + " " + chunk.text).strip()
    return text


def run_system(
    items: list[EvalItem],
    retriever,
    generate: Generate,
    *,
    top_k: int,
    log=lambda msg: None,
) -> tuple[list[TrainingTarget | None], list[list[RetrievedChunk]]]:
    """Retrieve and generate for every item. The retriever is held constant."""
    predictions: list[TrainingTarget | None] = []
    retrieved: list[list[RetrievedChunk]] = []
    for i, item in enumerate(items, start=1):
        chunks = retriever.retrieve(item.question, top_k=top_k)
        retrieved.append(chunks)
        predictions.append(generate(item.question, chunks))
        if i % 10 == 0 or i == len(items):
            log(f"generated {i}/{len(items)}")
    return predictions, retrieved


def run_eval(
    system: str,
    items: list[EvalItem],
    retriever,
    generate: Generate,
    corpus_text: CorpusText,
    supports: Supports,
    *,
    top_k: int = 6,
    bootstrap_n: int = 0,
    log=lambda msg: None,
) -> MetricScores:
    """Run one system end to end and return its results row."""
    predictions, retrieved = run_system(items, retriever, generate, top_k=top_k, log=log)
    log("scoring (judging entailment) ...")
    return score_system(
        system, items, predictions, retrieved, corpus_text, supports, bootstrap_n=bootstrap_n
    )
