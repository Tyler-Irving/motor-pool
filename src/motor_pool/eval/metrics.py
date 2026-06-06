"""Metric computation for the base+RAG vs finetuned+RAG comparison.

Most metrics are deterministic against the gold labels and the corpus (refusal
precision/recall, citation existence, schema validity). The support-dependent
metrics (citation support, hallucination, faithfulness) call an injected
`supports(claim, text) -> bool` checker, which in production is the shared
deterministic-plus-judge entailment from data_gen.validate. Injecting it keeps
this module pure and testable, and guarantees data-gen and eval use one checker.

A "claim" is the answer summary plus each procedure step's text. Citations are
canonicalized (data_gen.canonicalize) before being matched to the corpus, so
"Para 2-14" and "Section 2-14" compare equal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from motor_pool.data_gen.canonicalize import canonicalize_citation
from motor_pool.schemas import (
    EvalItem,
    GroundedAnswer,
    MetricScores,
    Refusal,
    RetrievedChunk,
    TrainingTarget,
)

Supports = Callable[[str, str], bool]
CorpusText = dict[tuple[str, str, str], str]  # canonical citation -> paragraph text

# The rate metrics that get bootstrap confidence intervals.
_CI_KEYS = (
    "hallucination_answerable",
    "citation_exists_rate",
    "valid_citation_rate",
    "refusal_f1",
    "faithfulness",
)


def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _claims(answer: GroundedAnswer) -> list[str]:
    claims = [answer.summary] if answer.summary.strip() else []
    claims.extend(step.text for step in answer.steps)
    return claims


def _citation_claims(answer: GroundedAnswer):
    """Yield (citation, supporting_claim) pairs across summary and step citations."""
    for citation in answer.citations:
        yield citation, answer.summary
    for step in answer.steps:
        for citation in step.citations:
            yield citation, step.text


def _item_record(
    item: EvalItem,
    prediction: TrainingTarget | None,
    retrieved: list[RetrievedChunk],
    corpus_text: CorpusText,
    supports: Supports,
) -> dict:
    rec = {
        "should_refuse": item.label == "refuse",
        "answered": isinstance(prediction, GroundedAnswer),
        "refused": isinstance(prediction, Refusal),
        "valid_schema": prediction is not None,
        "claims_supported": [],  # list[bool] over the answer's claims
        "citations": [],  # list[(exists, supported)]
    }
    if isinstance(prediction, GroundedAnswer):
        for claim in _claims(prediction):
            rec["claims_supported"].append(
                any(supports(claim, c.text) for c in retrieved)
            )
        for citation, claim in _citation_claims(prediction):
            canon = canonicalize_citation(citation)
            exists = canon in corpus_text
            supported = exists and supports(claim, corpus_text[canon])
            rec["citations"].append((exists, supported))
    return rec


def _aggregate(records: list[dict], system: str) -> dict:
    n = len(records)
    should = [r for r in records if r["should_refuse"]]
    answerable = [r for r in records if not r["should_refuse"]]

    tp = sum(1 for r in should if r["refused"])
    fn = len(should) - tp
    fp = sum(1 for r in answerable if r["refused"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def hallucinated(rs):
        answered = [r for r in rs if r["answered"]]
        return _mean(any(not s for s in r["claims_supported"]) for r in answered)

    faiths = [
        sum(r["claims_supported"]) / len(r["claims_supported"])
        for r in records
        if r["answered"] and r["claims_supported"]
    ]
    citations = [c for r in records for c in r["citations"]]

    return {
        "system": system,
        "n": n,
        "schema_valid_rate": _mean(r["valid_schema"] for r in records),
        "hallucination_answerable": hallucinated(answerable),
        "hallucination_should_refuse": hallucinated(should),
        "citation_exists_rate": _mean(e for e, _ in citations),
        "citation_supported_rate": _mean(s for _, s in citations),
        "valid_citation_rate": _mean(e and s for e, s in citations),
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": f1,
        "over_refusal_rate": (fp / len(answerable)) if answerable else 0.0,
        "faithfulness": _mean(faiths),
    }


def _bootstrap_ci(records: list[dict], system: str, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in _CI_KEYS}
    index = np.arange(len(records))
    for _ in range(n):
        picked = [records[i] for i in rng.choice(index, size=len(records), replace=True)]
        agg = _aggregate(picked, system)
        for key in _CI_KEYS:
            samples[key].append(agg[key])
    return {
        key: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        for key, vals in samples.items()
    }


def score_system(
    system: str,
    items: list[EvalItem],
    predictions: list[TrainingTarget | None],
    retrieved: list[list[RetrievedChunk]],
    corpus_text: CorpusText,
    supports: Supports,
    *,
    bootstrap_n: int = 0,
    seed: int = 3407,
) -> MetricScores:
    """Score one system over the eval set, returning one results row."""
    records = [
        _item_record(item, pred, chunks, corpus_text, supports)
        for item, pred, chunks in zip(items, predictions, retrieved)
    ]
    point = _aggregate(records, system)
    ci = _bootstrap_ci(records, system, bootstrap_n, seed) if bootstrap_n else {}
    return MetricScores(**point, ci=ci)
