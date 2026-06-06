"""Evaluation schemas: held-out eval items and the per-system metric row.

An EvalItem is one hand-verified question with a gold label (answerable or
refuse) and, for answerable items, the gold supporting section(s) used to score
citation correctness. MetricScores is one row of the results table.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .datagen import Bucket

EvalLabel = Literal["answerable", "refuse"]
RefuseReason = Literal["out_of_scope", "not_in_corpus", "insufficient_context", "ambiguous"]


class EvalItem(BaseModel):
    """One frozen, hand-verified evaluation item."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    bucket: Bucket
    label: EvalLabel
    refuse_reason: RefuseReason | None = None
    gold_sections: list[str] = Field(
        default_factory=list,
        description="Canonical paragraph keys (e.g. '2-27') that support an "
        "answerable item. Empty for refusals.",
    )
    notes: str | None = None


class MetricScores(BaseModel):
    """One row of the base+RAG vs finetuned+RAG results table."""

    system: str
    n: int
    schema_valid_rate: float
    hallucination_answerable: float
    hallucination_should_refuse: float
    citation_exists_rate: float
    citation_supported_rate: float
    valid_citation_rate: float
    refusal_precision: float
    refusal_recall: float
    refusal_f1: float
    over_refusal_rate: float
    faithfulness: float
    ci: dict[str, list[float]] = Field(
        default_factory=dict, description="metric name -> [low, high] bootstrap CI."
    )
