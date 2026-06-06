"""Data-generation schemas: training records, buckets, provenance, validation.

A TrainingRecord is one distillation example: a question, the chunks the live
Retriever returned for it, and the teacher's structured target (answer or
refusal). The retrieved_chunks are the model's input at train time, so the
student learns the behavior of grounding-on-context-and-citing, never the facts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .answer import TrainingTarget
from .chunk import RetrievedChunk

# Question buckets. answerable_* train grounded answers; the rest train refusals.
Bucket = Literal[
    "answerable_procedural",
    "answerable_factual",
    "hard_negative",
    "out_of_scope",
    "ambiguous",
]


class Provenance(BaseModel):
    """Where a training record came from. Carried for auditability."""

    teacher_model: str
    gen_timestamp: str
    bucket: Bucket


class TrainingRecord(BaseModel):
    """One distillation pair: (question + retrieved chunks) -> target."""

    question: str
    retrieved_chunks: list[RetrievedChunk]
    target: TrainingTarget
    provenance: Provenance


class ValidationResult(BaseModel):
    """Outcome of the data-gen validation gate for one record."""

    passed: bool
    reasons: list[str] = Field(
        default_factory=list,
        description="Rejection reasons when passed is False. Empty when passed.",
    )
