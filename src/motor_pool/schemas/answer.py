"""Answer schemas: the model's structured output contract.

This is the single shape used at three points: the distillation target during
data generation, the fine-tuned model's runtime output, and the value the CLI
(and the future V2 agent) consumes. A response is either a GroundedAnswer or a
Refusal, discriminated on `answer_type`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .citation import Citation


class ProcedureStep(BaseModel):
    """One ordered step of a procedure, with the citation(s) that support it."""

    order: int
    text: str
    citations: list[Citation] = Field(
        description="At least one. Enforced by the data-gen validator and eval."
    )


class GroundedAnswer(BaseModel):
    """An answer grounded in retrieved chunks. Every claim is cited."""

    answer_type: Literal["answer"] = "answer"
    summary: str
    steps: list[ProcedureStep] = Field(
        default_factory=list,
        description="Ordered steps for procedural answers. Empty for plain facts.",
    )
    citations: list[Citation] = Field(
        description="At least one. The sections/pages the summary draws on."
    )


class Refusal(BaseModel):
    """A refusal returned when the retrieved chunks do not support an answer."""

    answer_type: Literal["refusal"] = "refusal"
    reason: Literal[
        "out_of_scope",
        "not_in_corpus",
        "insufficient_context",
        "ambiguous",
    ]
    message: str


# Discriminated union on answer_type. Validate a bare value with:
#   from pydantic import TypeAdapter
#   TypeAdapter(TrainingTarget).validate_python(obj)
TrainingTarget = Annotated[
    GroundedAnswer | Refusal, Field(discriminator="answer_type")
]


# What a model actually emits at inference: it cites chunks by their [C#] index
# (1-based), which the runner resolves to the real retrieved citations. This
# keeps citation output tractable for small local models. A refusal is the same
# Refusal shape. resolve_model_output() turns this into a TrainingTarget.
class ModelStep(BaseModel):
    order: int
    text: str
    cited: list[int] = Field(
        default_factory=list, description="1-based [C#] indices this step relies on."
    )


class ModelAnswer(BaseModel):
    answer_type: Literal["answer"] = "answer"
    summary: str
    steps: list[ModelStep] = Field(default_factory=list)
    cited: list[int] = Field(default_factory=list)


ModelOutput = Annotated[ModelAnswer | Refusal, Field(discriminator="answer_type")]
