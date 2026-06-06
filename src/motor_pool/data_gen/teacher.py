"""Frontier teacher client.

The teacher generates the distillation target (a grounded cited answer or a
refusal) from a question and the chunks the live retriever returned. It uses the
exact same grounding contract as inference (inference.prompt.generate_grounded),
so a target the teacher produces is the same shape the student is trained to
emit. The teacher is a stronger local model (qwen3:30b) served via ollama.
"""

from __future__ import annotations

from motor_pool.inference.prompt import generate_grounded
from motor_pool.llm import ChatClient
from motor_pool.schemas import RetrievedChunk, TrainingTarget


def generate_target(
    question: str, chunks: list[RetrievedChunk], *, client: ChatClient
) -> TrainingTarget | None:
    """Call the teacher and return a validated GroundedAnswer or Refusal (or None)."""
    return generate_grounded(question, chunks, client=client)
