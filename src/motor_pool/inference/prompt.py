"""The grounding prompt contract, its resolver, and a client-based generator.

This is the single source of truth for how a question plus retrieved chunks are
presented to a model, and how the model's `[C#]`-indexed reply is turned back
into a TrainingTarget with real citations. It is shared by base+RAG, the
finetuned model, and the data-gen teacher, so training and inference cannot
drift. Keep this byte-stable once training data is generated against it.
"""

from __future__ import annotations

from pydantic import TypeAdapter

from motor_pool.llm import ChatClient, extract_json
from motor_pool.schemas import (
    GroundedAnswer,
    ModelAnswer,
    ModelOutput,
    ProcedureStep,
    Refusal,
    RetrievedChunk,
    TrainingTarget,
)

SYSTEM_PROMPT = (
    "You are a maintenance assistant for the HMMWV operator technical manual. "
    "Answer using ONLY the provided context passages, and cite the passages you "
    "use by their number. If the context does not contain the answer, refuse. "
    "Respond with a single JSON object and nothing else."
)

_CHUNK_CHARS = 700  # per-chunk budget in the prompt
_OUTPUT_SPEC = (
    "Respond with ONE JSON object.\n"
    'To answer: {"answer_type":"answer","summary":"<one sentence>",'
    '"steps":[{"order":1,"text":"<step>","cited":[1]}],"cited":[1,2]}. '
    "The cited arrays hold the [C#] numbers you used.\n"
    'To refuse: {"answer_type":"refusal","reason":'
    '"not_in_corpus|out_of_scope|insufficient_context|ambiguous","message":"<why>"}.'
)

_OUTPUT_ADAPTER = TypeAdapter(ModelOutput)


def format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        loc = chunk.citation.locator
        text = " ".join(chunk.text.split())[:_CHUNK_CHARS]
        lines.append(
            f"[C{i}] (para {loc.paragraph}, p.{chunk.citation.tm_page_label}) {text}"
        )
    return "\n".join(lines)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Return (system, user) for the grounding prompt."""
    user = (
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{format_context(chunks)}\n\n"
        f"{_OUTPUT_SPEC}"
    )
    return SYSTEM_PROMPT, user


def resolve_model_output(
    output: ModelOutput, chunks: list[RetrievedChunk]
) -> TrainingTarget:
    """Turn a [C#]-indexed model reply into a TrainingTarget with real citations.

    Indices outside the retrieved range are dropped (a fabricated citation to a
    non-provided chunk simply yields no citation, which eval then scores).
    """
    if isinstance(output, Refusal):
        return output

    def cites(indices: list[int]):
        return [chunks[i - 1].citation for i in indices if 1 <= i <= len(chunks)]

    steps = [
        ProcedureStep(order=s.order, text=s.text, citations=cites(s.cited))
        for s in output.steps
    ]
    return GroundedAnswer(summary=output.summary, steps=steps, citations=cites(output.cited))


def generate_grounded(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    client: ChatClient,
    retries: int = 2,
) -> TrainingTarget | None:
    """Generate a grounded answer or refusal. Returns None if output never parses."""
    system, user = build_messages(question, chunks)
    prompt = user
    for _ in range(retries + 1):
        text = client.complete(system, prompt, json_mode=True)
        try:
            output = _OUTPUT_ADAPTER.validate_python(extract_json(text))
            return resolve_model_output(output, chunks)
        except Exception as exc:  # parse or validation failure
            prompt = f"{user}\n\nReturn ONLY valid JSON. Your last reply failed: {str(exc)[:150]}"
    return None
