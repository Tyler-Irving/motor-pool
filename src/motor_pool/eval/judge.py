"""Constrained per-claim entailment judge, backed by the OpenAI-compatible client.

Binary entailment with the chunk text in context, a fixed rubric, temperature 0,
from a different model family than the teacher (self-preference bias). This is
the LLM half of the shared support check; the deterministic half (lexical floor)
lives in data_gen.validate.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from motor_pool.llm import ChatClient

_SYSTEM = (
    "You are a strict entailment judge for a US Army vehicle technical manual. "
    "You decide only whether a CHUNK of the manual directly supports a CLAIM. "
    "Answer true only if the chunk states or directly implies the claim. Do not "
    "use outside knowledge. Reply with JSON only."
)


class _Verdict(BaseModel):
    entailed: bool


def judge_support(claim: str, chunk_text: str, *, client: ChatClient) -> bool:
    """Return True if the judge rules chunk_text entails claim. False on any error."""
    user = (
        f"CHUNK:\n{chunk_text}\n\n"
        f"CLAIM:\n{claim}\n\n"
        'Does the CHUNK support the CLAIM? Respond with {"entailed": true} or '
        '{"entailed": false}.'
    )
    try:
        return client.complete_json(_SYSTEM, user, _Verdict).entailed
    except Exception:
        # A judge/transport failure is conservatively treated as "not supported".
        return False


def make_judge(client: ChatClient) -> Callable[[str, str], bool]:
    """Bind a client into a (claim, chunk_text) -> bool entailment callable."""
    return lambda claim, text: judge_support(claim, text, client=client)
