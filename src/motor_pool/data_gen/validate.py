"""The shared support checker and the data-gen validation gate.

`make_supports` combines a cheap deterministic lexical floor with the injected
LLM judge: a claim with almost no word overlap with the chunk is rejected without
a judge call (catches obvious fabrication and saves calls), otherwise the judge
decides entailment. eval.metrics and the data-gen gate both use this one checker,
so "accepted by the validator" and "counted valid by eval" can never disagree.

The judge is passed in as a callable so this module does not import eval.judge
(eval.metrics already imports data_gen.canonicalize; injecting avoids a cycle).
"""

from __future__ import annotations

import re
from typing import Callable

from ..schemas import GroundedAnswer, Refusal, TrainingRecord, ValidationResult

Judge = Callable[[str, str], bool]
Supports = Callable[[str, str], bool]

_WORD_RE = re.compile(r"[a-z0-9.\-]+")
# Multi-character numbers (skip bare step digits like "1"). Specs, torques, NSNs.
_NUM_RE = re.compile(r"\d[\d,\.]*\d")


def lexical_overlap(claim: str, text: str) -> float:
    """Fraction of a claim's content words that also appear in the chunk text."""
    claim_words = {w for w in _WORD_RE.findall(claim.lower()) if len(w) > 3}
    if not claim_words:
        return 0.0
    text_words = set(_WORD_RE.findall(text.lower()))
    return len(claim_words & text_words) / len(claim_words)


def make_supports(judge: Judge, min_overlap: float = 0.15) -> Supports:
    """Deterministic lexical floor, then the judge. Shared by data-gen and eval."""

    def supports(claim: str, text: str) -> bool:
        if lexical_overlap(claim, text) < min_overlap:
            return False
        return judge(claim, text)

    return supports


def validate_record(
    record: TrainingRecord,
    supports: Supports,
    *,
    require_citation_per_step: bool = True,
    reject_unsupported_numbers: bool = True,
) -> ValidationResult:
    """Run the deterministic-first validation gate over one training record.

    Citations are in-context by construction (the model cites retrieved chunks by
    index, which inference.prompt resolves), so existence and in-context checks
    are guaranteed. This gate enforces the rest: every claim is supported by a
    retrieved chunk, every step is cited, and no spec/number is invented.
    """
    target = record.target
    bucket = record.provenance.bucket
    if isinstance(target, Refusal):
        # A refusal on an answerable question is an over-refusal; rejecting it
        # keeps the base model's over-refusals out of the training data, so the
        # fine-tune learns to answer answerable questions rather than refuse more.
        if bucket.startswith("answerable"):
            return ValidationResult(
                passed=False, reasons=["over-refusal: refused an answerable question"]
            )
        return ValidationResult(passed=True)

    assert isinstance(target, GroundedAnswer)
    chunks = record.retrieved_chunks
    chunk_text = " ".join(c.text for c in chunks)
    reasons: list[str] = []

    if not target.citations:
        reasons.append("answer has no citations")
    if require_citation_per_step:
        for step in target.steps:
            if not step.citations:
                reasons.append(f"step {step.order} has no citation")

    claims = [target.summary] + [s.text for s in target.steps]
    for claim in claims:
        if not any(supports(claim, c.text) for c in chunks):
            reasons.append(f"unsupported claim: {claim[:48]}")

    if reject_unsupported_numbers:
        for claim in claims:
            for number in _NUM_RE.findall(claim):
                if number not in chunk_text:
                    reasons.append(f"leaked number {number}")

    return ValidationResult(passed=not reasons, reasons=reasons)
