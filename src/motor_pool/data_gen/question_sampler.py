"""Four-bucket question sampling for distillation.

answerable_* questions are back-generated from the data-gen pool paragraphs (the
corpus minus the held-out eval sections), so the student sees realistic top-k at
train time and no eval procedure leaks in. Refusal buckets drive refusal
training and are templated to be distinct from the eval set's curated refusals.
The live Retriever is run later by the pipeline, never the gold chunk injected.
"""

from __future__ import annotations

from typing import Callable

from ..schemas import Bucket, Chunk

QuestionFn = Callable[[Chunk, bool, int], str]

# Training refusal pools (distinct questions from the eval set's curated ones).
# Each bucket is capped at its distinct questions, never padded with duplicates.
_COMPONENTS = [
    "alternator", "starter", "water pump", "glow plug", "glow plug controller",
    "fuel filter", "fuel injection pump", "serpentine belt", "radiator", "thermostat",
    "brake caliper", "wheel bearing", "tie rod end", "ball joint", "universal joint",
    "fuel injector", "oil pump", "power steering pump", "transfer case",
    "front differential", "rear differential", "transmission", "cylinder head",
    "turbocharger",
]
_HARD_TEMPLATES = [
    "How do I replace the {c} at unit maintenance level?",
    "What is the torque specification for the {c}?",
    "How do I rebuild the {c}?",
]
_VEHICLES = [
    "M1 Abrams tank", "Bradley Fighting Vehicle", "Stryker", "M88 recovery vehicle",
    "UH-60 Black Hawk helicopter", "civilian pickup truck", "school bus", "forklift",
    "passenger sedan", "cargo van",
]
_TASKS = [
    "change the oil", "replace a tire", "start the engine", "check the coolant",
    "bleed the brakes", "jump-start the battery",
]
_GENERAL = [
    "What is the capital of France?", "How do I bake sourdough bread?",
    "What is the weather forecast for tomorrow?", "How do I tune a guitar?",
    "What languages are spoken in Canada?", "How do I write a resume?",
    "What is the boiling point of water?", "How do I knit a scarf?",
]
_AMBIGUOUS = [
    "How do I fix it?", "What is the procedure?", "Is it broken?", "What should I check?",
    "How often?", "Where is it located?", "What does it mean?", "Can you help with that?",
    "What is wrong?", "How do I do it?", "When is it due?", "What value should it be?",
    "What are the steps?", "Is that normal?", "Should I be worried?", "What now?",
    "How much?", "Which one?", "What size?", "Is it safe?", "What is the limit?",
    "How long?", "What do I need?", "Where do I start?", "What comes next?",
    "Is it required?", "How tight?", "Which way?", "How many?", "What torque?",
]


def _is_factual(chunk: Chunk) -> bool:
    return chunk.content_type == "table" or chunk.citation.locator.chapter == "1"


def _answerable(
    chunks: list[Chunk], count: int, question_fn: QuestionFn, bucket: Bucket, seen: set[str]
) -> list[tuple[str, Bucket]]:
    out: list[tuple[str, Bucket]] = []
    variant = 0
    while len(out) < count and chunks and variant < 8:
        progressed = False
        for chunk in chunks:
            if len(out) >= count:
                break
            question = question_fn(chunk, _is_factual(chunk), variant)
            key = question.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((question, bucket))
            progressed = True
        variant += 1
        if not progressed:
            break
    return out


def _refusals(n_hard: int, n_oos: int, n_amb: int) -> list[tuple[str, Bucket]]:
    """Distinct refusal questions, capped at the available templates (no padding)."""
    hard = [t.format(c=c) for c in _COMPONENTS for t in _HARD_TEMPLATES]
    oos = [f"How do I {t} on a {v}?" for v in _VEHICLES for t in _TASKS] + _GENERAL
    out = (
        [(q, "hard_negative") for q in hard[:n_hard]]
        + [(q, "out_of_scope") for q in oos[:n_oos]]
        + [(q, "ambiguous") for q in _AMBIGUOUS[:n_amb]]
    )
    seen: set[str] = set()
    deduped: list[tuple[str, Bucket]] = []
    for question, bucket in out:
        if question not in seen:
            seen.add(question)
            deduped.append((question, bucket))
    return deduped


def sample_questions(
    pool_chunks: list[Chunk],
    *,
    bucket_mix: dict[str, float],
    target_n: int,
    question_fn: QuestionFn,
) -> list[tuple[str, Bucket]]:
    """Return (question, bucket) pairs roughly matching bucket_mix * target_n.

    The small corpus caps distinct answerable questions at ~2 per pool paragraph,
    so the answerable count may fall short of the target; that is reported by the
    pipeline rather than padded with duplicates.
    """
    counts = {b: int(round(bucket_mix.get(b, 0.0) * target_n)) for b in bucket_mix}
    procedural = [c for c in pool_chunks if not _is_factual(c)]
    factual = [c for c in pool_chunks if _is_factual(c)]
    seen: set[str] = set()
    out: list[tuple[str, Bucket]] = []
    out += _answerable(procedural, counts.get("answerable_procedural", 0), question_fn, "answerable_procedural", seen)
    out += _answerable(factual, counts.get("answerable_factual", 0), question_fn, "answerable_factual", seen)
    out += _refusals(
        counts.get("hard_negative", 0), counts.get("out_of_scope", 0), counts.get("ambiguous", 0)
    )
    return out
