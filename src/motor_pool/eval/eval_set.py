"""Build the frozen, section-disjoint evaluation set from the -10 corpus.

Answerable items are templated from a held-out slice of paragraphs (with their
paragraph as the gold section); the rest of the corpus is the data-gen pool, so
no procedure seen in training appears in eval. Refusal items are curated: HMMWV
questions beyond the operator manual (hard negatives), other-vehicle/general
questions (out of scope), and vague questions (ambiguous). The held-out section
list is written alongside so data generation can exclude it.

The output is hand-verifiable: every answerable item maps to a real paragraph
and every refusal is genuinely uncovered. Edit data/eval/heldout_v1.jsonl freely
after generating; it is the frozen artifact, this builder just seeds it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from motor_pool.schemas import Chunk, EvalItem

# (chunk, is_factual, variant_index) -> a natural question answered by the chunk.
QuestionFn = Callable[[Chunk, bool, int], str]

# Curated refusals. These are genuinely outside the operator manual's scope.
_HARD_NEGATIVE = [
    "What is the torque specification for the cylinder head bolts?",
    "How do I rebuild the transmission?",
    "How do I replace the fuel injection pump?",
    "How do I adjust the engine valve lash?",
    "How do I overhaul the transfer case?",
    "What is the STE/ICE test procedure for the alternator?",
    "How do I bleed the hydraulic brake system at the master cylinder?",
    "How do I replace the glow plug controller?",
    "How do I disassemble the front differential?",
    "What is the resistance spec for the fuel solenoid windings?",
    "How do I replace the water pump?",
    "How do I set the injection timing?",
    "How do I rebuild the power steering gear?",
    "What is the unit-level procedure for replacing the starter?",
    "How do I pressure test the cooling system?",
    "How do I replace the turbocharger?",
    "How do I time the injection pump to the engine?",
    "What is the procedure for replacing the head gasket?",
    "How do I adjust the parking brake shoes at the backing plate?",
    "How do I replace the transmission oil cooler lines?",
]
_OUT_OF_SCOPE = [
    "How do I change the oil in an M1 Abrams tank?",
    "What is the recommended tire pressure for a Toyota Camry?",
    "How do I operate a Bradley Fighting Vehicle?",
    "What is the firing rate of an M2 machine gun?",
    "How do I fly a UH-60 Black Hawk helicopter?",
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "What is the fuel capacity of an Abrams tank?",
    "How do I jump start a Honda Civic?",
    "What is the top speed of an F-16?",
    "How do I replace the battery in a Ford F-150?",
    "What is the muzzle velocity of a 120mm tank round?",
    "How do I operate a forklift?",
    "What languages are spoken in Canada?",
    "How do I tune a guitar?",
    "What is the range of a Javelin missile?",
    "How do I drive a manual transmission car?",
    "What is the weather forecast for tomorrow?",
]
_AMBIGUOUS = [
    "How do I fix it?",
    "What is the procedure?",
    "Is it broken?",
    "What should I check?",
    "How often?",
    "Tell me about maintenance.",
    "What does it mean?",
    "Can you help with the thing?",
    "What is the spec?",
    "How do I do that?",
    "Where is it located?",
    "What is wrong with it?",
]

_NUM_PREFIX = re.compile(r"^\S+\.\s*")


def _title_phrase(chunk: Chunk) -> str:
    """The paragraph title, lowercased, without its 'N-NN.' number prefix."""
    first_line = chunk.text.splitlines()[0]
    return _NUM_PREFIX.sub("", first_line).strip().lower()


def _unique_paragraphs(chunks: list[Chunk]) -> dict[str, Chunk]:
    paras: dict[str, Chunk] = {}
    for chunk in chunks:
        para = chunk.citation.locator.paragraph or ""
        if para and para not in paras:
            paras[para] = chunk
    return paras


def _para_key(para: str):
    chapter, rest = para.split("-", 1)
    return (int(chapter), float(rest))


def _template(chunk: Chunk, factual: bool, variant: int) -> str:
    """Deterministic fallback question (used when no LLM generator is given)."""
    phrase = _title_phrase(chunk)
    if factual:
        options = [
            f"According to the operator manual, what are the {phrase}?",
            f"What does the manual specify for the {phrase}?",
        ]
    else:
        options = [
            f"What is the operator procedure for {phrase}?",
            f"How do I perform {phrase} on the HMMWV?",
        ]
    return options[variant % len(options)]


# Two framings so the two questions per paragraph differ (a general operator
# question and a more specific one). The model sees the paragraph text, not the
# title, so the questions read naturally.
_Q_PROMPTS = [
    (
        "You write one short, natural question a HMMWV operator would ask whose "
        "answer is contained in the passage. Output only the question ending with "
        "'?'. Do not mention 'the manual', 'the passage', or paragraph numbers.",
        "PASSAGE:\n{body}\n\nWrite one natural operator question answered by this passage.",
    ),
    (
        "You write one short, specific question a HMMWV operator would ask about a "
        "particular step, value, condition, or warning in the passage. Output only "
        "the question ending with '?'. Do not mention 'the manual' or 'the passage'.",
        "PASSAGE:\n{body}\n\nWrite one specific question answered by this passage, "
        "different from a generic overview question.",
    ),
]


def llm_question_fn(client) -> QuestionFn:
    """Build a question generator from a chat client (duck-typed: has .complete)."""

    def generate(chunk: Chunk, factual: bool, variant: int) -> str:
        body = " ".join(chunk.text.split())[:700]
        system, user = _Q_PROMPTS[variant % len(_Q_PROMPTS)]
        try:
            raw = client.complete(system, user.format(body=body))
        except Exception:
            return _template(chunk, factual, variant)
        question = raw.strip().splitlines()[-1].strip().strip('"').strip()
        ok = question.endswith("?") and 10 < len(question) < 200
        return question if ok else _template(chunk, factual, variant)

    return generate


def build_eval_set(
    chunks: list[Chunk],
    *,
    question_fn: QuestionFn | None = None,
    max_answerable: int = 90,
    questions_per_para: int = 2,
) -> tuple[list[EvalItem], list[str]]:
    """Return (eval_items, held_out_sections).

    Answerable questions come from `question_fn` (a model back-generating natural
    questions) or, if None, from deterministic templates. Refusals are curated.
    """
    paras = _unique_paragraphs(chunks)
    ordered = sorted(paras, key=_para_key)
    # Hold out every other paragraph for eval; the rest is the data-gen pool.
    held = ordered[::2][: (max_answerable + questions_per_para - 1) // questions_per_para]
    make_question = question_fn or _template

    items: list[EvalItem] = []
    seen: set[str] = set()
    for para in held:
        chunk = paras[para]
        factual = chunk.content_type == "table" or para.startswith("1-")
        bucket = "answerable_factual" if factual else "answerable_procedural"
        for variant in range(questions_per_para):
            if len(items) >= max_answerable:
                break
            question = make_question(chunk, factual, variant)
            key = re.sub(r"\s+", " ", question.strip().lower())
            if key in seen:  # the two framings converged; skip the duplicate
                continue
            seen.add(key)
            items.append(
                EvalItem(
                    id=f"a{len(items) + 1}",
                    question=question,
                    bucket=bucket,
                    label="answerable",
                    gold_sections=[para],
                )
            )

    refusals = (
        [(q, "hard_negative", "not_in_corpus") for q in _HARD_NEGATIVE]
        + [(q, "out_of_scope", "out_of_scope") for q in _OUT_OF_SCOPE]
        + [(q, "ambiguous", "ambiguous") for q in _AMBIGUOUS]
    )
    refusal_items = [
        EvalItem(
            id=f"r{n}",
            question=question,
            bucket=bucket,
            label="refuse",
            refuse_reason=reason,
        )
        for n, (question, bucket, reason) in enumerate(refusals, start=1)
    ]
    # Interleave answerable and refusal items so any prefix subset is balanced.
    return _interleave(items, refusal_items), held


def _interleave(a: list[EvalItem], b: list[EvalItem]) -> list[EvalItem]:
    keyed = [(i / max(1, len(a)), x) for i, x in enumerate(a)]
    keyed += [(i / max(1, len(b)), x) for i, x in enumerate(b)]
    keyed.sort(key=lambda t: t[0])
    return [x for _, x in keyed]


def write_eval_set(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def read_eval_set(path: Path) -> list[EvalItem]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalItem.model_validate_json(line) for line in lines if line.strip()]
