"""Turn TrainingRecords into chat examples for response-only QLoRA training.

The assistant turn is the exact [C#]-indexed ModelOutput JSON the model emits at
inference (built from the stored target via inference.prompt.to_model_output), so
training and inference share one contract. The system and user turns are the same
grounding prompt used everywhere else.
"""

from __future__ import annotations

from pathlib import Path

from motor_pool.inference.prompt import build_messages, to_model_output
from motor_pool.schemas import TrainingRecord


def to_messages(record: TrainingRecord) -> list[dict]:
    """Render one record as [system, user, assistant] chat messages."""
    system, user = build_messages(record.question, record.retrieved_chunks)
    assistant = to_model_output(record.target, record.retrieved_chunks).model_dump_json()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def read_records(path: str | Path) -> list[TrainingRecord]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [TrainingRecord.model_validate_json(line) for line in lines if line.strip()]


def build_dataset(path: str | Path):
    """Load a jsonl of TrainingRecords into an HF Dataset of {"messages": [...]}."""
    from datasets import Dataset

    return Dataset.from_list([{"messages": to_messages(r)} for r in read_records(path)])
