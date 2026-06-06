"""Turn TrainingRecords into chat-template training examples.

Uses the model-native template (chatml for Qwen2.5) and response-only masking so
loss falls on the assistant turn. A sample batch is checked for non-all-(-100)
labels before launch to catch the zero-loss masking trap.
"""

from __future__ import annotations

from motor_pool.schemas import TrainingRecord


def to_chat_messages(record: TrainingRecord) -> list[dict]:
    """Render one record as chat messages (system + user with chunks + assistant)."""
    raise NotImplementedError("Phase 6: TrainingRecord -> chat messages.")


def build_dataset(path: str):
    """Load a jsonl of TrainingRecords into a tokenized, response-masked dataset."""
    raise NotImplementedError("Phase 6: jsonl -> HF dataset with response masking.")
