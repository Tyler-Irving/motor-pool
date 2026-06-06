"""The grounded model: 4-bit base + LoRA adapter, emitting a TrainingTarget."""

from __future__ import annotations

from motor_pool.schemas import RetrievedChunk, TrainingTarget


class GroundedModel:
    def __init__(self, base_model_id: str, adapter_dir: str) -> None:
        self.base_model_id = base_model_id
        self.adapter_dir = adapter_dir

    def query(self, question: str, chunks: list[RetrievedChunk]) -> TrainingTarget:
        """Generate a grounded answer or refusal from the retrieved chunks."""
        raise NotImplementedError("Phase 7: load base + adapter; generate + parse.")
