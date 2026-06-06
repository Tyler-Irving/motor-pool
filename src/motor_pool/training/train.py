"""QLoRA trainer entrypoint.

Loads the 4-bit base, attaches the LoRA adapter, trains from TrainingConfig, and
saves the adapter only plus a run_manifest.json (seed, config hash, data sha256).
"""

from __future__ import annotations

from motor_pool.config import TrainingConfig


def train(config: TrainingConfig) -> str:
    """Run training. Returns the path to the saved adapter directory."""
    raise NotImplementedError("Phase 6: Unsloth QLoRA from config; save adapter only.")
