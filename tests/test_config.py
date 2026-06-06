"""Phase 0 gate: every shipped config loads and validates against its model.

Also pins the extra="forbid" behavior so a typo'd key fails at load time rather
than being silently ignored.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from motor_pool.config import (
    DataGenConfig,
    EvalConfig,
    IngestionConfig,
    RetrievalConfig,
    TrainingConfig,
    load_config,
)

CONFIGS = [
    ("configs/ingestion.yaml", IngestionConfig),
    ("configs/retrieval.yaml", RetrievalConfig),
    ("configs/data_gen.yaml", DataGenConfig),
    ("configs/training.yaml", TrainingConfig),
    ("configs/eval.yaml", EvalConfig),
]


@pytest.mark.parametrize("path,model", CONFIGS)
def test_shipped_config_validates(path: str, model: type) -> None:
    load_config(path, model)


def test_training_config_key_values() -> None:
    t = load_config("configs/training.yaml", TrainingConfig)
    assert t.model.base_model_id == "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    assert t.lora.target_modules[0] == "q_proj"
    # The chatml marker must keep its trailing newline through yaml decoding.
    assert t.data.instruction_part == "<|im_start|>user\n"
    assert t.train.optim == "paged_adamw_8bit"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IngestionConfig.model_validate({"chunk_target_tokenz": 999})
