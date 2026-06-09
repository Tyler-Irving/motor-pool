"""Typed configuration models and a yaml loader.

Every runnable stage is config-driven. Each config file under configs/ maps to
one pydantic model here. `load_config(path, Model)` reads the yaml and validates
it, so a malformed config fails loudly at load time, not deep inside a run.
Unknown keys are rejected (extra="forbid"), so a typo'd key is caught immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schemas.retrieval import RrfConfig

T = TypeVar("T", bound=BaseModel)


class _Strict(BaseModel):
    """Base for config models. Rejects unknown keys so typos fail at load time."""

    model_config = ConfigDict(extra="forbid")


def load_config(path: str | Path, model: type[T]) -> T:
    """Load a yaml file and validate it against `model`."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return model.model_validate(data)


class LlmConfig(_Strict):
    """An OpenAI-compatible chat endpoint (local via ollama/vLLM, or hosted)."""

    model: str
    base_url: str = "http://localhost:11434/v1"  # ollama default
    api_key_env: str = ""  # name of env var holding the key; blank for local
    temperature: float = 0.0
    max_tokens: int = 1024


# ----------------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------------
class IngestionConfig(_Strict):
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50
    ocr_enabled: bool = False
    ocr_min_chars_per_page: int = Field(
        default=10, description="Below this many extractable chars a page is scanned."
    )
    per_doc_overrides: dict[str, dict] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------
class RetrievalConfig(_Strict):
    embedder_id: str = "BAAI/bge-base-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    doc_prefix: str = ""
    normalize: bool = True
    vector_backend: Literal["numpy"] = "numpy"
    rrf: RrfConfig = Field(default_factory=RrfConfig)


# ----------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------
class ValidatorConfig(_Strict):
    require_citation_per_step: bool = True
    min_claim_chunk_overlap: float = Field(
        default=0.15, description="Lexical overlap floor between a claim and its chunk."
    )
    reject_unsupported_numbers: bool = True


class DataGenConfig(_Strict):
    teacher: LlmConfig = Field(
        default_factory=lambda: LlmConfig(model="qwen2.5:7b")
    )
    target_pairs: int = 2500
    over_generation_factor: float = 1.8
    # The answerable fraction is derived from bucket_mix (the answerable_* buckets),
    # so it is not carried separately here.
    bucket_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "answerable_procedural": 0.40,
            "answerable_factual": 0.25,
            "hard_negative": 0.15,
            "out_of_scope": 0.12,
            "ambiguous": 0.08,
        }
    )
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)


# ----------------------------------------------------------------------------
# Training (QLoRA, 4080-safe). Keys mirror configs/training.yaml.
# ----------------------------------------------------------------------------
class ModelCfg(_Strict):
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str | None = None  # None lets Unsloth pick (bf16 on Ada)


class QuantCfg(_Strict):
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class LoraCfg(_Strict):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    random_state: int = 3407


class TrainCfg(_Strict):
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 1.0
    seed: int = 3407
    logging_steps: int = 10
    save_steps: int = 200
    output_dir: str = "outputs/adapter"


class DataCfg(_Strict):
    train_path: str = "data/train/train.jsonl"
    val_path: str = "data/train/val.jsonl"
    chat_template: str = "chatml"
    train_on_responses_only: bool = True
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"
    packing: bool = False


class SaveCfg(_Strict):
    save_adapter_only: bool = True
    merge_16bit: bool = False  # gated, reserved for future vLLM serving


class TrainingConfig(_Strict):
    model: ModelCfg = Field(default_factory=ModelCfg)
    quant: QuantCfg = Field(default_factory=QuantCfg)
    lora: LoraCfg = Field(default_factory=LoraCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    save: SaveCfg = Field(default_factory=SaveCfg)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
class EvalConfig(_Strict):
    # The system-under-test generator. base+RAG uses the student base model;
    # finetuned+RAG overrides this with the merged finetuned model.
    generator: LlmConfig = Field(
        default_factory=lambda: LlmConfig(model="qwen2.5:7b", max_tokens=1024)
    )
    # Different model family from the data-gen teacher (self-preference bias).
    judge: LlmConfig = Field(
        default_factory=lambda: LlmConfig(model="llama3.1:8b", max_tokens=512)
    )
    eval_set_path: str = "data/eval/heldout_v1.jsonl"
    retrieval_top_k: int = 6
    min_claim_overlap: float = 0.15
    bootstrap_n: int = 1000
    metrics: dict[str, bool] = Field(
        default_factory=lambda: {
            "hallucination": True,
            "citation": True,
            "refusal": True,
            "faithfulness": True,
        }
    )


# ----------------------------------------------------------------------------
# Agent (V2.0 shell)
# ----------------------------------------------------------------------------
class AgentConfig(_Strict):
    """The V2.0 agent shell: a planner model plus loop and retrieval bounds."""

    # qwen2.5:7b emits reliable flat JSON; qwen3 thinking mode broke nested
    # contracts in V1, so the planner stays on the proven model by default.
    planner: LlmConfig = Field(default_factory=lambda: LlmConfig(model="qwen2.5:7b"))
    max_steps: int = 6
    max_consecutive_errors: int = 2
    retrieval_top_k: int = 6
    enable_get_procedure: bool = False
    retrieval_config_path: str = "configs/retrieval.yaml"
