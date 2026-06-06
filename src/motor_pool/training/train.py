"""QLoRA trainer (Unsloth on PEFT/bitsandbytes/TRL), config-driven, 4080-safe.

Loads the 4-bit base, attaches a LoRA adapter, trains response-only on the
distillation data, and saves the adapter (not the merged model). unsloth/trl are
imported lazily so the package imports without the train extra installed.
"""

from __future__ import annotations

import json
from pathlib import Path

from motor_pool.config import TrainingConfig


def _render(tokenizer):
    def render(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    return render


def train(config: TrainingConfig) -> str:
    """Run QLoRA training. Returns the saved adapter directory."""
    # Unsloth must be imported before trl/transformers/peft for its optimizations.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    from .dataset import build_dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.base_model_id,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
        use_rslora=config.lora.use_rslora,
        random_state=config.lora.random_state,
    )

    dataset = build_dataset(config.data.train_path).map(_render(tokenizer))
    # Pre-flight: the response marker must appear in the rendered text, or
    # response-only masking silently masks everything and loss is zero.
    if config.data.response_part not in dataset[0]["text"]:
        raise RuntimeError(
            f"response marker {config.data.response_part!r} not found in the rendered "
            "chat text; check the chat template and instruction/response parts."
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=config.model.max_seq_length,
            per_device_train_batch_size=config.train.per_device_train_batch_size,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            warmup_ratio=config.train.warmup_ratio,
            num_train_epochs=config.train.num_train_epochs,
            learning_rate=config.train.learning_rate,
            logging_steps=config.train.logging_steps,
            optim=config.train.optim,
            weight_decay=config.train.weight_decay,
            lr_scheduler_type=config.train.lr_scheduler_type,
            max_grad_norm=config.train.max_grad_norm,
            seed=config.train.seed,
            output_dir=config.train.output_dir,
            report_to="none",
        ),
    )
    if config.data.train_on_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=config.data.instruction_part,
            response_part=config.data.response_part,
        )
        _assert_labels_present(trainer)

    trainer.train()

    out = Path(config.train.output_dir)
    model.save_pretrained(str(out))  # adapter only
    tokenizer.save_pretrained(str(out))
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "base_model_id": config.model.base_model_id,
                "lora": config.lora.model_dump(),
                "train": config.train.model_dump(),
                "seed": config.train.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(out)


def _assert_labels_present(trainer) -> None:
    """After response-only masking, a sample must have some unmasked (>=0) labels."""
    try:
        collated = trainer.data_collator([trainer.train_dataset[0]])
        labels = collated["labels"][0].tolist()
    except Exception:
        return  # cannot introspect on this version; rely on the marker pre-check
    if all(label == -100 for label in labels):
        raise RuntimeError(
            "all labels are -100 after response-only masking; the instruction/response "
            "markers do not match the chat template (training loss would be zero)."
        )
