"""The grounded model: a local Qwen2.5-7B (4-bit) with optional LoRA adapter.

Loads via unsloth and generates a grounded answer or refusal using the exact
same prompt contract as everything else (inference.prompt). The base model
(no adapter) and the finetuned model (base + adapter) run through this one path,
so the eval comparison isolates the LoRA. unsloth/torch are imported lazily.
"""

from __future__ import annotations

from pydantic import TypeAdapter

from motor_pool.inference.prompt import build_messages, resolve_model_output
from motor_pool.llm import extract_json
from motor_pool.schemas import ModelOutput, RetrievedChunk, TrainingTarget

_OUTPUT_ADAPTER = TypeAdapter(ModelOutput)


class GroundedModel:
    """A loaded local model (base, or base + LoRA adapter) that emits TrainingTargets."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 1024) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens

    @classmethod
    def load(cls, model_name: str, *, max_seq_length: int = 2048, max_new_tokens: int = 1024) -> "GroundedModel":
        """Load a base model id, or an adapter directory (which pulls in its base)."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
        return cls(model, tokenizer, max_new_tokens)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> TrainingTarget | None:
        """Generate a grounded answer or refusal. Returns None if output does not parse."""
        import torch

        system, user = build_messages(question, chunks)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                input_ids=inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        try:
            return resolve_model_output(_OUTPUT_ADAPTER.validate_python(extract_json(text)), chunks)
        except Exception:
            return None
