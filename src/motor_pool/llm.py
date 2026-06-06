"""A small provider-agnostic chat client (OpenAI-compatible Chat Completions).

One client talks to a local model server (ollama, vLLM, LM Studio) or a hosted
API, selected by base_url + model. Used by the data-gen teacher and the eval
judge. Structured output is requested as JSON and validated against a Pydantic
schema with a retry, so a model that returns slightly-off JSON gets one nudge.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_JSON_RE = re.compile(r"\{.*\}", re.S)


class LlmError(RuntimeError):
    """Raised when the model cannot produce valid output."""


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating code fences/prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if match:
            return json.loads(match.group(0))
        raise


@dataclass
class ChatClient:
    """OpenAI-compatible chat client. `transport` is for tests (an httpx transport)."""

    model: str
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0
    transport: object | None = None

    @classmethod
    def from_config(cls, cfg) -> "ChatClient":
        """Build from an LlmConfig, resolving the api key from its named env var."""
        api_key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        return cls(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def complete_json(
        self, system: str, user: str, schema: type[T], *, retries: int = 2
    ) -> T:
        """Return a validated `schema` instance, retrying on invalid JSON."""
        prompt = user
        last = ""
        for _ in range(retries + 1):
            text = self.complete(system, prompt, json_mode=True)
            try:
                return schema.model_validate(extract_json(text))
            except (ValidationError, json.JSONDecodeError) as exc:
                last = str(exc)[:200]
                prompt = (
                    f"{user}\n\nReturn ONLY a JSON object matching the schema. "
                    f"Your previous reply was invalid: {last}"
                )
        raise LlmError(
            f"{self.model} did not return valid JSON after {retries + 1} attempts: {last}"
        )

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        kwargs: dict = {"base_url": self.base_url, "timeout": self.timeout}
        if self.transport is not None:
            kwargs["transport"] = self.transport
        with httpx.Client(**kwargs) as client:
            response = client.post(path, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
