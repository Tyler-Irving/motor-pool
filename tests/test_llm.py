"""Phase 4 gate: the OpenAI-compatible chat client (no network, via MockTransport)."""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from motor_pool.llm import ChatClient, LlmError, extract_json


class _Verdict(BaseModel):
    supported: bool


def _client(content: str) -> ChatClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return ChatClient(model="m", transport=httpx.MockTransport(handler))


def test_extract_json_plain() -> None:
    assert extract_json('{"supported": true}') == {"supported": True}


def test_extract_json_code_fence() -> None:
    assert extract_json('```json\n{"supported": false}\n```') == {"supported": False}


def test_extract_json_embedded_in_prose() -> None:
    assert extract_json('Here is the answer: {"supported": true} done') == {"supported": True}


def test_complete_json_validates() -> None:
    assert _client('{"supported": true}').complete_json("s", "u", _Verdict).supported is True


def test_complete_json_retries_then_raises() -> None:
    with pytest.raises(LlmError):
        _client("not json at all").complete_json("s", "u", _Verdict, retries=1)


def test_from_config_resolves_api_key(monkeypatch) -> None:
    monkeypatch.setenv("MY_KEY", "secret")

    class Cfg:
        model = "m"
        base_url = "http://x/v1"
        api_key_env = "MY_KEY"
        temperature = 0.0
        max_tokens = 256

    client = ChatClient.from_config(Cfg())
    assert client.api_key == "secret"
    assert client.model == "m"
