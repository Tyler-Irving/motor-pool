"""V2.0 gate: LlmPlanner without a live model, via ChatClient(transport=MockTransport).

Exactly the test_llm.py idiom: a canned {"choices":[{"message":{"content": <json>}}]}
response. Confirms the planner parses a call_tool and a finish, and degrades to a
PlannerError on unparseable output.
"""

from __future__ import annotations

import httpx
from test_retriever_contract import _retriever

from motor_pool.agent.planner import LlmPlanner
from motor_pool.agent.registry import build_registry
from motor_pool.llm import ChatClient
from motor_pool.schemas import CallTool, Finish, ModelAnswer, PlannerError


def _planner(content: str) -> LlmPlanner:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return LlmPlanner(ChatClient(model="m", transport=httpx.MockTransport(handler)))


def _registry():
    return build_registry(_retriever(), default_top_k=3)


def test_llm_planner_emits_call_tool() -> None:
    content = '{"kind":"call_tool","tool":"retrieve","args":{"query":"oil"},"rationale":"need facts"}'
    decision = _planner(content).decide("how do I check oil?", [], _registry())
    assert isinstance(decision, CallTool)
    assert decision.tool == "retrieve"
    assert decision.args == {"query": "oil"}


def test_llm_planner_emits_finish() -> None:
    content = (
        '{"kind":"finish","answer":{"answer_type":"answer","summary":"check the dipstick",'
        '"steps":[{"order":1,"text":"pull the dipstick","cited":[1]}],"cited":[1]}}'
    )
    decision = _planner(content).decide("how do I check oil?", [], _registry())
    assert isinstance(decision, Finish)
    assert isinstance(decision.answer, ModelAnswer)
    assert decision.answer.summary == "check the dipstick"


def test_llm_planner_degrades_to_planner_error() -> None:
    decision = _planner("not json at all").decide("q", [], _registry())
    assert isinstance(decision, PlannerError)
