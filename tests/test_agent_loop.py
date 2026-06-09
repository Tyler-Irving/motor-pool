"""V2.0 gate: the agent loop, driven entirely by a ScriptedPlanner (no model).

Every loop branch (happy path, budget, unknown tool, invalid args, tool exception,
planner error + consecutive abort, trace serialization) is exercised with zero
network, the agent analog of test_llm.py's MockTransport.
"""

from __future__ import annotations

from pydantic import BaseModel

from motor_pool.agent.loop import run_agent
from motor_pool.agent.planner import ScriptedPlanner
from motor_pool.agent.registry import ToolRegistry
from motor_pool.agent.tools import ToolBase
from motor_pool.config import AgentConfig
from motor_pool.schemas import (
    AgentResult,
    CallTool,
    Finish,
    ModelAnswer,
    ModelStep,
    PlannerError,
    Refusal,
    ToolResult,
)


class _EchoArgs(BaseModel):
    text: str


class _StubTool(ToolBase):
    name = "echo"
    description = "echo back text"
    args_model = _EchoArgs

    def run(self, args: _EchoArgs) -> ToolResult:
        return ToolResult(tool=self.name, ok=True, content=f"echo: {args.text}", data={"chunks": []})


class _BoomArgs(BaseModel):
    pass


class _RaisingTool(ToolBase):
    name = "boom"
    description = "always raises"
    args_model = _BoomArgs

    def run(self, args: _BoomArgs) -> ToolResult:
        raise RuntimeError("kaboom")


def _registry(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _finish(summary: str = "done") -> Finish:
    return Finish(
        answer=ModelAnswer(summary=summary, steps=[ModelStep(order=1, text="do it", cited=[1])], cited=[1])
    )


def test_call_then_finish() -> None:
    planner = ScriptedPlanner([CallTool(tool="echo", args={"text": "hi"}), _finish("the answer")])
    result = run_agent("q", planner=planner, registry=_registry(_StubTool()), config=AgentConfig())
    assert result.stop_reason == "final"
    assert isinstance(result.answer, ModelAnswer)
    assert result.answer.summary == "the answer"
    assert result.used_tools == ["echo"]
    assert len(result.trace.steps) == 2
    assert result.trace.steps[0].result.ok
    assert result.trace.steps[0].result.elapsed_ms is not None


def test_budget_exhausted_refuses() -> None:
    planner = ScriptedPlanner([CallTool(tool="echo", args={"text": "x"})] * 5)
    result = run_agent("q", planner=planner, registry=_registry(_StubTool()), config=AgentConfig(max_steps=2))
    assert result.stop_reason == "budget_exhausted"
    assert isinstance(result.answer, Refusal)
    assert len(result.trace.steps) == 2


def test_unknown_tool_graceful() -> None:
    planner = ScriptedPlanner([CallTool(tool="nope", args={}), _finish()])
    result = run_agent("q", planner=planner, registry=_registry(_StubTool()), config=AgentConfig())
    step0 = result.trace.steps[0]
    assert step0.result.ok is False
    assert step0.result.error_kind == "unknown_tool"
    assert result.stop_reason == "final"  # the loop continued to the finish
    assert result.used_tools == []


def test_invalid_args_graceful() -> None:
    planner = ScriptedPlanner([CallTool(tool="echo", args={}), _finish()])  # missing 'text'
    result = run_agent("q", planner=planner, registry=_registry(_StubTool()), config=AgentConfig())
    assert result.trace.steps[0].result.error_kind == "invalid_args"
    assert result.stop_reason == "final"


def test_tool_exception_caught() -> None:
    planner = ScriptedPlanner([CallTool(tool="boom", args={}), _finish()])
    result = run_agent("q", planner=planner, registry=_registry(_RaisingTool()), config=AgentConfig())
    res0 = result.trace.steps[0].result
    assert res0.ok is False and res0.error_kind == "tool_error"
    assert "kaboom" in res0.error
    assert result.stop_reason == "final"


def test_planner_error_and_consecutive_abort() -> None:
    planner = ScriptedPlanner([PlannerError(message="boom")] * 5)
    result = run_agent(
        "q", planner=planner, registry=_registry(_StubTool()), config=AgentConfig(max_consecutive_errors=2)
    )
    assert result.stop_reason == "planner_error"
    assert result.answer is None
    # The 3rd consecutive error trips the abort (count > max_consecutive_errors=2).
    assert len(result.trace.steps) == 3


class _BadPlanner:
    """A misbehaving planner that RETURNS a non-decision instead of raising."""

    def decide(self, question, history, registry):
        return None


def test_non_decision_return_becomes_planner_error() -> None:
    # run_agent must stay total: a planner returning garbage is recorded as a
    # PlannerError step, never an exception out of run_agent.
    result = run_agent(
        "q", planner=_BadPlanner(), registry=_registry(_StubTool()), config=AgentConfig(max_consecutive_errors=0)
    )
    assert result.stop_reason == "planner_error"
    assert result.answer is None
    assert len(result.trace.steps) == 1
    assert result.trace.steps[0].decision.kind == "error"


def test_trace_json_roundtrip() -> None:
    planner = ScriptedPlanner([CallTool(tool="echo", args={"text": "hi"}), _finish()])
    result = run_agent("q", planner=planner, registry=_registry(_StubTool()), config=AgentConfig())
    back = AgentResult.model_validate_json(result.model_dump_json())
    assert back == result
    assert back.trace.trace_version == 1
    assert back.trace.total_ms is not None
