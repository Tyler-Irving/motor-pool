"""The pure, model-free agent loop.

run_agent drives planner decisions against the tool registry with a bounded step
budget and graceful error handling. It imports no model / retrieval / httpx code,
so the entire orchestration (including every failure branch) is exercised with a
ScriptedPlanner and stub tools. Guarantees: the loop always terminates (bounded by
config.max_steps); the only success exit is an explicit Finish; a stuck planner is
aborted after max_consecutive_errors; and every unknown-tool / invalid-args /
tool-exception / planner-no-parse is a RECORDED step in the trace, never an
exception out of run_agent.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..schemas import (
    AgentResult,
    AgentStep,
    AgentTrace,
    CallTool,
    Finish,
    PlannerError,
    Refusal,
    StopReason,
    ToolResult,
)

if TYPE_CHECKING:
    from ..config import AgentConfig
    from .interfaces import Planner
    from .registry import ToolRegistry


def _dispatch(call: CallTool, registry: "ToolRegistry") -> ToolResult:
    """Validate args and run the tool. Never raises: every failure is a ToolResult."""
    tool = registry.get(call.tool)
    if tool is None:
        return ToolResult(
            tool=call.tool, ok=False, error=f"unknown tool: {call.tool}", error_kind="unknown_tool"
        )
    try:
        args = tool.args_model.model_validate(call.args)
    except ValidationError as exc:
        return ToolResult(
            tool=call.tool, ok=False, error=str(exc)[:300], error_kind="invalid_args"
        )
    start = perf_counter()
    try:
        result = tool.run(args)
    except Exception as exc:  # tool blew up; keep the loop alive
        return ToolResult(
            tool=call.tool,
            ok=False,
            error=repr(exc)[:300],
            error_kind="tool_error",
            elapsed_ms=(perf_counter() - start) * 1000,
        )
    if result.elapsed_ms is None:
        result = result.model_copy(update={"elapsed_ms": (perf_counter() - start) * 1000})
    return result


def _budget_refusal() -> Refusal:
    return Refusal(
        reason="insufficient_context",
        message="step budget exhausted before an answer was reached",
    )


def run_agent(
    question: str,
    *,
    planner: "Planner",
    registry: "ToolRegistry",
    config: "AgentConfig",
) -> AgentResult:
    history: list[AgentStep] = []
    used_tools: list[str] = []
    consecutive_errors = 0
    t0 = perf_counter()
    answer = None
    stop_reason: StopReason = "budget_exhausted"

    for index in range(config.max_steps):
        try:
            decision = planner.decide(question, history, registry)
        except Exception as exc:  # belt-and-suspenders: a planner that escapes its own guard
            decision = PlannerError(message=repr(exc)[:300])
        if not isinstance(decision, (CallTool, Finish, PlannerError)):
            # A planner that RETURNS garbage (not raises) must not slip into dispatch.
            decision = PlannerError(message=f"planner returned non-decision: {type(decision).__name__!r}")

        if isinstance(decision, PlannerError):
            history.append(AgentStep(index=index, decision=decision))
            consecutive_errors += 1
            if consecutive_errors > config.max_consecutive_errors:
                stop_reason = "planner_error"
                answer = None
                break
            continue

        if isinstance(decision, Finish):
            history.append(AgentStep(index=index, decision=decision))
            stop_reason = "final"
            answer = decision.answer
            break

        # CallTool: dispatch is total (never raises); a failed tool call is recorded
        # and the loop continues. Only PlannerError counts toward the abort.
        result = _dispatch(decision, registry)
        history.append(AgentStep(index=index, decision=decision, result=result))
        if result.ok:
            consecutive_errors = 0
            if result.tool not in used_tools:
                used_tools.append(result.tool)
    else:
        # Loop ran to the step budget without an explicit Finish.
        answer = _budget_refusal()
        stop_reason = "budget_exhausted"

    trace = AgentTrace(steps=history, total_ms=(perf_counter() - t0) * 1000)
    return AgentResult(
        question=question,
        answer=answer,
        stop_reason=stop_reason,
        used_tools=used_tools,
        trace=trace,
    )
