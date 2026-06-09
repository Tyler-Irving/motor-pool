"""Agent protocol and trace schemas: the V2 agent's structured contract.

V2 wraps the V1 retriever (and, later, deterministic tools) behind an agent that
plans tool calls and emits a grounded answer. These pydantic-only models are that
contract:

- the FLAT planner decision the model emits, a discriminated union on `kind`
  (call_tool / finish / error), kept single-level because V1 found small local
  models reliable on flat JSON but not nested;
- the per-step records (AgentStep) and a fully serializable AgentTrace; and
- AgentResult, what run_agent returns.

AgentResult.model_dump_json() is the exact payload a future GUI renders (see
docs/v2-gui-vision.md), so this module is also the GUI contract; it is versioned
via AgentTrace.trace_version. The final answer reuses the V1 ModelOutput ([C#]
1-based citations) so the agent speaks the same grounded-answer contract as the
rest of the system.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .answer import ModelOutput

ErrorKind = Literal["unknown_tool", "invalid_args", "tool_error"]
StopReason = Literal["final", "budget_exhausted", "planner_error"]


class ToolResult(BaseModel):
    """The outcome of one tool dispatch. Failures are values here, never exceptions."""

    tool: str
    ok: bool
    content: str = Field(
        default="", description="Text shown back to the planner (the [C#] context block)."
    )
    data: dict[str, Any] | None = Field(
        default=None, description="Structured payload for the trace / future GUI."
    )
    error: str | None = None
    error_kind: ErrorKind | None = None
    elapsed_ms: float | None = None


class ToolSpec(BaseModel):
    """A self-describing tool entry: name, description, JSON-schema of its args.

    Built from a tool's args_model.model_json_schema(); used to render the planner
    prompt and (later) a GUI tool catalog with argument forms.
    """

    name: str
    description: str
    args_schema: dict[str, Any]


class CallTool(BaseModel):
    """Planner decision: call a tool.

    FLAT (no nested object) so a small local model emits it reliably; per-tool arg
    typing is enforced deterministically at dispatch, not by the model.
    """

    kind: Literal["call_tool"] = "call_tool"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class Finish(BaseModel):
    """Planner decision: stop and answer. Reuses the V1 ModelOutput contract."""

    kind: Literal["finish"] = "finish"
    answer: ModelOutput


class PlannerError(BaseModel):
    """Planner decision: the planner could not produce a valid decision."""

    kind: Literal["error"] = "error"
    message: str = ""


# Discriminated union on `kind`. Validate a bare value with:
#   from pydantic import TypeAdapter
#   TypeAdapter(PlannerDecision).validate_python(obj)
PlannerDecision = Annotated[
    CallTool | Finish | PlannerError, Field(discriminator="kind")
]


class AgentStep(BaseModel):
    """One loop iteration: the planner's decision and (for a tool call) its result."""

    index: int
    decision: PlannerDecision
    result: ToolResult | None = None


class AgentTrace(BaseModel):
    """The full, serializable record of an agent run. Versioned for the GUI contract."""

    steps: list[AgentStep] = Field(default_factory=list)
    total_ms: float | None = None
    trace_version: int = 1


class AgentResult(BaseModel):
    """What run_agent returns: the final answer (or None), why it stopped, the trace."""

    question: str
    answer: ModelOutput | None
    stop_reason: StopReason
    used_tools: list[str] = Field(default_factory=list)
    trace: AgentTrace
