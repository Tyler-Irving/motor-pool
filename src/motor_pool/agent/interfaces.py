"""Agent tool and planner protocols: the two duck-typed seams of the V2 loop.

Mirrors retrieval/interfaces.py. run_agent depends ONLY on these protocols (and
the ToolRegistry), never on ChatClient, retrieval, or httpx, so the whole loop is
exercised offline with a ScriptedPlanner and stub tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from ..schemas import AgentStep, PlannerDecision, ToolResult, ToolSpec

if TYPE_CHECKING:
    from .registry import ToolRegistry


@runtime_checkable
class Tool(Protocol):
    """A callable tool. name/description/args_model drive dispatch and the catalog."""

    name: str
    description: str
    args_model: type[BaseModel]

    def run(self, args: BaseModel) -> ToolResult: ...

    def spec(self) -> ToolSpec: ...


@runtime_checkable
class Planner(Protocol):
    """Decides the next action given the question and history. The testability seam."""

    def decide(
        self, question: str, history: list[AgentStep], registry: "ToolRegistry"
    ) -> PlannerDecision: ...
