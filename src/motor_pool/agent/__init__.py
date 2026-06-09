"""The V2 agent: routes a question through tool calls to a grounded, cited answer.

V2.0 is the agent SHELL. It wraps the V1 retriever as tool #1 and is the seam the
future deterministic tools (STE/ICE lookup, parts lookup, fault-tree traversal)
register into. The split from V1 carries over: retrieval (and the deterministic
tools) handle facts; the agent only routes, and the finetuned cite/structure/
refuse behavior carries over for the final answer. No tool data lives in weights.

The loop (run_agent) is a pure function over two Protocol seams, Planner and Tool,
so the whole orchestration is testable with no live model (ScriptedPlanner) and no
network.
"""

from __future__ import annotations

from .interfaces import Planner, Tool
from .loop import run_agent
from .planner import LlmPlanner, ScriptedPlanner
from .registry import ToolRegistry, build_registry
from .tools import GetProcedureTool, RetrieveTool

__all__ = [
    "run_agent",
    "Tool",
    "Planner",
    "ToolRegistry",
    "build_registry",
    "ScriptedPlanner",
    "LlmPlanner",
    "RetrieveTool",
    "GetProcedureTool",
]
