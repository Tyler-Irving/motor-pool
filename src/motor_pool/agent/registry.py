"""The tool registry: holds tools by name, renders their specs, and is the object
the planner and loop consult. build_registry wires the V1 retriever as tool #1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas import ToolSpec

if TYPE_CHECKING:
    from ..retrieval.interfaces import ProcedureFetcher, Retriever
    from .interfaces import Tool


class ToolRegistry:
    """An ordered, name-keyed set of tools. Registration order is the catalog order."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: "Tool") -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> "Tool | None":
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """The self-describing tool catalog, for the planner prompt and a future GUI."""
        return [tool.spec() for tool in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools


def build_registry(
    retriever: "Retriever",
    *,
    fetcher: "ProcedureFetcher | None" = None,
    default_top_k: int = 6,
    enable_get_procedure: bool = False,
) -> ToolRegistry:
    """Register the V1 retriever as tool #1 (and optionally the procedure-fetch stub)."""
    from .tools import GetProcedureTool, RetrieveTool

    registry = ToolRegistry().register(RetrieveTool(retriever, default_top_k=default_top_k))
    if enable_get_procedure and fetcher is not None:
        registry.register(GetProcedureTool(fetcher))
    return registry
