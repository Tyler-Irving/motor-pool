"""Concrete agent tools.

RetrieveTool wraps the V1 retriever as tool #1; its text output reuses
inference.prompt.format_context so the planner sees the established [C#] context
block (not a reimplementation), and it carries the serialized chunks in
ToolResult.data so the trace and a future GUI can resolve citations.
GetProcedureTool is a thin V2.1 seam over the ProcedureFetcher.

format_context is imported lazily inside run() so the agent package imports
without pulling the inference/llm stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ..schemas import ToolResult, ToolSpec

if TYPE_CHECKING:
    from ..retrieval.interfaces import ProcedureFetcher, Retriever


class ToolBase:
    """Mixin giving spec() from the concrete tool's name/description/args_model."""

    name: str
    description: str
    args_model: type[BaseModel]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema=self.args_model.model_json_schema(),
        )


class RetrieveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int | None = None
    # The retriever's metadata filters (source_doc_id, content_type) are deliberately
    # NOT exposed to the planner in V2.0: a small local model misformats the doc id
    # (e.g. "TM 9-2320-280-10") and the exact-match filter then silently zeroes out
    # retrieval. The RetrievalFilters seam stays in the retriever for the future
    # deterministic tools to use intentionally.


class RetrieveTool(ToolBase):
    """Search the operator manual; returns cited [C#] context. The V1 RAG tool."""

    name = "retrieve"
    description = (
        "Search the HMMWV operator technical manual for passages relevant to a "
        "query. Returns cited [C#] context passages. Use this for any factual or "
        "procedural question about the vehicle. Pass a focused natural-language query; "
        "optionally set top_k to control how many passages come back."
    )
    args_model = RetrieveArgs

    def __init__(self, retriever: "Retriever", *, default_top_k: int = 6) -> None:
        self._retriever = retriever
        self._default_top_k = default_top_k

    def run(self, args: RetrieveArgs) -> ToolResult:
        from motor_pool.inference.prompt import format_context

        chunks = self._retriever.retrieve(args.query, top_k=args.top_k or self._default_top_k)
        if not chunks:
            return ToolResult(
                tool=self.name, ok=True, content="(no matching passages)", data={"chunks": []}
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            content=format_context(chunks),
            data={"chunks": [c.model_dump(mode="json") for c in chunks]},
        )


class GetProcedureArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_doc_id: str
    parent_id: str


class GetProcedureTool(ToolBase):
    """Fetch the full ordered set of sibling chunks under a section. A V2.1 seam."""

    name = "get_procedure"
    description = (
        "Fetch the full ordered set of sibling chunks under a section / parent id, "
        "for reading a complete procedure rather than isolated passages."
    )
    args_model = GetProcedureArgs

    def __init__(self, fetcher: "ProcedureFetcher") -> None:
        self._fetcher = fetcher

    def run(self, args: GetProcedureArgs) -> ToolResult:
        chunks = self._fetcher.get_procedure(args.source_doc_id, args.parent_id)
        if not chunks:
            return ToolResult(
                tool=self.name,
                ok=False,
                content="(no such procedure)",
                error="no_procedure",
                error_kind="tool_error",
            )
        from motor_pool.inference.prompt import format_context

        return ToolResult(
            tool=self.name,
            ok=True,
            content=format_context(chunks),
            data={"chunks": [c.model_dump(mode="json") for c in chunks]},
        )
