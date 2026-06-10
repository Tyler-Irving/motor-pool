"""V2.0 gate: RetrieveTool / registry over the SAME stub backends as
test_retriever_contract.py (reuse, not reinvent). No model, no network.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_retriever_contract import _retriever

from motor_pool.agent.registry import build_registry
from motor_pool.agent.tools import GetProcedureArgs, GetProcedureTool, RetrieveArgs, RetrieveTool
from motor_pool.schemas import ToolSpec


def test_retrieve_tool_formats_citations() -> None:
    tool = RetrieveTool(_retriever(), default_top_k=3)
    res = tool.run(RetrieveArgs(query="widgets"))
    assert res.ok
    assert "[C1]" in res.content and "[C2]" in res.content
    assert len(res.data["chunks"]) == 3
    # data carries resolvable citations for the trace / GUI
    assert res.data["chunks"][0]["citation"]["source_doc_id"] == "TM-9-2320-280-10"


def test_retrieve_tool_respects_top_k() -> None:
    tool = RetrieveTool(_retriever(), default_top_k=2)
    assert len(tool.run(RetrieveArgs(query="widgets")).data["chunks"]) == 2  # default
    assert len(tool.run(RetrieveArgs(query="widgets", top_k=1)).data["chunks"]) == 1  # override


def test_retrieve_args_reject_filters() -> None:
    # The planner-facing surface is query + top_k only; a hallucinated metadata
    # filter that could silently zero out retrieval is rejected, not honored.
    with pytest.raises(ValidationError):
        RetrieveArgs(query="widgets", source_doc_id="TM 9-2320-280-10")


def test_registry_specs_shape() -> None:
    registry = build_registry(_retriever(), default_top_k=4)
    specs = registry.specs()
    assert [s.name for s in specs] == ["retrieve"]
    assert isinstance(specs[0], ToolSpec)
    assert set(specs[0].args_schema["properties"]) == {"query", "top_k"}


def test_get_procedure_stub() -> None:
    tool = GetProcedureTool(_retriever())
    ok = tool.run(GetProcedureArgs(source_doc_id="TM-9-2320-280-10", parent_id="sec"))
    assert ok.ok
    assert len(ok.data["chunks"]) == 3
    bad = tool.run(GetProcedureArgs(source_doc_id="TM-9-2320-280-10", parent_id="nope"))
    assert bad.ok is False
    assert bad.error == "no_procedure"
