"""Composition root for the agent CLI command.

Wires the V1 retriever + tools + LlmPlanner from config, keeping cli.py thin and
all heavy imports (retrieval, embedder, ChatClient) lazy here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AgentConfig, RetrievalConfig
    from .interfaces import Planner
    from .registry import ToolRegistry


def build_agent(
    indexes_dir: str | Path,
    agent_cfg: "AgentConfig",
    retrieval_cfg: "RetrievalConfig",
) -> tuple["Planner", "ToolRegistry"]:
    """Build (planner, registry) for a real run. Same retriever wiring as `query`."""
    from ..retrieval.embedder_bge import BgeEmbedder
    from ..retrieval.hybrid_retriever import load_retriever
    from .planner import LlmPlanner
    from .registry import build_registry

    embedder = BgeEmbedder(
        retrieval_cfg.embedder_id,
        query_prefix=retrieval_cfg.query_prefix,
        doc_prefix=retrieval_cfg.doc_prefix,
        normalize=retrieval_cfg.normalize,
    )
    retriever = load_retriever(Path(indexes_dir), embedder, retrieval_cfg.rrf)
    registry = build_registry(
        retriever,
        fetcher=retriever,
        default_top_k=agent_cfg.retrieval_top_k,
        enable_get_procedure=agent_cfg.enable_get_procedure,
    )
    planner = LlmPlanner.from_config(agent_cfg)
    return planner, registry
