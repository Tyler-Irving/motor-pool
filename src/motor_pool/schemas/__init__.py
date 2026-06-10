"""Public schema contract for Motor Pool.

This package is pydantic-only with zero heavy dependencies. It is the shared
contract consumed by ingestion, retrieval, data_gen, training, eval, and the CLI.
"""

from __future__ import annotations

from .agent import (
    AgentResult,
    AgentStep,
    AgentTrace,
    CallTool,
    ErrorKind,
    Finish,
    PlannerDecision,
    PlannerError,
    StopReason,
    ToolResult,
    ToolSpec,
)
from .answer import (
    GroundedAnswer,
    ModelAnswer,
    ModelOutput,
    ModelStep,
    ProcedureStep,
    Refusal,
    TrainingTarget,
)
from .chunk import Chunk, ContentType, RetrievedChunk
from .citation import Citation, Locator, ParagraphLocator
from .datagen import Bucket, Provenance, TrainingRecord, ValidationResult
from .eval import EvalItem, EvalLabel, MetricScores, RefuseReason
from .manifest import CorpusManifest, IndexManifest, TmEntry
from .retrieval import RetrievalFilters, RrfConfig

__all__ = [
    # agent
    "AgentResult",
    "AgentStep",
    "AgentTrace",
    "CallTool",
    "ErrorKind",
    "Finish",
    "PlannerDecision",
    "PlannerError",
    "StopReason",
    "ToolResult",
    "ToolSpec",
    # citation
    "Citation",
    "Locator",
    "ParagraphLocator",
    # chunk
    "Chunk",
    "ContentType",
    "RetrievedChunk",
    # answer
    "GroundedAnswer",
    "ModelAnswer",
    "ModelOutput",
    "ModelStep",
    "ProcedureStep",
    "Refusal",
    "TrainingTarget",
    # retrieval
    "RetrievalFilters",
    "RrfConfig",
    # datagen
    "Bucket",
    "Provenance",
    "TrainingRecord",
    "ValidationResult",
    # eval
    "EvalItem",
    "EvalLabel",
    "MetricScores",
    "RefuseReason",
    # manifest
    "CorpusManifest",
    "IndexManifest",
    "TmEntry",
]
