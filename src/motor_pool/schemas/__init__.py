"""Public schema contract for Motor Pool.

This package is pydantic-only with zero heavy dependencies. It is the shared
contract consumed by ingestion, retrieval, data_gen, training, eval, and the CLI.
"""

from __future__ import annotations

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
