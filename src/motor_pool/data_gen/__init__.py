"""Distillation data generation.

Builds (question + retrieved chunks -> grounded cited answer OR refusal) pairs
from a frontier teacher, then runs a deterministic-first validation gate. The
citation canonicalizer and the support checker here are shared with eval so that
"accepted by the validator" and "counted valid by eval" use identical logic.
"""

from __future__ import annotations

from .canonicalize import (
    canonicalize_citation,
    normalize_doc_id,
    normalize_page,
    normalize_token,
)

__all__ = [
    "canonicalize_citation",
    "normalize_doc_id",
    "normalize_page",
    "normalize_token",
]
