"""Reciprocal rank fusion.

Fuses ranked lists by position only, so BM25 scores and cosine similarities
never need to be normalized against each other. k=60 is the standard default
(Cormack et al.). Ties break deterministically by chunk id so runs reproduce.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ranking.

    Args:
        ranked_lists: each inner list is ids ordered best-first.
        k: the RRF constant. Must be positive.
        weights: optional per-list weights, defaults to 1.0 each.

    Returns:
        (chunk_id, fused_score) pairs ordered best-first.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked, start=1):  # 1-based ranks
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))

    # Highest score first; ties broken by id ascending for determinism.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
