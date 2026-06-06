"""Phase 0 gate: reciprocal rank fusion correctness and determinism."""

from __future__ import annotations

import pytest

from motor_pool.retrieval.rrf import reciprocal_rank_fusion


def test_single_list_uses_one_based_ranks() -> None:
    fused = reciprocal_rank_fusion([["x"]], k=60)
    assert fused == [("x", 1.0 / 61)]


def test_known_fusion_with_tiebreak() -> None:
    a = ["d1", "d2", "d3"]
    b = ["d2", "d1", "d4"]
    fused = reciprocal_rank_fusion([a, b], k=60)
    ids = [doc for doc, _ in fused]
    # d1 and d2 score equally; ties break by id ascending. d3 before d4 likewise.
    assert ids == ["d1", "d2", "d3", "d4"]
    top_score = fused[0][1]
    assert top_score == pytest.approx(1.0 / 61 + 1.0 / 62)


def test_weights_shift_ranking() -> None:
    a = ["d1", "d2", "d3"]
    b = ["d2", "d1", "d4"]
    fused = reciprocal_rank_fusion([a, b], k=60, weights=[2.0, 1.0])
    assert fused[0][0] == "d1"
    assert fused[0][1] == pytest.approx(2.0 / 61 + 1.0 / 62)


def test_fuses_on_id_not_position() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60))
    # Both ids appear at rank 1 and rank 2 across the two lists, so scores are equal.
    assert fused["a"] == pytest.approx(fused["b"])


def test_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
