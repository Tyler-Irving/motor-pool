"""Phase 4 gate: the lexical floor and the shared supports() composition."""

from __future__ import annotations

from motor_pool.data_gen.validate import lexical_overlap, make_supports


def test_lexical_overlap() -> None:
    assert lexical_overlap("remove the locking pin", "remove locking pin now") == 1.0
    assert lexical_overlap("torque the lug nuts", "the sky is blue") == 0.0


def test_floor_short_circuits_judge() -> None:
    calls: list[tuple[str, str]] = []

    def judge(claim: str, text: str) -> bool:
        calls.append((claim, text))
        return True

    supports = make_supports(judge, min_overlap=0.5)
    # Almost no overlap: judge is not consulted, result is False.
    assert supports("torque the lug nuts to spec", "the sky is blue today") is False
    assert calls == []
    # Enough overlap: judge is consulted and its verdict is returned.
    assert supports("remove the locking pin", "remove the locking pin now") is True
    assert len(calls) == 1
