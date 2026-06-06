"""Results-table emission: markdown + json for the base+RAG vs finetuned+RAG run."""

from __future__ import annotations

import json
from pathlib import Path

from motor_pool.schemas import MetricScores

# (field, column label). Hallucination is split into the answerable and
# should-refuse subsets because the failure mode differs (fabricating vs failing
# to refuse).
_COLUMNS = [
    ("schema_valid_rate", "Schema-Valid"),
    ("hallucination_answerable", "Halluc(ans)"),
    ("hallucination_should_refuse", "Halluc(refuse)"),
    ("citation_exists_rate", "Cite-Exists"),
    ("citation_supported_rate", "Cite-Support"),
    ("valid_citation_rate", "Valid-Cite"),
    ("refusal_precision", "Refuse-P"),
    ("refusal_recall", "Refuse-R"),
    ("refusal_f1", "Refuse-F1"),
    ("over_refusal_rate", "Over-Refuse"),
    ("faithfulness", "Faithfulness"),
]


def _cell(score: MetricScores, key: str) -> str:
    value = getattr(score, key) * 100
    if key in score.ci:
        low, high = score.ci[key]
        return f"{value:.1f} [{low * 100:.0f}-{high * 100:.0f}]"
    return f"{value:.1f}"


def emit_table(scores: list[MetricScores], *, out_dir: Path) -> str:
    """Write results.md and results.json. Returns the markdown path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps([s.model_dump() for s in scores], indent=2), encoding="utf-8"
    )

    header = ["System", "N"] + [label for _, label in _COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for score in scores:
        row = [score.system, str(score.n)] + [_cell(score, k) for k, _ in _COLUMNS]
        lines.append("| " + " | ".join(row) + " |")

    md = (
        "# Results: base+RAG vs finetuned+RAG\n\n"
        "All values are percentages. Bracketed values are 95 percent bootstrap "
        "confidence intervals. Retrieval is held identical across systems.\n\n"
        + "\n".join(lines)
        + "\n"
    )
    md_path = out_dir / "results.md"
    md_path.write_text(md, encoding="utf-8")
    return str(md_path)
