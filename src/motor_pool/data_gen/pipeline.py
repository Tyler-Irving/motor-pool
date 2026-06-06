"""Data-generation pipeline: sample -> retrieve -> teach -> validate -> persist.

Over-generates against the bucket mix, runs the live retriever for every question
(never injecting the gold chunk), has the teacher produce a target, and runs the
deterministic-first validation gate. Writes train/val jsonl plus a rejection
report so the data quality is auditable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from motor_pool.llm import ChatClient
from motor_pool.retrieval.interfaces import Retriever
from motor_pool.schemas import Chunk, Provenance, TrainingRecord

from .question_sampler import QuestionFn, sample_questions
from .teacher import generate_target
from .validate import Supports, validate_record


def generate_dataset(
    pool_chunks: list[Chunk],
    retriever: Retriever,
    teacher_client: ChatClient,
    supports: Supports,
    question_fn: QuestionFn,
    *,
    bucket_mix: dict[str, float],
    target_pairs: int,
    over_generation_factor: float = 1.8,
    require_citation_per_step: bool = True,
    reject_unsupported_numbers: bool = True,
    top_k: int = 6,
    timestamp: str = "",
    log=lambda msg: None,
) -> tuple[list[TrainingRecord], dict]:
    """Generate and validate distillation records. Returns (records, report)."""
    samples = sample_questions(
        pool_chunks,
        bucket_mix=bucket_mix,
        target_n=int(target_pairs * over_generation_factor),
        question_fn=question_fn,
    )

    # Pass 1: teach every sample. The teacher model stays loaded the whole pass,
    # so there is no per-item reload (the judge runs in pass 2). This matters a
    # lot when teacher and judge cannot co-reside in VRAM.
    log(f"sampled {len(samples)} questions; teaching ...")
    taught: list[tuple[str, str, list, object]] = []
    for i, (question, bucket) in enumerate(samples, start=1):
        chunks = retriever.retrieve(question, top_k=top_k)
        target = generate_target(question, chunks, client=teacher_client)
        taught.append((question, bucket, chunks, target))
        if i % 25 == 0 or i == len(samples):
            log(f"taught {i}/{len(samples)}")

    # Pass 2: validate. The judge model stays loaded for this whole pass.
    log("validating with the judge ...")
    records: list[TrainingRecord] = []
    rejected: Counter = Counter()
    for question, bucket, chunks, target in taught:
        if target is None:
            rejected["schema_invalid"] += 1
            continue
        record = TrainingRecord(
            question=question,
            retrieved_chunks=chunks,
            target=target,
            provenance=Provenance(
                teacher_model=teacher_client.model, gen_timestamp=timestamp, bucket=bucket
            ),
        )
        result = validate_record(
            record,
            supports,
            require_citation_per_step=require_citation_per_step,
            reject_unsupported_numbers=reject_unsupported_numbers,
        )
        if not result.passed:
            rejected[result.reasons[0].split(":")[0]] += 1
            continue
        records.append(record)
        if len(records) >= target_pairs:
            break

    report = {
        "sampled": len(samples),
        "schema_valid": sum(1 for t in taught if t[3] is not None),
        "kept": len(records),
        "rejected": dict(rejected),
        "by_bucket": dict(Counter(r.provenance.bucket for r in records)),
    }
    return records, report


def write_dataset(
    records: list[TrainingRecord],
    train_path: Path,
    val_path: Path,
    *,
    val_every: int = 10,
) -> tuple[int, int]:
    """Split records into train/val (every val_every-th to val) and write jsonl."""
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    n_train = n_val = 0
    with open(train_path, "w", encoding="utf-8") as ftrain, open(val_path, "w", encoding="utf-8") as fval:
        for i, record in enumerate(records):
            line = record.model_dump_json() + "\n"
            if i % val_every == val_every - 1:
                fval.write(line)
                n_val += 1
            else:
                ftrain.write(line)
                n_train += 1
    return n_train, n_val


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
