# Motor Pool

A grounded diagnostic assistant over public-domain vehicle technical manuals.
The anchor vehicle is the HMMWV, using public US Army Technical Manuals (TMs)
that carry Distribution Statement A (approved for public release).

This is V1 of a two-stage project. V1 builds the retriever and a fine-tuned
model whose only job is behavior: cite the source section, return structured
procedures, and refuse when the manuals do not cover the question. V2 will wrap
the retriever as one tool among deterministic tools. V2 is not built here.

## Design

Retrieval handles facts. Fine-tuning handles behavior only. Manual knowledge is
never stored in model weights. The retriever supplies the facts in-context at
query time; the fine-tuned adapter governs citing, structuring, and refusing.
Every retrieved chunk carries citation metadata (source TM, paragraph or
section, page) so answers can point back to the manual.

Two consequences of that split:

- The retriever is a clean, tool-callable interface (`retrieve(query, *, top_k,
  filters) -> list[RetrievedChunk]`). It is stateless and touches no model
  weights. That is the seam V2 builds on.
- The fine-tuning data is built so every target is derivable only from the
  chunks provided in-context, and a validation pass rejects any answer that
  introduces a fact or citation absent from those chunks.

## Corpus

V1 uses one TM from the M998 (HMMWV) baseline family:

- TM 9-2320-280-10, Operator's Manual (421 pages).

It is a US Government work (17 USC 105) under Distribution Statement A, and is
born-digital with selectable text (OCR is wired as a fallback but is not needed
in practice). The PDF is never committed. `corpus/manifest.yaml` lists it with
its source URL and a sha256 pinned at download time, and `motor-pool download`
fetches it into `corpus/pdfs/` (gitignored).

The TM uses classic chapter / section / paragraph numbering, not Work Packages,
so the procedure-level chunk unit is the numbered paragraph (for example
2-104.1), with its parent section as context.

The unit-maintenance manual (TM 9-2320-280-20-1) was evaluated and deferred to
V2: its chapter-2 diagnostic flowcharts and letter-spaced text do not ingest
cleanly with the prose chunker. Keeping the corpus at the operator level also
gives a crisp scope boundary for the refusal behavior (operator versus unit and
depot maintenance).

## Stack

- Python, src layout, managed with uv.
- Hybrid retrieval: BM25 (bm25s) plus dense embeddings (bge-base-en-v1.5),
  combined with reciprocal rank fusion. Dense search is brute-force cosine over
  a numpy array; the corpus is small enough that this needs no vector database.
- Pydantic for every schema.
- QLoRA fine-tuning via Unsloth on PEFT, bitsandbytes, and TRL, config-driven
  through yaml. Base model: Qwen2.5-7B-Instruct (Apache 2.0). Defaults are sized
  for a single RTX 4080 (16GB): 4-bit base, gradient checkpointing, sequence
  length 2048.

## Layout

```
configs/      yaml config per stage (retrieval, ingestion, data_gen, training, eval)
corpus/       manifest + download script (PDFs gitignored)
data/         frozen eval set (committed); generated training pairs (gitignored)
indexes/      build artifacts: chunks.jsonl, dense embeddings, bm25 index (gitignored)
outputs/      adapters, eval tables, run logs (gitignored)
src/motor_pool/
  schemas/    the shared Pydantic contract
  ingestion/  PDF parse, structure detection, procedure-level chunking
  retrieval/  embedder, bm25, vector store, RRF, HybridRetriever, interfaces
  data_gen/   question sampling, teacher client, validation, canonicalization
  training/   chat-template dataset, QLoRA trainer
  eval/       judge, metrics, runner, results table
  inference/  prompt formatting, grounded model
  cli.py      command line entrypoint
tests/        schema, RRF, canonicalization, and config tests
```

## Install

Requires uv. The default install is light (pydantic, pyyaml, typer, numpy).
Heavy and phase-specific dependencies live in optional groups and are installed
per stage.

```
uv sync                       # core deps
uv sync --extra ingest        # Phase 1: PDF parsing
uv sync --extra retrieve      # Phase 2: bm25s + embeddings
uv sync --extra train         # Phase 6: Unsloth QLoRA stack
```

## Usage

```
motor-pool download           # fetch corpus PDFs, verify sha256
motor-pool ingest             # parse and chunk into indexes/chunks.jsonl
motor-pool index              # build the dense + bm25 indexes
motor-pool query "..."        # retrieve chunks (and answer, once trained)
motor-pool gen-data           # build distillation training pairs
motor-pool train              # config-driven QLoRA
motor-pool eval               # base+RAG vs finetuned+RAG, emit results table
```

## Evaluation

The deliverable is one table comparing base+RAG against finetuned+RAG over a
frozen, hand-verified eval set, with the retriever held identical across both
systems. Metrics: hallucination rate (split into answerable and should-refuse),
citation accuracy (exists and supports), refusal accuracy (precision, recall,
F1, over-refusal), and faithfulness. The table is produced in Phase 7.

```
(results table goes here)
```

## Model and license notes

- Base model Qwen2.5-7B-Instruct is Apache 2.0. The 7B-Instruct size
  specifically is Apache 2.0; some other Qwen2.5 sizes are not.
- The fine-tuning artifact is a LoRA adapter, kept separate from the base
  weights. Loading it requires the base model.
- The TMs are public-domain US Government works under Distribution Statement A.
  No manual PDF is committed to this repository.

## Build status

The phased build is tracked in the project plan.

Phase 0 (skeleton) is complete: schemas, RRF, citation canonicalization, and
config loaders are implemented and tested.

Phase 1 (corpus + ingestion) is complete for the operator manual. `motor-pool
download` fetches and sha256-verifies the PDF; `motor-pool ingest` parses it into
201 procedure-level chunks across 111 numbered paragraphs (Chapters 1-3), each
with a citation whose printed page label is recovered from page geometry.
Citations were cross-checked against an independent PDF text extractor.

Phase 2 (hybrid retrieval) is complete. `motor-pool index` builds a bm25s lexical
index and bge-base-en-v1.5 dense embeddings; `motor-pool query` fuses them with
reciprocal rank fusion and prints cited results. The `Retriever` interface (the
V2 tool seam) is frozen: a stateless `retrieve(query, *, top_k, filters)`
returning cited chunks, plus a separate `ProcedureFetcher`.

Phase 3 (unit-maintenance manual) was resolved by scoping it out: TM
9-2320-280-20-1 ingests at about 45 percent messy chunks (letter-spaced text and
multi-column diagnostic flowcharts), so V1 keeps the clean 201-chunk operator
corpus and defers the maintenance manual to a dedicated V2 parser.

Phase 4 (eval scorer + held-out set) is complete. `motor-pool build-eval-set`
builds a frozen, section-disjoint eval set (answerable questions from held-out
paragraphs with gold sections, plus curated refusals). `motor-pool eval` runs a
system over it and emits the results table: refusal precision/recall/F1,
over-refusal, citation existence and support, hallucination (split by answerable
vs should-refuse), and faithfulness, with bootstrap confidence intervals.
Teacher and judge are local models via ollama through one OpenAI-compatible
client (a different family for the judge). The judge (Llama-3.1-8B) was validated
against a three-annotator panel on 40 entailment pairs: 92 percent agreement,
Cohen kappa 0.85, with a slight lean toward over-accepting (86 percent precision,
100 percent recall), so judged metrics are read as a mild overestimate while the
deterministic metrics do not depend on it. The full base+RAG baseline runs end
to end (generator Qwen2.5-7B, judge Llama-3.1-8B). Data generation and training
remain typed stubs.
