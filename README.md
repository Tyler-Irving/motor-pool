# Motor Pool

**A grounded diagnostic assistant over public-domain HMMWV technical manuals: hybrid retrieval plus a behavior-only QLoRA fine-tune, measured with an honest evaluation harness.**

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/V1-complete-success)

The thesis: **retrieval handles facts, fine-tuning handles behavior.** Manual knowledge is never baked into model weights. The retriever supplies facts in-context at query time; the fine-tuned adapter only governs behavior, learning to cite the source section, return structured procedures, and refuse when the manuals do not cover the question.

## Results

base+RAG versus finetuned+RAG over a frozen, hand-verified, section-disjoint eval set of 136 items. Both systems share the same retriever, top_k, chunk set, and local 4-bit runtime, so the only variable is the LoRA adapter. No output-format enforcement at generation time, so this measures whether the model produces valid cited output on its own. Values are percentages.

| System | Schema-Valid | Refusal F1 | Over-Refuse | Cite-Support | Hallucination | Faithfulness |
| --- | --- | --- | --- | --- | --- | --- |
| base+RAG | 52.2 | 13.8 | 4.7 | 81.7 | 9.7 | 94.7 |
| **finetuned+RAG** | **89.7** | **89.8** | 4.7 | 81.6 | 11.1 | 93.7 |

The behavior-only fine-tune moves exactly the behaviors it targets and little else:

- **Structured-output adherence: 52 to 90 percent.** The model learns to emit valid cited JSON on its own.
- **Refusal F1: 14 to 90 percent** (recall 8 to 88, precision 50 to 92). It refuses when the manual does not cover the question, without becoming refusal-happy (over-refusal stays flat at 4.7 percent).
- **Citation support and faithfulness are essentially unchanged,** because retrieval supplies the facts for both systems.

That is the architecture thesis in numbers.

## How it works

- **Hybrid retrieval.** BM25 (bm25s) plus dense embeddings (bge-base-en-v1.5), combined with reciprocal rank fusion. Dense search is brute-force cosine over a numpy array; the corpus is small enough to need no vector database. Every chunk carries citation metadata (source TM, paragraph, page) so answers point back to the manual.
- **Behavior-only fine-tune.** QLoRA via Unsloth on a Qwen2.5-7B-Instruct base (Apache 2.0), sized for a single RTX 4080 (16GB): 4-bit base, r16, ~17 minutes to train. Training data is self-distilled with rejection sampling and validated so every target is derivable only from its in-context chunks; a validation pass rejects any answer that introduces a fact or citation absent from those chunks.
- **Honest evaluation.** Deterministic checks (schema-valid, refusal precision/recall/F1, citation existence) carry the headline. The entailment judge is a different model family (Llama-3.1-8B) from the data-generation teacher, and was validated against a three-annotator panel (92 percent agreement, Cohen kappa 0.85) so judged metrics are read as a mild, consistent overestimate.
- **A clean V2 seam.** The retriever is a stateless, tool-callable interface, `retrieve(query, *, top_k, filters) -> list[RetrievedChunk]`, that touches no model weights. V2 will wrap it as one tool among deterministic tools.

## Corpus

V1 uses TM 9-2320-280-10, the HMMWV Operator's Manual (421 pages): a public-domain US Government work (17 USC 105) under Distribution Statement A, approved for public release. It ingests into 201 procedure-level chunks across 111 numbered paragraphs, each with a page-accurate citation. No manual PDF is committed; `corpus/manifest.yaml` pins the source URL and sha256, and `motor-pool download` fetches and verifies it. Scoping the corpus to the operator level also sharpens the refusal boundary (operator versus unit and depot maintenance).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Heavy, phase-specific dependencies live in optional groups.

```bash
uv sync                       # core deps
uv sync --extra retrieve      # bm25s + embeddings
uv sync --extra train         # Unsloth QLoRA stack

motor-pool download           # fetch corpus PDF, verify sha256
motor-pool ingest             # parse and chunk into indexes/chunks.jsonl
motor-pool index              # build the dense + bm25 indexes
motor-pool query "How do I check the engine oil level?"   # retrieve cited chunks
motor-pool final-eval         # base+RAG vs finetuned+RAG, emit the results table
```

## Layout

```
configs/      yaml config per stage (retrieval, ingestion, data_gen, training, eval)
corpus/       manifest + download script (PDFs gitignored)
data/         frozen eval set (committed); generated training pairs (gitignored)
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

## License

MIT (see `LICENSE`). The base model Qwen2.5-7B-Instruct is Apache 2.0; the fine-tuning artifact is a LoRA adapter kept separate from the base weights. The technical manuals are public-domain US Government works under Distribution Statement A, and no manual PDF is committed to this repository.
