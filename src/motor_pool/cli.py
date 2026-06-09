"""Motor Pool command line interface.

One entrypoint per stage. Commands are stubbed in Phase 0 and wired to their
modules in later phases. `motor-pool --help` lists everything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    add_completion=False,
    help="Grounded diagnostic assistant over public-domain vehicle technical manuals.",
)


def _quiet_ml_logging(*, offline: bool = False) -> None:
    """Silence cosmetic startup chatter from the ML stack for clean CLI output.

    Progress bars and the HF-token nudge are disabled via env (which must be set
    before the libraries import), and the noisy loggers are raised to ERROR. This
    only declutters output for interactive use; it changes no behavior. Not used
    by `index`, where the embedding progress bar is useful build feedback.

    Pass offline=True only when every needed model is already cached (e.g. `query`,
    which runs after `index` has fetched the embedder). It pins HF to the local
    cache, which also drops the hub's unauthenticated-request warning. Leave it
    False for commands that may still need to download (e.g. `final-eval`).
    """
    import logging
    import os
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # transformers resets its own logger level on import, so a plain setLevel is
    # overridden; this env var sets its default verbosity before it imports.
    # (unsloth resets it again when it patches a model; GroundedModel.load
    # re-asserts this for the final-eval path.)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # Library-internal deprecation surfaced on every generate(); not actionable here.
    warnings.filterwarnings("ignore", message=r".*attention mask API.*", category=FutureWarning)
    for name in ("huggingface_hub", "transformers", "sentence_transformers", "bm25s", "torchao"):
        logging.getLogger(name).setLevel(logging.ERROR)


@app.command()
def download(
    only: list[str] = typer.Option([], "--only", help="Restrict to these TM numbers."),
    manifest: Path = typer.Option(Path("corpus/manifest.yaml"), help="Manifest path."),
    pdfs: Path = typer.Option(Path("corpus/pdfs"), help="Output directory for PDFs."),
) -> None:
    """Download corpus PDFs from the manifest and verify sha256."""
    from motor_pool.corpus import download_corpus

    result = download_corpus(manifest, pdfs, only=set(only) or None)
    for tm in result.tms:
        typer.echo(f"{tm.tm_number}: sha256={tm.sha256}")
    typer.echo("Pin any newly computed sha256 into the manifest.")


@app.command()
def ingest(
    only: list[str] = typer.Option([], "--only", help="Restrict to these TM numbers."),
    manifest: Path = typer.Option(Path("corpus/manifest.yaml"), help="Manifest path."),
    pdfs: Path = typer.Option(Path("corpus/pdfs"), help="Directory of downloaded PDFs."),
    out: Path = typer.Option(Path("indexes/chunks.jsonl"), help="Chunks output path."),
) -> None:
    """Parse and chunk the downloaded corpus into a chunks.jsonl."""
    from motor_pool.config import IngestionConfig, load_config
    from motor_pool.ingestion.pipeline import ingest_corpus

    config = load_config("configs/ingestion.yaml", IngestionConfig)
    chunks = ingest_corpus(manifest, pdfs, out, config=config, only=set(only) or None)
    typer.echo(f"ingested {len(chunks)} chunks -> {out}")


@app.command()
def index(
    indexes: Path = typer.Option(Path("indexes"), help="Index dir (reads chunks.jsonl)."),
    config_path: Path = typer.Option(Path("configs/retrieval.yaml"), "--config", help="Retrieval config."),
) -> None:
    """Build the dense and bm25 indexes from indexes/chunks.jsonl."""
    from datetime import datetime

    from motor_pool.config import RetrievalConfig, load_config
    from motor_pool.ingestion.pipeline import build_indexes, read_chunks_jsonl
    from motor_pool.retrieval.embedder_bge import BgeEmbedder

    config = load_config(config_path, RetrievalConfig)
    try:
        chunks = read_chunks_jsonl(indexes / "chunks.jsonl")
    except FileNotFoundError:
        typer.echo(f"{indexes / 'chunks.jsonl'} not found; run `motor-pool ingest` first.")
        raise typer.Exit(code=1)
    embedder = BgeEmbedder(
        config.embedder_id,
        query_prefix=config.query_prefix,
        doc_prefix=config.doc_prefix,
        normalize=config.normalize,
    )
    manifest = build_indexes(
        chunks,
        out_dir=indexes,
        embedder=embedder,
        normalize=config.normalize,
        corpus_sha256s=sorted({c.citation.source_pdf_sha256 for c in chunks}),
        built_at=datetime.now().isoformat(timespec="seconds"),
    )
    typer.echo(
        f"indexed {manifest.chunk_count} chunks with {manifest.embedder_id} "
        f"({manifest.dims}d) -> {indexes}"
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="The diagnostic question."),
    top_k: Optional[int] = typer.Option(None, help="Chunks to retrieve (default: config rrf.top_k)."),
    dedupe: bool = typer.Option(
        True, "--dedupe/--no-dedupe",
        help="Collapse multiple chunks of the same paragraph into one cited line.",
    ),
    indexes: Path = typer.Option(Path("indexes"), help="Index directory."),
    config_path: Path = typer.Option(Path("configs/retrieval.yaml"), "--config", help="Retrieval config."),
) -> None:
    """Retrieve the most relevant chunks for a question and print their sources."""
    _quiet_ml_logging(offline=True)
    from motor_pool.config import RetrievalConfig, load_config
    from motor_pool.retrieval.embedder_bge import BgeEmbedder
    from motor_pool.retrieval.hybrid_retriever import load_retriever

    config = load_config(config_path, RetrievalConfig)
    embedder = BgeEmbedder(
        config.embedder_id,
        query_prefix=config.query_prefix,
        doc_prefix=config.doc_prefix,
        normalize=config.normalize,
    )
    try:
        retriever = load_retriever(indexes, embedder, config.rrf)
    except FileNotFoundError:
        typer.echo(f"index not found under {indexes}; run `motor-pool index` first.")
        raise typer.Exit(code=1)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        typer.echo("no results")
        return
    if dedupe:
        # Results arrive sorted by score, so the first chunk seen for a paragraph
        # is its best; keep that one and count the rest as "+N more".
        groups: dict[tuple[str, str], list] = {}
        for result in results:
            key = (result.citation.source_doc_id, result.citation.locator.paragraph)
            groups.setdefault(key, []).append(result)
        display = [(group[0], len(group) - 1) for group in groups.values()]
    else:
        display = [(result, 0) for result in results]
    for rank, (result, extra) in enumerate(display, start=1):
        locator = result.citation.locator
        more = f"  (+{extra} more chunk{'s' if extra != 1 else ''})" if extra else ""
        typer.echo(
            f"[{rank}] {result.score:.4f}  SOURCE: {result.citation.source_doc_id} "
            f"para {locator.paragraph} p.{result.citation.tm_page_label}{more}"
        )
        typer.echo(f"      {result.text.splitlines()[0][:96]}")


@app.command(name="gen-data")
def gen_data(
    limit: Optional[int] = typer.Option(None, help="Cap target pairs (quick run)."),
    indexes: Path = typer.Option(Path("indexes"), help="Index directory."),
    data_config: Path = typer.Option(Path("configs/data_gen.yaml"), "--data-config"),
    retrieval_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--retrieval-config"),
    eval_config: Path = typer.Option(Path("configs/eval.yaml"), "--eval-config"),
    out_dir: Path = typer.Option(Path("data/train"), help="Output directory."),
    heldout: Path = typer.Option(Path("data/eval/heldout_sections.json")),
) -> None:
    """Generate and validate distillation training pairs (excludes eval sections)."""
    import json
    from datetime import datetime

    from motor_pool.config import DataGenConfig, EvalConfig, RetrievalConfig, load_config
    from motor_pool.data_gen.pipeline import generate_dataset, write_dataset, write_report
    from motor_pool.data_gen.validate import make_supports
    from motor_pool.eval.eval_set import llm_question_fn
    from motor_pool.eval.judge import make_judge
    from motor_pool.ingestion.pipeline import read_chunks_jsonl
    from motor_pool.llm import ChatClient
    from motor_pool.retrieval.embedder_bge import BgeEmbedder
    from motor_pool.retrieval.hybrid_retriever import load_retriever

    dcfg = load_config(data_config, DataGenConfig)
    rcfg = load_config(retrieval_config, RetrievalConfig)
    ecfg = load_config(eval_config, EvalConfig)
    held = set(json.loads(heldout.read_text(encoding="utf-8"))) if heldout.exists() else set()
    chunks = read_chunks_jsonl(indexes / "chunks.jsonl")
    pool = [c for c in chunks if c.citation.locator.paragraph not in held]
    typer.echo(
        f"pool: {len(pool)} chunks "
        f"({len({c.citation.locator.paragraph for c in pool})} paragraphs); "
        f"excluded {len(held)} held-out eval sections"
    )

    embedder = BgeEmbedder(
        rcfg.embedder_id, query_prefix=rcfg.query_prefix,
        doc_prefix=rcfg.doc_prefix, normalize=rcfg.normalize,
    )
    retriever = load_retriever(indexes, embedder, rcfg.rrf)
    teacher_client = ChatClient.from_config(dcfg.teacher)
    supports = make_supports(make_judge(ChatClient.from_config(ecfg.judge)), ecfg.min_claim_overlap)
    # Questions come from a fast model at higher temperature (variety matters more
    # than power here); the teacher is reserved for the targets.
    question_fn = llm_question_fn(
        ChatClient(model="qwen2.5:7b", base_url=dcfg.teacher.base_url, max_tokens=80, temperature=0.6)
    )
    target = limit or dcfg.target_pairs

    typer.echo(
        f"generating up to {target} pairs "
        f"(teacher={dcfg.teacher.model}, judge={ecfg.judge.model}) ..."
    )
    records, report = generate_dataset(
        pool, retriever, teacher_client, supports, question_fn,
        bucket_mix=dcfg.bucket_mix, target_pairs=target,
        over_generation_factor=dcfg.over_generation_factor,
        require_citation_per_step=dcfg.validator.require_citation_per_step,
        reject_unsupported_numbers=dcfg.validator.reject_unsupported_numbers,
        top_k=ecfg.retrieval_top_k,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        log=typer.echo,
    )
    n_train, n_val = write_dataset(records, out_dir / "train.jsonl", out_dir / "val.jsonl")
    write_report(report, out_dir / "datagen_report.json")
    typer.echo(f"kept {report['kept']} pairs {report['by_bucket']} -> {out_dir} (train {n_train}, val {n_val})")
    typer.echo(f"rejections: {report['rejected']}")


@app.command(name="final-eval")
def final_eval(
    adapter: Path = typer.Option(Path("outputs/adapter"), help="Finetuned LoRA adapter dir."),
    base_model: Optional[str] = typer.Option(None, help="Base model id (default: training.yaml)."),
    limit: Optional[int] = typer.Option(None, help="Score only the first N eval items."),
    indexes: Path = typer.Option(Path("indexes"), help="Index directory."),
    eval_config: Path = typer.Option(Path("configs/eval.yaml"), "--eval-config"),
    retrieval_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--retrieval-config"),
    training_config: Path = typer.Option(Path("configs/training.yaml"), "--training-config"),
    out_dir: Path = typer.Option(Path("outputs/final"), help="Results output directory."),
) -> None:
    """Run base+RAG and finetuned+RAG via the local model; emit the two-row table."""
    _quiet_ml_logging()
    import gc

    import torch

    from motor_pool.config import EvalConfig, RetrievalConfig, TrainingConfig, load_config
    from motor_pool.data_gen.validate import make_supports
    from motor_pool.eval.eval_set import read_eval_set
    from motor_pool.eval.judge import make_judge
    from motor_pool.eval.report import emit_table
    from motor_pool.eval.runner import build_corpus_text, run_eval
    from motor_pool.ingestion.pipeline import read_chunks_jsonl
    from motor_pool.inference.grounded_model import GroundedModel
    from motor_pool.llm import ChatClient
    from motor_pool.retrieval.embedder_bge import BgeEmbedder
    from motor_pool.retrieval.hybrid_retriever import load_retriever

    ecfg = load_config(eval_config, EvalConfig)
    rcfg = load_config(retrieval_config, RetrievalConfig)
    tcfg = load_config(training_config, TrainingConfig)
    items = read_eval_set(Path(ecfg.eval_set_path))
    if limit:
        items = items[:limit]
    chunks = read_chunks_jsonl(indexes / "chunks.jsonl")
    corpus_text = build_corpus_text(chunks)
    embedder = BgeEmbedder(
        rcfg.embedder_id, query_prefix=rcfg.query_prefix,
        doc_prefix=rcfg.doc_prefix, normalize=rcfg.normalize,
    )
    retriever = load_retriever(indexes, embedder, rcfg.rrf)
    supports = make_supports(make_judge(ChatClient.from_config(ecfg.judge)), ecfg.min_claim_overlap)

    scores = []
    for system, model_name in [
        ("base+RAG", base_model or tcfg.model.base_model_id),
        ("finetuned+RAG", str(adapter)),
    ]:
        typer.echo(f"loading {system} ({model_name}) ...")
        model = GroundedModel.load(model_name, max_seq_length=tcfg.model.max_seq_length)
        typer.echo(f"running {system} over {len(items)} items ...")
        scores.append(
            run_eval(
                system, items, retriever, model.generate, corpus_text, supports,
                top_k=ecfg.retrieval_top_k, bootstrap_n=ecfg.bootstrap_n, log=typer.echo,
            )
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    md_path = emit_table(scores, out_dir=out_dir)
    typer.echo(f"results -> {md_path}\n")
    typer.echo(Path(md_path).read_text(encoding="utf-8"))


@app.command()
def train(
    config_path: Path = typer.Option(Path("configs/training.yaml"), "--config"),
) -> None:
    """Run config-driven QLoRA training, saving a LoRA adapter."""
    from motor_pool.config import TrainingConfig, load_config
    from motor_pool.training.train import train as run_train

    config = load_config(config_path, TrainingConfig)
    typer.echo(f"training {config.model.base_model_id} on {config.data.train_path} ...")
    out = run_train(config)
    typer.echo(f"adapter saved -> {out}")


@app.command(name="build-eval-set")
def build_eval_set_cmd(
    indexes: Path = typer.Option(Path("indexes"), help="Index dir (reads chunks.jsonl)."),
    out: Path = typer.Option(Path("data/eval/heldout_v1.jsonl"), help="Eval set output."),
    model: Optional[str] = typer.Option(
        None, help="ollama model to back-generate natural questions (else templates)."
    ),
) -> None:
    """Build the frozen, section-disjoint evaluation set from the corpus."""
    import json
    from collections import Counter

    from motor_pool.eval.eval_set import build_eval_set, llm_question_fn, write_eval_set
    from motor_pool.ingestion.pipeline import read_chunks_jsonl

    chunks = read_chunks_jsonl(indexes / "chunks.jsonl")
    question_fn = None
    if model:
        from motor_pool.llm import ChatClient

        typer.echo(f"back-generating questions with {model} ...")
        question_fn = llm_question_fn(ChatClient(model=model, max_tokens=80))
    items, held = build_eval_set(chunks, question_fn=question_fn)
    write_eval_set(items, out)
    (out.parent / "heldout_sections.json").write_text(
        json.dumps(sorted(held), indent=2), encoding="utf-8"
    )
    buckets = dict(Counter(i.bucket for i in items))
    typer.echo(f"wrote {len(items)} eval items -> {out}  {buckets}")
    typer.echo(f"held out {len(held)} sections (data-gen must exclude these)")


@app.command(name="eval")
def eval_cmd(
    system: str = typer.Option("base+RAG", help="Label for the system under test."),
    model: Optional[str] = typer.Option(None, help="Override the generator model."),
    judge_model: Optional[str] = typer.Option(None, "--judge", help="Override the judge model."),
    limit: Optional[int] = typer.Option(None, help="Score only the first N items."),
    indexes: Path = typer.Option(Path("indexes"), help="Index directory."),
    eval_config: Path = typer.Option(Path("configs/eval.yaml"), "--eval-config"),
    retrieval_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--retrieval-config"),
    out_dir: Path = typer.Option(Path("outputs"), help="Results output directory."),
) -> None:
    """Run a system over the eval set and emit the results table."""
    from motor_pool.config import EvalConfig, RetrievalConfig, load_config
    from motor_pool.data_gen.validate import make_supports
    from motor_pool.eval.eval_set import read_eval_set
    from motor_pool.eval.judge import make_judge
    from motor_pool.eval.report import emit_table
    from motor_pool.eval.runner import build_corpus_text, run_eval
    from motor_pool.ingestion.pipeline import read_chunks_jsonl
    from motor_pool.inference.prompt import generate_grounded
    from motor_pool.llm import ChatClient
    from motor_pool.retrieval.embedder_bge import BgeEmbedder
    from motor_pool.retrieval.hybrid_retriever import load_retriever

    ecfg = load_config(eval_config, EvalConfig)
    rcfg = load_config(retrieval_config, RetrievalConfig)
    try:
        items = read_eval_set(Path(ecfg.eval_set_path))
    except FileNotFoundError:
        typer.echo(f"{ecfg.eval_set_path} not found; run `motor-pool build-eval-set` first.")
        raise typer.Exit(code=1)
    if limit:
        items = items[:limit]

    chunks = read_chunks_jsonl(indexes / "chunks.jsonl")
    corpus_text = build_corpus_text(chunks)
    embedder = BgeEmbedder(
        rcfg.embedder_id,
        query_prefix=rcfg.query_prefix,
        doc_prefix=rcfg.doc_prefix,
        normalize=rcfg.normalize,
    )
    try:
        retriever = load_retriever(indexes, embedder, rcfg.rrf)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    gen_cfg = ecfg.generator if model is None else ecfg.generator.model_copy(update={"model": model})
    judge_cfg = ecfg.judge if judge_model is None else ecfg.judge.model_copy(update={"model": judge_model})
    gen_client = ChatClient.from_config(gen_cfg)
    supports = make_supports(make_judge(ChatClient.from_config(judge_cfg)), ecfg.min_claim_overlap)

    def generate(question, retrieved_chunks):
        return generate_grounded(question, retrieved_chunks, client=gen_client)

    typer.echo(
        f"running {system} over {len(items)} items "
        f"(gen={gen_cfg.model}, judge={ecfg.judge.model}) ..."
    )
    scores = run_eval(
        system, items, retriever, generate, corpus_text, supports,
        top_k=ecfg.retrieval_top_k, bootstrap_n=ecfg.bootstrap_n, log=typer.echo,
    )
    md_path = emit_table([scores], out_dir=out_dir)
    typer.echo(f"results -> {md_path}\n")
    typer.echo(Path(md_path).read_text(encoding="utf-8"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
