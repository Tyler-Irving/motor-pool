"""Evaluation harness: base+RAG vs finetuned+RAG over a frozen eval set.

Deterministic checks (citation existence, schema validity, refusal-label match)
carry the headline numbers. A constrained, different-family LLM judge supplies
per-claim entailment for citation support and hallucination. The retriever,
top_k, and per-item chunk set are held identical across both systems.
"""
