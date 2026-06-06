"""Grounded inference: load the 4-bit base + adapter, answer from retrieved chunks.

The adapter governs behavior (cite, structure, refuse); the retriever supplies
the facts in-context. The inference prompt byte-matches the training format, or
the learned behavior degrades.
"""
