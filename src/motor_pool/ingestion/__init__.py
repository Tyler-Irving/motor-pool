"""TM PDF ingestion: parse, detect structure, chunk at the procedure level.

The anchor -280 TMs are born-digital, so `pdf_text` is the primary path and
`ocr` is a wired fallback expected to be a no-op. Chunking is at the numbered
paragraph (the procedure-level unit), with tables and warning blocks handled
specially and troubleshooting MALFUNCTION blocks kept whole (Phase 3).
"""
