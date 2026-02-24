# mcp-ebook-read MVP Status

## Overview
This document summarizes the current MVP implementation status, what has been validated, and what should be done next.

## Scope Decisions (Locked)
- Retrieval stack: **Qdrant + FastEmbed only** (single path, no BM25 fallback).
- Transport: **stdio** MCP server.
- Ingest profiles:
  - `book`: PDF (Docling) + EPUB (EbookLib/lxml)
  - `paper`: PDF (Docling + GROBID, fail-fast)
- Fail-fast policy: no silent downgrade paths for required components.

## Implemented MVP Components

### 1. Server and Tool Surface
- MCP server entrypoint with unified response envelope (`ok/data/error/trace_id`).
- Implemented tools:
  - `library_scan`
  - `document_ingest_pdf_book`
  - `document_ingest_epub_book`
  - `document_ingest_pdf_paper`
  - `search`
  - `search_in_outline_node`
  - `read`
  - `read_outline_node`
  - `pdf_list_images`
  - `pdf_read_image`
  - `epub_list_images`
  - `epub_read_image`
  - `storage_list_sidecars`
  - `storage_delete_document`
  - `storage_cleanup_sidecars`
  - `get_outline`
  - `render_pdf_page`

### 2. Parsing Pipeline
- PDF parser using Docling + Pix2Text (formula-specific).
- EPUB parser using EbookLib + lxml.
- GROBID client for `document_ingest_pdf_paper` metadata/outline enrichment.
- `paper` profile behavior is fail-fast when GROBID is unavailable/fails.
- PDF book parser now includes:
  - TOC cleanup/de-noise for outline quality.
  - Section-level page range assignment for chunk locators.
  - Formula marker replacement with Pix2Text LaTeX candidates (with textual fallback + unresolved marker).

### 3. Storage and Indexing
- SQLite catalog/chunk persistence (per-folder sidecar: `<doc_dir>/.mcp-ebook-read/catalog.db`).
- SQLite maintenance includes:
  - Removing records for files deleted from scan root.
  - Sidecar maintenance tools for listing, document deletion, and cleanup.
  - Optional explicit compaction via `storage_cleanup_sidecars(..., compact_catalog=true)`.
- Qdrant vector index integration.
- Stable UUID point ID mapping for Qdrant compatibility.
- FastEmbed embeddings for both ingest indexing and query retrieval.

### 4. Evidence and Readback
- Page rendering for PDF (`render_pdf_page`) to local evidence images.
- PDF figure extraction and readback (`pdf_list_images`, `pdf_read_image`) with page/bbox and local image paths.
  - Extraction mode is on-demand (triggered on first PDF image tool call, not during ingest).
- EPUB image extraction and readback (`epub_list_images`, `epub_read_image`) with chapter mapping and local image paths.
- Locator-based `read` with configurable before/after context window.
- Outline-node direct read (`read_outline_node`) for chapter-oriented reading.
- Outline-node constrained retrieval (`search_in_outline_node`) for chapter-scoped search.

### 5. Project and Ops Setup
- `uv`-based dependency management and scripts.
- README includes:
  - One-command Docker setup for Qdrant and GROBID.
  - Claude Code JSON MCP config examples (`uvx`).
  - Codex TOML MCP config examples.
- `.gitignore` excludes:
  - `extern/`
  - `.mcp-ebook-read/`
  - `tests/samples/`

## Validated Results

### Code Quality
- `uv run ruff check .` passes.
- Unit tests pass (`pytest`): current suite validates storage and read path basics.
- Added no-label formula benchmark tooling for sample PDFs:
  - `uv run mcp-ebook-formula-benchmark --samples-dir tests/samples/pdf-papers`
  - Tracks unresolved-rate, heuristic LaTeX validity rate, and parse stability rate.

### Sample Data Validation (`tests/samples`)
- `book` profile:
  - PDF books: ingest success.
  - EPUB books: ingest success.
  - Search + read flow validated with real data.
- `paper` profile:
  - Works with GROBID configured and reachable.
  - For large papers, default timeout can be insufficient; increasing `GROBID_TIMEOUT_SECONDS` (for example `120`) resolves observed timeout failure.

## Known Limitations (Current MVP)
- No OCR/VLM fallback path yet.
- No HTTP/SSE transport yet (stdio only).
- No end-to-end integration tests are checked into CI with real sample files.
- Docling/GROBID output merge is minimal (title/outline/metadata blending); advanced structure alignment is not implemented.
- Search is vector-only; no reranking/hybrid strategy.
- PDF page-range assignment is heuristic and depends on TOC/heading quality in source documents.
- Formula extraction quality depends on Pix2Text coverage; complex inline formulas may remain unresolved.

## Priority Next Steps

### P0 (Stability and Test Depth)
1. Add integration tests that run against controlled sample docs (book/paper) with explicit environment guards.
2. Add observability around ingest stages (structured timing/log fields per stage).
3. Harden error messages and error-code coverage for parser/index/network failures.

### P1 (Retrieval Quality)
1. Improve chunking strategy (section-aware and size-aware chunk boundaries).
2. Add optional metadata filters in vector search (doc type/profile/section).
3. Add retrieval diagnostics endpoint/tool (embedding model, collection stats, top-k debug info).

### P2 (Document Fidelity)
1. Improve PDF locator fidelity further with bbox-level anchors (page-level already implemented).
2. Improve paper-mode merge between Docling body and GROBID structure.
3. Add explicit citation/reference extraction output for paper workflows.

### P3 (Operational Improvements)
1. Add optional HTTP transport mode (without changing stdio default).
2. Add a reproducible local run script (env bootstrap + service checks).
3. Pin Docker image versions by digest for stronger reproducibility.

## Suggested Acceptance Criteria for MVP Completion
- `book` and `paper` sample ingest success rate reaches target with configured services and documented timeouts.
- Search/read/render workflows are stable across all sample sets.
- Startup docs are sufficient for one-command service setup and MCP client wiring.
- No fallback paths violate the single-path + fail-fast policy.
