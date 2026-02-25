# mcp-ebook-read Agent Guide

## Scope
- Build a Codex-usable MCP focused on reading EPUB and PDF books/papers.
- Keep the implementation minimal first, then iterate.

## Product Mission
- Build toward as-lossless-as-possible parsing of EPUB/PDF books and papers so LLMs can understand source materials with high fidelity.
- Make the MCP a practical reading companion that collaborates deeply with human users and supports large-scale reading workloads.

## Typical Scenarios
- Use outline-first navigation plus selected chapter reading to summarize content, answer reading questions, and guide users through difficult sections.
- Help users bring books/papers into real engineering projects, then evaluate incremental value through reading, formula understanding, and code implementation.

## Environment
- Python package/dependency workflow uses `uv`.
- Prefer `uv run ...` for all local commands.
- Prefer replying to users in Simplified Chinese in chat responses only.
- Prefer English for project files, documentation, and code comments.

## Constraints
- `extern/` is reference-only.
- Do not modify files under `extern/`.
- Do not track `extern/` in git.

## Decision Policy
- For any uncertain decision, stop and ask the user before proceeding.
- Apply fail-fast principle across the project.
- Use a single implementation path for each feature; avoid redundant fallback/bypass paths unless explicitly approved.
- During iteration, always optimize for LLM best practices when using this MCP to read books and papers.
- If compatibility and breaking changes conflict, prefer breaking changes.
- When shipping breaking changes, remove parameters/fields that are no longer used; do not keep deprecated placeholders.
- Continuously simplify the codebase: remove redundant branches, stale outputs, and dead compatibility layers.
- MCP tool names must encode file format and use-case explicitly to minimize LLM confusion; avoid generic multi-mode tool names.

## Quality Gate
- Format with `ruff`: `uv run ruff format .`
- Lint with `ruff`: `uv run ruff check .`

## Guide Evolution
- AGENTS.md is a living guide.
- Record validated best practices and lessons learned during implementation.
- For service components requiring extra setup (for example Qdrant), document the full setup process in README.md.

## Release Flow
- PyPI publishing is automated via GitHub Actions.
- When a new version tag is pushed from `main`, the release workflow automatically publishes that version to PyPI.

## Lessons Learned
- Keep MCP protocol output isolated from logs: write logs to stderr only.
- Keep retrieval single-path: Qdrant + FastEmbed only; avoid adding BM25 fallback.
- Qdrant container must publish host port 6333 (or set QDRANT_URL explicitly) before starting the MCP server.
- Runtime must pass README-documented env vars explicitly; `document_ingest_pdf_paper` always requires `GROBID_URL` (and usually `GROBID_TIMEOUT_SECONDS`).
- Prefer chapter-oriented tools (`search_in_outline_node`, `read_outline_node`) over free-form global retrieval for book reading tasks.
- For PDF formulas, use a dedicated formula path (Docling formula enrichment + Pix2Text) and fail-fast when the formula engine is required but unavailable.
- Catalog persistence must include cleanup: delete removed docs on scan, and use explicit sidecar cleanup tools for manual compaction when needed.
- Maintain a no-label PDF formula benchmark on sample papers; gate regressions with unresolved-rate, heuristic LaTeX validity rate, and parse stability rate.
- EPUB ingest must extract embedded images to local evidence files and expose explicit image tools (`epub_list_images`, `epub_read_image`) for multimodal LLM workflows.
- PDF ingest must extract figure/table images with page+bbox metadata and expose explicit tools (`pdf_list_images`, `pdf_read_image`) for multimodal LLM workflows.
- Persistence is per-document-folder sidecar (`<doc_dir>/.mcp-ebook-read`); avoid global hidden data roots that are detached from source files.
- Expose explicit storage maintenance tools so LLMs can inspect and clean persistence safely (`storage_list_sidecars`, `storage_delete_document`, `storage_cleanup_sidecars`).
