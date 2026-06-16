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
- Prefer explicit observability over silent degradation: when parsing/extraction falls back, loses fidelity, or cannot explain an output confidently, surface structured diagnostics to the MCP caller so the agent can notice, complain, and route feedback to the human user.

## Quality Gate
- Format with `ruff`: `uv run ruff format .`
- Lint with `ruff`: `uv run ruff check .`

## Guide Evolution
- AGENTS.md is a living guide.
- Record validated best practices and lessons learned during implementation.
- For optional service components requiring extra setup (for example GROBID), document the full setup process in README.md.

## Release Flow
- PyPI publishing is automated via GitHub Actions.
- This project does not use `$codex-commit-ai-stats` or any AI-stat commit suffix workflow; use normal git commit messages.
- Release order is mandatory:
  1. Merge all intended release changes into `main` (including version bump in `pyproject.toml`) and push `main`.
  2. Create a version tag from that `main` commit (for example `v0.1.2`).
  3. Push the tag to remote.
- GitHub Actions publishes to PyPI from the pushed tag; bumping version without pushing the matching tag means the release is incomplete.

## Lessons Learned
- Keep MCP protocol output isolated from logs: write logs to stderr only.
- Stdlib logging `extra` payloads must avoid reserved `LogRecord` keys like `message`; Python 3.13 raises `KeyError` when they are overwritten.
- Keep retrieval local-first and single-path by default: SQLite FTS5 + sidecar DocumentGraph. Do not reintroduce required external vector services.
- Treat page, artifact, reference, and citation nodes as first-class DocumentGraph citizens; avoid hiding them as unstructured metadata only.
- Source freshness is part of sidecar validity: READY documents must be considered stale when the source path is missing or source mtime/hash changes.
- Sidecar schema compatibility follows the breaking-change policy: incompatible `catalog.db` files are backed up and replaced with a fresh current-schema catalog instead of carrying legacy migrations.
- The MCP must start without Qdrant, FastEmbed, GROBID, or any other external retrieval service.
- GROBID is optional paper metadata/reference enrichment only; PDF paper ingest must still work without `GROBID_URL` and must surface skipped-enrichment diagnostics.
- Ingest finalization must be staged and validated: write new artifacts into a temporary document workspace, apply DB updates to a staging SQLite copy, validate the DocumentGraph, then atomically replace the active sidecar.
- Heavy PDF parsing, formula recovery, and Docling table/figure extraction should run in isolated worker processes with a structured timeout (`PDF_PARSE_TIMEOUT_SECONDS`) so hangs become agent-visible errors instead of wedging the MCP server.
- Prefer `library_explore` for cross-book/paper discovery, `document_explore` for one ingested document, and `document_node` for precise graph-node reads; explore outputs should include `why_included`, diagnostics, truncation notices, and ambiguity candidates.
- Keep `search_in_outline_node` and `read_outline_node` as focused chapter/section tools after an outline node is known.
- For PDF formulas, use a dedicated formula path (Docling formula enrichment + Pix2Text) and fail-fast when the formula engine is required but unavailable.
- Formula read tools should register rendered evidence images as artifact graph nodes so multimodal LLMs can revisit the exact visual evidence by stable node id.
- Catalog persistence must include cleanup: delete removed docs on scan, and use explicit sidecar cleanup tools for manual compaction when needed.
- Maintain a no-label PDF formula benchmark on sample papers; gate regressions with unresolved-rate, heuristic LaTeX validity rate, and parse stability rate.
- EPUB ingest must extract embedded images to local evidence files and expose explicit image tools (`epub_list_images`, `epub_read_image`) for multimodal LLM workflows.
- Parser-provided raw artifacts should be persisted under the document sidecar and exposed as artifact graph nodes for high-fidelity fallback reads.
- PDF ingest must extract figure/table images with page+bbox metadata and expose explicit tools (`pdf_list_images`, `pdf_read_image`) for multimodal LLM workflows.
- Prefer eager full parsing for ingest completeness. Optimize initialization with resource-aware parallelism, batching, and isolated workers instead of lazy extraction paths that can cause agents to miss content.
- Avoid duplicate high-cost parser passes: when Docling parse already has the source document in memory, reuse it for table/figure visual evidence instead of converting the same PDF again.
- PDF image/table/figure read tools must be read-only over persisted sidecar evidence. If evidence artifacts are missing, surface an actionable error and require force reingest instead of silently re-extracting during reads.
- Treat bocpy as an optional performance experiment until corpus benchmarks prove it beats the stdlib scheduler on target machines; do not make it a required runtime dependency without measured evidence. Current real parser stacks use C extensions that fail in CPython sub-interpreters (`lxml.etree`, `_pydantic_core`, PyMuPDF-related modules), so prefer stdlib process isolation for production ingest parallelism.
- Persistence is per-document-folder sidecar (`<doc_dir>/.mcp-ebook-read`); avoid global hidden data roots that are detached from source files.
- Expose explicit storage maintenance tools so LLMs can inspect and clean persistence safely (`storage_list_sidecars`, `storage_delete_document`, `storage_cleanup_sidecars`).
- Runtime diagnostics should be agent-visible and actionable: include enough structured warning/error detail, hints, and degradation metadata in MCP outputs so real reading sessions can reveal parser weaknesses and drive predictable iteration on sample PDFs/EPUBs.
