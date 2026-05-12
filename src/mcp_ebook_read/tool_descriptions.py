"""LLM-facing MCP tool descriptions.

Keep wording explicit: these strings are part of the agent routing surface,
not incidental Python docstrings.
"""

from __future__ import annotations

LIBRARY_SCAN = (
    "Scan a local root for EPUB/PDF documents and register them in sidecar "
    "catalogs. Use this before doc_id-only tools after a fresh server restart."
)

STORAGE_LIST_SIDECARS = (
    "List per-folder .mcp-ebook-read sidecars under a root. Use this to "
    "rediscover known documents after restart and inspect local persistence."
)

STORAGE_DELETE_DOCUMENT = (
    "Delete one document from sidecar persistence and vector index. Use only "
    "when the user asks to remove persisted MCP state for a specific document."
)

STORAGE_CLEANUP_SIDECARS = (
    "Clean sidecar catalogs under a root by pruning missing documents, orphan "
    "artifacts, and optionally compacting SQLite catalogs."
)

DOCUMENT_INGEST_PDF_BOOK = (
    "Queue high-fidelity background ingest for a non-scanned PDF book. Use for "
    "book-length PDFs where outline-first chapter reading is expected."
)

DOCUMENT_INGEST_EPUB_BOOK = (
    "Queue high-fidelity background ingest for an EPUB book. Use for EPUB "
    "books before outline navigation, chapter reading, or EPUB image access."
)

DOCUMENT_INGEST_PDF_PAPER = (
    "Queue high-fidelity background ingest for a non-scanned PDF paper. Use "
    "for academic papers; GROBID enriches metadata while Docling remains the "
    "canonical page-aware structure parser."
)

DOCUMENT_INGEST_STATUS = (
    "Read current status for a background ingest job. Use after any "
    "document_ingest_* tool until status is succeeded or failed."
)

DOCUMENT_INGEST_LIST_JOBS = (
    "List recent ingest jobs for one document. Use when the latest job_id was "
    "lost or the user asks for ingest history."
)

DOCUMENT_AUTOTUNE_PDF_PARSER = (
    "Benchmark Docling PDF parser profiles on sampled pages and persist the "
    "best local performance profile. Use before long PDF ingest runs."
)

SEARCH = (
    "Global semantic search over indexed document chunks. Prefer "
    "search_in_outline_node for chapter-scoped book/paper reading after an "
    "outline node is known."
)

SEARCH_IN_OUTLINE_NODE = (
    "Search within one outline node or chapter. Use this for focused reading "
    "questions after get_outline identifies the relevant section."
)

READ = (
    "Read a chunk window around a locator returned by search. Use for precise "
    "local context, not for selecting chapters."
)

READ_OUTLINE_NODE = (
    "Read one outline node or chapter directly. Prefer this for chapter "
    "summaries and guided reading after get_outline."
)

EPUB_LIST_IMAGES = (
    "List extracted images from an EPUB book, optionally scoped to an outline "
    "node. Use before epub_read_image in multimodal reading workflows."
)

EPUB_READ_IMAGE = (
    "Return one EPUB image local path plus nearby text context. Use when a "
    "multimodal LLM needs to inspect a figure, diagram, or illustration."
)

PDF_LIST_IMAGES = (
    "List extracted PDF images, optionally scoped to an outline node. Use for "
    "general PDF visual evidence; use pdf_list_figures or pdf_list_tables for "
    "Docling-detected figure/table objects."
)

PDF_READ_IMAGE = (
    "Return one extracted PDF image local path plus nearby text context. Use "
    "when a multimodal LLM needs to inspect PDF visual evidence."
)

PDF_LIST_TABLES = (
    "List Docling-extracted PDF tables with diagnostics, optionally scoped to "
    "an outline node. Use for table-centric reading before pdf_read_table."
)

PDF_READ_TABLE = (
    "Read one extracted PDF table with structured rows, markdown/html, "
    "evidence image, nearby context, and diagnostics."
)

PDF_LIST_FIGURES = (
    "List Docling-detected PDF figures with diagnostics, optionally scoped to "
    "an outline node. Use for figure-centric reading before pdf_read_figure."
)

PDF_READ_FIGURE = (
    "Read one extracted PDF figure with local path, caption, nearby context, "
    "and diagnostics."
)

PDF_BOOK_LIST_FORMULAS = (
    "List formulas from a PDF book, optionally scoped to an outline node. Use "
    "for formula-centric book reading; do not use the paper formula tool here."
)

PDF_BOOK_READ_FORMULA = (
    "Read one formula from a PDF book with LaTeX, status, context, and evidence image."
)

PDF_PAPER_LIST_FORMULAS = (
    "List formulas from a PDF paper, optionally scoped to an outline node. Use "
    "for formula-centric paper reading; do not use the book formula tool here."
)

PDF_PAPER_READ_FORMULA = (
    "Read one formula from a PDF paper with LaTeX, status, context, and evidence image."
)

GET_OUTLINE = (
    "Return the document outline for EPUB/PDF. Use this first for "
    "outline-first navigation, chapter selection, and node-scoped reading."
)

RENDER_PDF_PAGE = (
    "Render one PDF page to a PNG evidence image. Use when page-level visual "
    "evidence is needed beyond extracted figures, tables, images, or formulas."
)

EVAL_EXPORT_READING_SESSIONS = (
    "Export opt-in reading-session capture events from sidecars under a root. "
    "Use to inspect real MCP reading calls and retrieval evidence without exposing file paths."
)

EVAL_REPLAY_READING_SESSIONS = (
    "Replay captured search reading-session events under a root to detect retrieval drift. "
    "Only events captured with query text enabled are replayable."
)

DOCTOR_HEALTH_CHECK = (
    "Run deterministic diagnostics for Qdrant, GROBID, FastEmbed cache, parser "
    "dependencies, sidecar catalogs, stale pipeline metadata, and vector consistency."
)
