# mcp-ebook-read overview
- Purpose: local MCP server for high-fidelity EPUB/PDF book and paper reading, search, formula/image/table access, and sidecar persistence.
- Stack: Python 3.13, uv, pydantic, mcp, docling, pix2text, pymupdf, ebooklib, lxml, and SQLite FTS5.
- Main code: `src/mcp_ebook_read/` with `service.py` as application orchestration, `server.py` as MCP tool surface, `parsers/` for EPUB/PDF parsing, `store/catalog.py` for per-document-folder sidecar SQLite persistence and DocumentGraph storage.
- Retrieval is local-first: SQLite FTS5 plus sidecar DocumentGraph. Qdrant/FastEmbed are not required and should not be reintroduced as startup dependencies.
- GROBID is optional PDF paper metadata/reference enrichment only; ingest must work without `GROBID_URL`.
- Tests live in `tests/`; sample docs live in `tests/samples/`.
- Ingest is asynchronous via background jobs.
