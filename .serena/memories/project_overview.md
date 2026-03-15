# mcp-ebook-read overview
- Purpose: local MCP server for high-fidelity EPUB/PDF book and paper reading, search, formula/image access, and sidecar persistence.
- Stack: Python 3.13, uv, pydantic, mcp, docling, pix2text, pymupdf, qdrant-client + fastembed.
- Main code: `src/mcp_ebook_read/` with `service.py` as application orchestration, `server.py` as MCP tool surface, `parsers/` for EPUB/PDF parsing, `index/vector.py` for Qdrant embedding/search, `store/catalog.py` for sidecar sqlite persistence.
- Tests live in `tests/`; sample docs live in `tests/samples/`.
- Runtime requires Qdrant and GROBID; ingest is asynchronous via background jobs.