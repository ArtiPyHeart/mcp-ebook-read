# Suggested commands
- Install/sync env: `uv sync`
- Run MCP server: `QDRANT_URL=http://127.0.0.1:6333 GROBID_URL=http://127.0.0.1:8070 GROBID_TIMEOUT_SECONDS=120 uv run mcp-ebook-read`
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Unit/integration tests: `uv run pytest`
- Targeted tests: `uv run pytest tests/test_service_core.py tests/test_integration_e2e.py -q`
- Benchmarks: `uv run mcp-ebook-formula-benchmark --samples-dir <dir>`