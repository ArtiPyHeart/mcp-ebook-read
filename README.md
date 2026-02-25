# mcp-ebook-read

A local MCP server for Codex to read and retrieve content from EPUB/PDF documents.

## One-Command Docker Setup

### Qdrant (required)

```bash
docker rm -f qdrant 2>/dev/null || true && docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.16.3
```

### GROBID (required by startup preflight and `document_ingest_pdf_paper`)

```bash
docker rm -f grobid 2>/dev/null || true && docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
```

## Verify Services

```bash
curl -sS http://localhost:6333/collections
curl -sS http://localhost:8070/api/isalive
```

Expected:
- Qdrant returns JSON with `"status":"ok"`
- GROBID returns `true`

## Run MCP Server (PyPI via `uvx`)

```bash
QDRANT_URL=http://localhost:6333 GROBID_URL=http://localhost:8070 GROBID_TIMEOUT_SECONDS=120 uvx mcp-ebook-read
```

If startup preflight fails, the server exits with a structured error payload on stderr that includes missing env vars and setup hints.

### First Run Recommendation

Before configuring this MCP inside an MCP client, run it once manually from a terminal:

```bash
QDRANT_URL=http://localhost:6333 GROBID_URL=http://localhost:8070 GROBID_TIMEOUT_SECONDS=120 uvx --refresh mcp-ebook-read
```

This pre-resolves and aligns runtime dependencies, which helps avoid long first-time activation latency after MCP client configuration.

## Environment Variables

Required:
- `QDRANT_URL` (for example `http://127.0.0.1:6333`)
- `GROBID_URL` (for example `http://127.0.0.1:8070`)

Optional:
- `GROBID_TIMEOUT_SECONDS` (default `20`; recommended `120` for large papers)
- `QDRANT_COLLECTION` (default `mcp_ebook_read_chunks`)
- `QDRANT_TIMEOUT_SECONDS` (default `10`)
- `FASTEMBED_MODEL` (FastEmbed model override)
- `DOCLING_FORMULA_ENRICHMENT` (`true` by default)
- `PDF_FORMULA_REQUIRE_ENGINE` (`true` by default)
- `PDF_FORMULA_BATCH_SIZE` (`auto` by default; or an explicit integer)

## Persistence Model

- Persistence is sidecar-based and auto-routed by document location.
- For each document, MCP writes state to `<document_dir>/.mcp-ebook-read/`.
- Sidecar contains:
  - `catalog.db`
  - `docs/<doc_id>/reading/reading.md`
  - `docs/<doc_id>/assets/...`
  - `docs/<doc_id>/evidence/...`

## Notes
- Use `library_scan` to discover `.pdf`/`.epub` files under a root and register updates/removals.
- Use `search` for global semantic retrieval and `read` for locator-based chunk windows.
- Startup preflight is fail-fast and requires both Qdrant and GROBID to be configured and reachable.
- Use `document_ingest_pdf_book` for PDF books.
- Use `document_ingest_epub_book` for EPUB books.
- Use `document_ingest_pdf_paper` for PDF papers (Docling + GROBID, fail-fast).
- Use `search_in_outline_node` when you need chapter-scoped retrieval (recommended for reading workflows).
- Use `get_outline` to fetch document outline nodes before chapter/formula/image scoped reading.
- Use `read_outline_node` to read a chapter/outline node directly without locator stitching.
- Use `render_pdf_page` for PDF evidence rendering.
- PDF image extraction is on-demand: ingest does not pre-extract PDF images.
- Use `pdf_list_images` to trigger/list extracted PDF figure/table images (optionally scoped to one outline node).
- Use `pdf_read_image` to get one extracted PDF image path plus nearby text context.
- Use `pdf_book_list_formulas` / `pdf_book_read_formula` for formula-centric reading on PDF books.
- Use `pdf_paper_list_formulas` / `pdf_paper_read_formula` for formula-centric reading on PDF papers.
- Use `epub_list_images` to list extracted EPUB images (optionally scoped to one outline node).
- Use `epub_read_image` to get one EPUB image path plus nearby text context.
- Use `storage_list_sidecars` to inspect sidecar persistence under a root.
- Use `storage_delete_document` to remove one document's persisted state.
- Use `storage_cleanup_sidecars` to prune missing docs/orphan artifacts and compact catalogs.
- For large papers, increase `GROBID_TIMEOUT_SECONDS` (for example `120`) to reduce timeout failures.
- PDF ingest now uses a mixed formula pipeline:
  - Docling structure extraction with `do_formula_enrichment`.
  - Pix2Text as the primary formula recovery engine.
  - Fail-fast when formula markers exist but Pix2Text is unavailable.
- Optional formula env controls:
  - `DOCLING_FORMULA_ENRICHMENT` (`true` by default)
  - `PDF_FORMULA_REQUIRE_ENGINE` (`true` by default)
  - `PDF_FORMULA_BATCH_SIZE` (`auto` by default; auto-detected from CPU and memory, or set an explicit integer)
- Sidecar cleanup is explicit:
  - `library_scan` no longer triggers threshold-based auto compaction.
  - Use `storage_cleanup_sidecars(..., compact_catalog=true)` when you want compaction.

## No-Label Formula Benchmark

Use your own non-scanned PDF corpus as a no-label regression baseline (without manual annotations).

```bash
uvx mcp-ebook-formula-benchmark \
  --samples-dir /ABSOLUTE/PATH/TO/pdf-formula-benchmark-corpus \
  --passes 2 \
  --max-unresolved-rate 0.15 \
  --min-latex-valid-rate 0.85 \
  --min-stability-rate 1.0
```

Output is JSON with per-document metrics and a threshold pass/fail flag. Exit code is `0` when thresholds pass, otherwise `2`.

## Claude Code MCP Configuration (JSON via `uvx`)

You can register this server in a Claude Code compatible `mcpServers` JSON config.

### Published package

```json
{
  "mcpServers": {
    "mcp-ebook-read": {
      "command": "uvx",
      "args": [
        "mcp-ebook-read"
      ],
      "env": {
        "QDRANT_URL": "http://127.0.0.1:6333",
        "QDRANT_COLLECTION": "mcp_ebook_read_chunks",
        "GROBID_URL": "http://127.0.0.1:8070",
        "GROBID_TIMEOUT_SECONDS": "120"
      }
    }
  }
}
```

### Security note
- Do not put real passwords, API keys, or tokens directly in committed JSON files.
- Use environment variables or secret managers, and keep example values as placeholders only.

## Codex MCP Configuration (TOML)

You can also configure MCP servers in Codex using TOML style (for example in a Codex MCP config file).

### Example

```toml
[mcp_servers.mcp-ebook-read]
command = "uvx"
args = [ "mcp-ebook-read" ]
startup_timeout_sec = 60

[mcp_servers.mcp-ebook-read.env]
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_COLLECTION = "mcp_ebook_read_chunks"
GROBID_URL = "http://127.0.0.1:8070"
GROBID_TIMEOUT_SECONDS = "120"
```
