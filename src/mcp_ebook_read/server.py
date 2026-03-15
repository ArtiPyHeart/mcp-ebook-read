"""MCP server entrypoint."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from mcp.server.fastmcp import FastMCP

from mcp_ebook_read.errors import AppError, ErrorCode, to_error_payload
from mcp_ebook_read.logging import make_trace_id, setup_logging
from mcp_ebook_read.service import AppService

setup_logging()
logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-ebook-read")
service: AppService | None = None
T = TypeVar("T")


def _require_service() -> AppService:
    if service is None:
        raise AppError(
            ErrorCode.STARTUP_DEPENDENCY_NOT_READY,
            "Service is not initialized.",
            details={
                "hint": "Startup preflight failed or cli_entry() was not used.",
            },
        )
    return service


def tool_handler(fn: Callable[..., T]) -> Callable[..., dict[str, Any]]:
    """Wrap tool results with standard response envelope."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        trace_id = make_trace_id()
        try:
            data = fn(*args, **kwargs)
            return {
                "ok": True,
                "data": data,
                "error": None,
                "trace_id": trace_id,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool_failed", extra={"trace_id": trace_id})
            return {
                "ok": False,
                "data": None,
                "error": to_error_payload(exc),
                "trace_id": trace_id,
            }

    return wrapper


@mcp.tool()
@tool_handler
def library_scan(root: str, patterns: list[str] | None = None) -> dict[str, Any]:
    """Scan root for PDF/EPUB files and register changes in catalog."""
    effective_patterns = patterns or ["**/*.pdf", "**/*.epub"]
    return _require_service().library_scan(root=root, patterns=effective_patterns)


@mcp.tool()
@tool_handler
def storage_list_sidecars(root: str, limit: int = 100) -> dict[str, Any]:
    """List per-folder sidecar persistence state under one root."""
    return _require_service().storage_list_sidecars(root=root, limit=limit)


@mcp.tool()
@tool_handler
def storage_delete_document(
    doc_id: str | None = None,
    path: str | None = None,
    remove_artifacts: bool = True,
) -> dict[str, Any]:
    """Delete one document from persistence (catalog + vector + optional local artifacts)."""
    return _require_service().storage_delete_document(
        doc_id=doc_id,
        path=path,
        remove_artifacts=remove_artifacts,
    )


@mcp.tool()
@tool_handler
def storage_cleanup_sidecars(
    root: str,
    remove_missing_documents: bool = True,
    remove_orphan_artifacts: bool = True,
    compact_catalog: bool = True,
) -> dict[str, Any]:
    """Cleanup sidecar persistence under one root."""
    return _require_service().storage_cleanup_sidecars(
        root=root,
        remove_missing_documents=remove_missing_documents,
        remove_orphan_artifacts=remove_orphan_artifacts,
        compact_catalog=compact_catalog,
    )


@mcp.tool()
@tool_handler
def document_ingest_pdf_book(
    doc_id: str | None = None,
    path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Queue one PDF book ingest job for background execution."""
    return _require_service().document_ingest_pdf_book(
        doc_id=doc_id, path=path, force=force
    )


@mcp.tool()
@tool_handler
def document_ingest_epub_book(
    doc_id: str | None = None,
    path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Queue one EPUB book ingest job for background execution."""
    return _require_service().document_ingest_epub_book(
        doc_id=doc_id, path=path, force=force
    )


@mcp.tool()
@tool_handler
def document_ingest_pdf_paper(
    doc_id: str | None = None,
    path: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Queue one PDF paper ingest job for background execution."""
    return _require_service().document_ingest_pdf_paper(
        doc_id=doc_id, path=path, force=force
    )


@mcp.tool()
@tool_handler
def document_ingest_status(
    doc_id: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Read current status for one background ingest job, or the latest job for a document."""
    return _require_service().document_ingest_status(doc_id=doc_id, job_id=job_id)


@mcp.tool()
@tool_handler
def document_ingest_list_jobs(
    doc_id: str,
    limit: int = 20,
) -> dict[str, Any]:
    """List recent background ingest jobs for one document."""
    return _require_service().document_ingest_list_jobs(doc_id=doc_id, limit=limit)


@mcp.tool()
@tool_handler
def search(
    query: str, doc_ids: list[str] | None = None, top_k: int = 20
) -> dict[str, Any]:
    """Vector search in Qdrant and return locator-rich hits."""
    return _require_service().search(query=query, doc_ids=doc_ids, top_k=top_k)


@mcp.tool()
@tool_handler
def search_in_outline_node(
    doc_id: str,
    node_id: str,
    query: str,
    top_k: int = 20,
) -> dict[str, Any]:
    """Vector search constrained to one outline node/chapter."""
    return _require_service().search_in_outline_node(
        doc_id=doc_id, node_id=node_id, query=query, top_k=top_k
    )


@mcp.tool()
@tool_handler
def read(
    locator: dict[str, Any], before: int = 1, after: int = 1, format: str = "markdown"
) -> dict[str, Any]:
    """Read chunk window around a locator."""
    return _require_service().read(
        locator=locator, before=before, after=after, out_format=format
    )


@mcp.tool()
@tool_handler
def read_outline_node(
    doc_id: str,
    node_id: str,
    format: str = "markdown",
    max_chunks: int = 120,
) -> dict[str, Any]:
    """Read one outline node/chapter directly."""
    return _require_service().read_outline_node(
        doc_id=doc_id,
        node_id=node_id,
        out_format=format,
        max_chunks=max_chunks,
    )


@mcp.tool()
@tool_handler
def epub_list_images(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """List extracted images from an EPUB document (optionally scoped to one outline node)."""
    return _require_service().epub_list_images(
        doc_id=doc_id,
        node_id=node_id,
        limit=limit,
    )


@mcp.tool()
@tool_handler
def epub_read_image(doc_id: str, image_id: str) -> dict[str, Any]:
    """Read one extracted EPUB image with local path and nearby text context."""
    return _require_service().epub_read_image(doc_id=doc_id, image_id=image_id)


@mcp.tool()
@tool_handler
def pdf_list_images(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """List extracted images from a PDF document (optionally scoped to one outline node)."""
    return _require_service().pdf_list_images(
        doc_id=doc_id,
        node_id=node_id,
        limit=limit,
    )


@mcp.tool()
@tool_handler
def pdf_read_image(doc_id: str, image_id: str) -> dict[str, Any]:
    """Read one extracted PDF image with local path and nearby text context."""
    return _require_service().pdf_read_image(doc_id=doc_id, image_id=image_id)


@mcp.tool()
@tool_handler
def pdf_book_list_formulas(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
    status: str | None = None,
) -> dict[str, Any]:
    """List formulas from a PDF book (optionally scoped to one outline node)."""
    return _require_service().pdf_book_list_formulas(
        doc_id=doc_id,
        node_id=node_id,
        limit=limit,
        status=status,
    )


@mcp.tool()
@tool_handler
def pdf_book_read_formula(doc_id: str, formula_id: str) -> dict[str, Any]:
    """Read one formula from a PDF book with context and evidence image."""
    return _require_service().pdf_book_read_formula(
        doc_id=doc_id,
        formula_id=formula_id,
    )


@mcp.tool()
@tool_handler
def pdf_paper_list_formulas(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
    status: str | None = None,
) -> dict[str, Any]:
    """List formulas from a PDF paper (optionally scoped to one outline node)."""
    return _require_service().pdf_paper_list_formulas(
        doc_id=doc_id,
        node_id=node_id,
        limit=limit,
        status=status,
    )


@mcp.tool()
@tool_handler
def pdf_paper_read_formula(doc_id: str, formula_id: str) -> dict[str, Any]:
    """Read one formula from a PDF paper with context and evidence image."""
    return _require_service().pdf_paper_read_formula(
        doc_id=doc_id,
        formula_id=formula_id,
    )


@mcp.tool()
@tool_handler
def get_outline(doc_id: str) -> dict[str, Any]:
    """Return document outline for EPUB/PDF."""
    return _require_service().get_outline(doc_id)


@mcp.tool()
@tool_handler
def render_pdf_page(doc_id: str, page: int, dpi: int = 200) -> dict[str, Any]:
    """Render PDF page to PNG evidence image."""
    return _require_service().render_pdf_page(doc_id=doc_id, page=page, dpi=dpi)


def cli_entry() -> None:
    """CLI entrypoint for packaged execution."""
    global service
    try:
        service = AppService.from_env()
    except Exception as exc:  # noqa: BLE001
        trace_id = make_trace_id()
        logger.exception("startup_failed", extra={"trace_id": trace_id})
        payload = {
            "ok": False,
            "data": None,
            "error": to_error_payload(exc),
            "trace_id": trace_id,
        }
        print(json.dumps(payload, ensure_ascii=True), file=sys.stderr)
        raise SystemExit(1) from exc

    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli_entry()
