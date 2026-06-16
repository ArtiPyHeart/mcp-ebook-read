"""Contract-first MCP operation registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from mcp_ebook_read import tool_descriptions as desc

OperationScope = Literal["read", "write", "admin"]
OperationFormat = Literal["epub", "pdf", "document", "storage", "library", "generic"]
OperationUseCase = Literal[
    "scan",
    "storage",
    "ingest",
    "search",
    "read",
    "image",
    "table",
    "figure",
    "formula",
    "outline",
    "render",
    "eval",
    "diagnostic",
]

_service_provider: Callable[[], Any] | None = None


@dataclass(frozen=True)
class ToolOperation:
    """Single source of truth for an MCP-facing operation."""

    name: str
    handler: Callable[..., dict[str, Any]]
    description: str
    scope: OperationScope
    file_format: OperationFormat
    use_case: OperationUseCase


def set_service_provider(provider: Callable[[], Any]) -> None:
    """Set the AppService provider used by registered operation handlers."""
    global _service_provider
    _service_provider = provider


def _service() -> Any:
    if _service_provider is None:
        raise RuntimeError("MCP operation service provider is not initialized.")
    return _service_provider()


def library_scan(
    root: str | None = None, patterns: list[str] | None = None
) -> dict[str, Any]:
    effective_patterns = patterns or ["**/*.pdf", "**/*.epub"]
    return _service().library_scan(root=root, patterns=effective_patterns)


def library_explore(
    query: str, root: str | None = None, top_k: int = 12
) -> dict[str, Any]:
    return _service().library_explore(root=root, query=query, top_k=top_k)


def library_ingest_documents(
    root: str | None = None,
    force: bool = False,
    max_documents: int = 0,
) -> dict[str, Any]:
    return _service().library_ingest_documents(
        root=root,
        force=force,
        max_documents=max_documents,
    )


def library_ingest_status(
    root: str | None = None,
    limit_running: int = 20,
    limit_failed: int = 20,
    limit_queued: int = 20,
) -> dict[str, Any]:
    return _service().library_ingest_status(
        root=root,
        limit_running=limit_running,
        limit_failed=limit_failed,
        limit_queued=limit_queued,
    )


def storage_list_sidecars(root: str | None = None, limit: int = 100) -> dict[str, Any]:
    return _service().storage_list_sidecars(root=root, limit=limit)


def storage_delete_document(
    doc_id: str | None = None,
    path: str | None = None,
    remove_artifacts: bool = True,
) -> dict[str, Any]:
    return _service().storage_delete_document(
        doc_id=doc_id,
        path=path,
        remove_artifacts=remove_artifacts,
    )


def storage_cleanup_sidecars(
    root: str | None = None,
    remove_missing_documents: bool = True,
    remove_orphan_artifacts: bool = True,
    compact_catalog: bool = True,
) -> dict[str, Any]:
    return _service().storage_cleanup_sidecars(
        root=root,
        remove_missing_documents=remove_missing_documents,
        remove_orphan_artifacts=remove_orphan_artifacts,
        compact_catalog=compact_catalog,
    )


def document_ingest(
    doc_id: str | None = None,
    path: str | None = None,
    root: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    return _service().document_ingest(doc_id=doc_id, path=path, root=root, force=force)


def document_ingest_status(doc_id: str, job_id: str | None = None) -> dict[str, Any]:
    return _service().document_ingest_status(doc_id=doc_id, job_id=job_id)


def document_ingest_list_jobs(doc_id: str, limit: int = 20) -> dict[str, Any]:
    return _service().document_ingest_list_jobs(doc_id=doc_id, limit=limit)


def document_explore(doc_id: str, query: str, top_k: int = 8) -> dict[str, Any]:
    return _service().document_explore(doc_id=doc_id, query=query, top_k=top_k)


def document_node(doc_id: str, node_id: str) -> dict[str, Any]:
    return _service().document_node(doc_id=doc_id, node_id=node_id)


def search_in_outline_node(
    doc_id: str,
    node_id: str,
    query: str,
    top_k: int = 20,
) -> dict[str, Any]:
    return _service().search_in_outline_node(
        doc_id=doc_id,
        node_id=node_id,
        query=query,
        top_k=top_k,
    )


def read_outline_node(
    doc_id: str,
    node_id: str,
    format: str = "markdown",
    max_chunks: int = 120,
) -> dict[str, Any]:
    return _service().read_outline_node(
        doc_id=doc_id,
        node_id=node_id,
        out_format=format,
        max_chunks=max_chunks,
    )


def epub_list_images(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    return _service().epub_list_images(doc_id=doc_id, node_id=node_id, limit=limit)


def epub_read_image(doc_id: str, image_id: str) -> dict[str, Any]:
    return _service().epub_read_image(doc_id=doc_id, image_id=image_id)


def pdf_list_images(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    return _service().pdf_list_images(doc_id=doc_id, node_id=node_id, limit=limit)


def pdf_read_image(doc_id: str, image_id: str) -> dict[str, Any]:
    return _service().pdf_read_image(doc_id=doc_id, image_id=image_id)


def pdf_list_tables(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    return _service().pdf_list_tables(doc_id=doc_id, node_id=node_id, limit=limit)


def pdf_read_table(doc_id: str, table_id: str) -> dict[str, Any]:
    return _service().pdf_read_table(doc_id=doc_id, table_id=table_id)


def pdf_list_figures(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    return _service().pdf_list_figures(doc_id=doc_id, node_id=node_id, limit=limit)


def pdf_read_figure(doc_id: str, figure_id: str) -> dict[str, Any]:
    return _service().pdf_read_figure(doc_id=doc_id, figure_id=figure_id)


def pdf_list_formulas(
    doc_id: str,
    node_id: str | None = None,
    limit: int = 200,
    status: str | None = None,
) -> dict[str, Any]:
    return _service().pdf_list_formulas(
        doc_id=doc_id,
        node_id=node_id,
        limit=limit,
        status=status,
    )


def pdf_read_formula(doc_id: str, formula_id: str) -> dict[str, Any]:
    return _service().pdf_read_formula(doc_id=doc_id, formula_id=formula_id)


def get_outline(doc_id: str) -> dict[str, Any]:
    return _service().get_outline(doc_id)


def render_pdf_page(doc_id: str, page: int, dpi: int = 200) -> dict[str, Any]:
    return _service().render_pdf_page(doc_id=doc_id, page=page, dpi=dpi)


def eval_export_reading_sessions(
    root: str | None = None, limit: int = 500
) -> dict[str, Any]:
    return _service().eval_export_reading_sessions(root=root, limit=limit)


def eval_replay_reading_sessions(
    root: str | None = None, limit: int = 100
) -> dict[str, Any]:
    return _service().eval_replay_reading_sessions(root=root, limit=limit)


def doctor_health_check(root: str | None = None) -> dict[str, Any]:
    return _service().doctor_health_check(root=root)


OPERATIONS: tuple[ToolOperation, ...] = (
    ToolOperation(
        "library_scan", library_scan, desc.LIBRARY_SCAN, "read", "library", "scan"
    ),
    ToolOperation(
        "library_explore",
        library_explore,
        desc.LIBRARY_EXPLORE,
        "read",
        "library",
        "search",
    ),
    ToolOperation(
        "library_ingest_documents",
        library_ingest_documents,
        desc.LIBRARY_INGEST_DOCUMENTS,
        "write",
        "library",
        "ingest",
    ),
    ToolOperation(
        "library_ingest_status",
        library_ingest_status,
        desc.LIBRARY_INGEST_STATUS,
        "read",
        "library",
        "ingest",
    ),
    ToolOperation(
        "storage_list_sidecars",
        storage_list_sidecars,
        desc.STORAGE_LIST_SIDECARS,
        "read",
        "storage",
        "storage",
    ),
    ToolOperation(
        "storage_delete_document",
        storage_delete_document,
        desc.STORAGE_DELETE_DOCUMENT,
        "admin",
        "storage",
        "storage",
    ),
    ToolOperation(
        "storage_cleanup_sidecars",
        storage_cleanup_sidecars,
        desc.STORAGE_CLEANUP_SIDECARS,
        "admin",
        "storage",
        "storage",
    ),
    ToolOperation(
        "document_ingest",
        document_ingest,
        desc.DOCUMENT_INGEST,
        "write",
        "document",
        "ingest",
    ),
    ToolOperation(
        "document_ingest_status",
        document_ingest_status,
        desc.DOCUMENT_INGEST_STATUS,
        "read",
        "generic",
        "ingest",
    ),
    ToolOperation(
        "document_ingest_list_jobs",
        document_ingest_list_jobs,
        desc.DOCUMENT_INGEST_LIST_JOBS,
        "read",
        "generic",
        "ingest",
    ),
    ToolOperation(
        "document_explore",
        document_explore,
        desc.DOCUMENT_EXPLORE,
        "read",
        "document",
        "search",
    ),
    ToolOperation(
        "document_node",
        document_node,
        desc.DOCUMENT_NODE,
        "read",
        "document",
        "read",
    ),
    ToolOperation(
        "search_in_outline_node",
        search_in_outline_node,
        desc.SEARCH_IN_OUTLINE_NODE,
        "read",
        "generic",
        "search",
    ),
    ToolOperation(
        "read_outline_node",
        read_outline_node,
        desc.READ_OUTLINE_NODE,
        "read",
        "generic",
        "read",
    ),
    ToolOperation(
        "epub_list_images",
        epub_list_images,
        desc.EPUB_LIST_IMAGES,
        "read",
        "epub",
        "image",
    ),
    ToolOperation(
        "epub_read_image",
        epub_read_image,
        desc.EPUB_READ_IMAGE,
        "read",
        "epub",
        "image",
    ),
    ToolOperation(
        "pdf_list_images", pdf_list_images, desc.PDF_LIST_IMAGES, "read", "pdf", "image"
    ),
    ToolOperation(
        "pdf_read_image", pdf_read_image, desc.PDF_READ_IMAGE, "read", "pdf", "image"
    ),
    ToolOperation(
        "pdf_list_tables", pdf_list_tables, desc.PDF_LIST_TABLES, "read", "pdf", "table"
    ),
    ToolOperation(
        "pdf_read_table", pdf_read_table, desc.PDF_READ_TABLE, "read", "pdf", "table"
    ),
    ToolOperation(
        "pdf_list_figures",
        pdf_list_figures,
        desc.PDF_LIST_FIGURES,
        "read",
        "pdf",
        "figure",
    ),
    ToolOperation(
        "pdf_read_figure",
        pdf_read_figure,
        desc.PDF_READ_FIGURE,
        "read",
        "pdf",
        "figure",
    ),
    ToolOperation(
        "pdf_list_formulas",
        pdf_list_formulas,
        desc.PDF_LIST_FORMULAS,
        "read",
        "pdf",
        "formula",
    ),
    ToolOperation(
        "pdf_read_formula",
        pdf_read_formula,
        desc.PDF_READ_FORMULA,
        "read",
        "pdf",
        "formula",
    ),
    ToolOperation(
        "get_outline", get_outline, desc.GET_OUTLINE, "read", "generic", "outline"
    ),
    ToolOperation(
        "render_pdf_page",
        render_pdf_page,
        desc.RENDER_PDF_PAGE,
        "read",
        "pdf",
        "render",
    ),
    ToolOperation(
        "eval_export_reading_sessions",
        eval_export_reading_sessions,
        desc.EVAL_EXPORT_READING_SESSIONS,
        "read",
        "generic",
        "eval",
    ),
    ToolOperation(
        "eval_replay_reading_sessions",
        eval_replay_reading_sessions,
        desc.EVAL_REPLAY_READING_SESSIONS,
        "read",
        "generic",
        "eval",
    ),
    ToolOperation(
        "doctor_health_check",
        doctor_health_check,
        desc.DOCTOR_HEALTH_CHECK,
        "read",
        "generic",
        "diagnostic",
    ),
)

OPERATIONS_BY_NAME: dict[str, ToolOperation] = {op.name: op for op in OPERATIONS}
