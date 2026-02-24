"""Application error definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    STARTUP_DEPENDENCY_NOT_READY = "STARTUP_DEPENDENCY_NOT_READY"
    SCAN_INVALID_ROOT = "SCAN_INVALID_ROOT"
    INGEST_DOC_NOT_FOUND = "INGEST_DOC_NOT_FOUND"
    INGEST_UNSUPPORTED_TYPE = "INGEST_UNSUPPORTED_TYPE"
    INGEST_PDF_DOCLING_FAILED = "INGEST_PDF_DOCLING_FAILED"
    INGEST_PDF_FORMULA_ENGINE_UNAVAILABLE = "INGEST_PDF_FORMULA_ENGINE_UNAVAILABLE"
    INGEST_PAPER_GROBID_UNAVAILABLE = "INGEST_PAPER_GROBID_UNAVAILABLE"
    INGEST_PAPER_GROBID_FAILED = "INGEST_PAPER_GROBID_FAILED"
    INGEST_EPUB_PARSE_FAILED = "INGEST_EPUB_PARSE_FAILED"
    SEARCH_INDEX_NOT_READY = "SEARCH_INDEX_NOT_READY"
    READ_LOCATOR_NOT_FOUND = "READ_LOCATOR_NOT_FOUND"
    READ_IMAGE_NOT_FOUND = "READ_IMAGE_NOT_FOUND"
    RENDER_PAGE_FAILED = "RENDER_PAGE_FAILED"


class AppError(Exception):
    """Domain error with stable error code for MCP responses."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def to_error_payload(exc: Exception) -> dict[str, Any]:
    """Convert an exception to MCP error payload."""
    if isinstance(exc, AppError):
        return {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details or None,
        }

    return {
        "code": "INTERNAL_ERROR",
        "message": str(exc),
        "details": None,
    }
