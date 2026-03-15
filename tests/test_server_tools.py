from __future__ import annotations

from types import SimpleNamespace

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read import server


def test_document_autotune_pdf_parser_tool_wraps_success(
    monkeypatch,
) -> None:
    calls: list[tuple[str | None, str | None, int]] = []

    def autotune(*, doc_id: str | None, path: str | None, sample_pages: int):
        calls.append((doc_id, path, sample_pages))
        return {"profile_path": "/tmp/docling.json", "sample_pages": sample_pages}

    monkeypatch.setattr(
        server,
        "service",
        SimpleNamespace(document_autotune_pdf_parser=autotune),
    )
    monkeypatch.setattr(server, "make_trace_id", lambda: "trace-autotune")

    payload = server.document_autotune_pdf_parser(
        doc_id="doc1",
        path="/tmp/book.pdf",
        sample_pages=9,
    )

    assert payload == {
        "ok": True,
        "data": {"profile_path": "/tmp/docling.json", "sample_pages": 9},
        "error": None,
        "trace_id": "trace-autotune",
    }
    assert calls == [("doc1", "/tmp/book.pdf", 9)]


def test_document_autotune_pdf_parser_tool_wraps_error(monkeypatch) -> None:
    def autotune(*, doc_id: str | None, path: str | None, sample_pages: int):
        raise AppError(
            ErrorCode.INGEST_PDF_DOCLING_FAILED,
            "autotune failed",
            details={"path": path, "sample_pages": sample_pages, "doc_id": doc_id},
        )

    monkeypatch.setattr(
        server,
        "service",
        SimpleNamespace(document_autotune_pdf_parser=autotune),
    )
    monkeypatch.setattr(server, "make_trace_id", lambda: "trace-error")

    payload = server.document_autotune_pdf_parser(path="/tmp/book.pdf")

    assert payload["ok"] is False
    assert payload["data"] is None
    assert payload["trace_id"] == "trace-error"
    assert payload["error"] == {
        "code": ErrorCode.INGEST_PDF_DOCLING_FAILED,
        "message": "autotune failed",
        "details": {
            "path": "/tmp/book.pdf",
            "sample_pages": 20,
            "doc_id": None,
        },
    }
