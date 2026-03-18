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


def test_pdf_table_and_figure_tools_wrap_success(monkeypatch) -> None:
    calls: list[tuple[str, str, object]] = []

    def list_tables(*, doc_id: str, node_id: str | None, limit: int):
        calls.append(("list_tables", doc_id, node_id, limit))
        return {"tables_count": 1}

    def read_table(*, doc_id: str, table_id: str):
        calls.append(("read_table", doc_id, table_id))
        return {"table": {"table_id": table_id}}

    def list_figures(*, doc_id: str, node_id: str | None, limit: int):
        calls.append(("list_figures", doc_id, node_id, limit))
        return {"figures_count": 1}

    def read_figure(*, doc_id: str, figure_id: str):
        calls.append(("read_figure", doc_id, figure_id))
        return {"figure": {"figure_id": figure_id}}

    monkeypatch.setattr(
        server,
        "service",
        SimpleNamespace(
            pdf_list_tables=list_tables,
            pdf_read_table=read_table,
            pdf_list_figures=list_figures,
            pdf_read_figure=read_figure,
        ),
    )
    monkeypatch.setattr(server, "make_trace_id", lambda: "trace-visuals")

    list_tables_payload = server.pdf_list_tables(doc_id="doc1", node_id="n1", limit=9)
    read_table_payload = server.pdf_read_table(doc_id="doc1", table_id="t1")
    list_figures_payload = server.pdf_list_figures(doc_id="doc1", node_id=None, limit=7)
    read_figure_payload = server.pdf_read_figure(doc_id="doc1", figure_id="f1")

    assert list_tables_payload == {
        "ok": True,
        "data": {"tables_count": 1},
        "error": None,
        "trace_id": "trace-visuals",
    }
    assert read_table_payload == {
        "ok": True,
        "data": {"table": {"table_id": "t1"}},
        "error": None,
        "trace_id": "trace-visuals",
    }
    assert list_figures_payload == {
        "ok": True,
        "data": {"figures_count": 1},
        "error": None,
        "trace_id": "trace-visuals",
    }
    assert read_figure_payload == {
        "ok": True,
        "data": {"figure": {"figure_id": "f1"}},
        "error": None,
        "trace_id": "trace-visuals",
    }
    assert calls == [
        ("list_tables", "doc1", "n1", 9),
        ("read_table", "doc1", "t1"),
        ("list_figures", "doc1", None, 7),
        ("read_figure", "doc1", "f1"),
    ]


def test_pdf_table_tool_wraps_error(monkeypatch) -> None:
    def read_table(*, doc_id: str, table_id: str):
        raise AppError(
            ErrorCode.READ_TABLE_NOT_FOUND,
            "table missing",
            details={"doc_id": doc_id, "table_id": table_id},
        )

    monkeypatch.setattr(
        server,
        "service",
        SimpleNamespace(pdf_read_table=read_table),
    )
    monkeypatch.setattr(server, "make_trace_id", lambda: "trace-table-error")

    payload = server.pdf_read_table(doc_id="doc1", table_id="missing")

    assert payload == {
        "ok": False,
        "data": None,
        "error": {
            "code": ErrorCode.READ_TABLE_NOT_FOUND,
            "message": "table missing",
            "details": {"doc_id": "doc1", "table_id": "missing"},
        },
        "trace_id": "trace-table-error",
    }
