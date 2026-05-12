from __future__ import annotations

import inspect

from mcp_ebook_read import server
from mcp_ebook_read.operations import OPERATIONS, OPERATIONS_BY_NAME


def test_operation_registry_is_unique_and_exposed_by_server() -> None:
    names = [operation.name for operation in OPERATIONS]

    assert len(names) == len(set(names))
    assert set(OPERATIONS_BY_NAME) == set(names)
    for operation in OPERATIONS:
        assert getattr(server, operation.name) is not None


def test_operation_handlers_keep_explicit_signatures_for_mcp_schema() -> None:
    for operation in OPERATIONS:
        signature = inspect.signature(operation.handler)
        assert signature.parameters
        for parameter in signature.parameters.values():
            assert parameter.kind not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }


def test_operation_metadata_is_explicit() -> None:
    for operation in OPERATIONS:
        assert operation.description.strip()
        assert operation.scope in {"read", "write", "admin"}
        assert operation.file_format in {
            "epub",
            "pdf",
            "storage",
            "library",
            "generic",
        }
        assert operation.use_case in {
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
        }


def test_llm_descriptions_pin_reading_routing_language() -> None:
    descriptions = {operation.name: operation.description for operation in OPERATIONS}

    assert "PDF book" in descriptions["document_ingest_pdf_book"]
    assert "PDF paper" in descriptions["document_ingest_pdf_paper"]
    assert "EPUB book" in descriptions["document_ingest_epub_book"]
    assert "outline-first" in descriptions["get_outline"]
    assert "chapter-scoped" in descriptions["search"]
    assert "formula-centric book" in descriptions["pdf_book_list_formulas"]
    assert "formula-centric paper" in descriptions["pdf_paper_list_formulas"]
    assert "multimodal" in descriptions["epub_read_image"]
    assert "Docling-extracted PDF tables" in descriptions["pdf_list_tables"]
    assert "Docling-detected PDF figures" in descriptions["pdf_list_figures"]
    assert (
        "reading-session capture events" in descriptions["eval_export_reading_sessions"]
    )
    assert "retrieval drift" in descriptions["eval_replay_reading_sessions"]
    assert "Qdrant" in descriptions["doctor_health_check"]
