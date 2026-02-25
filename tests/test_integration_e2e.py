from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mcp_ebook_read.service import AppService


pytestmark = pytest.mark.e2e


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AssertionError(f"RUN_MCP_E2E=1 requires {name}")
    return value


def test_e2e_scan_ingest_search_read_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if os.environ.get("RUN_MCP_E2E") != "1":
        pytest.skip("Set RUN_MCP_E2E=1 to run integration smoke tests")

    qdrant_url = _require_env("QDRANT_URL")
    grobid_url = _require_env("GROBID_URL")

    samples_root = Path("tests/samples").resolve()
    if not samples_root.exists():
        pytest.skip("tests/samples not found")

    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("GROBID_URL", grobid_url)
    monkeypatch.setenv(
        "GROBID_TIMEOUT_SECONDS", os.environ.get("GROBID_TIMEOUT_SECONDS", "120")
    )
    monkeypatch.setenv("QDRANT_COLLECTION", f"mcp_ebook_read_e2e_{int(time.time())}")

    service = AppService.from_env()

    scan = service.library_scan(str(samples_root), ["**/*.pdf", "**/*.epub"])
    discovered = len(scan["added"]) + len(scan["updated"]) + scan["unchanged_count"]
    assert discovered >= 3

    book_pdf = (
        samples_root
        / "pdf-books"
        / "Topological Data Analysis in High-Frequency Trading With Python.pdf"
    )
    book_epub = (
        samples_root / "epub-books" / "Solana Development with Rust and Anchor.epub"
    )
    paper_pdf = (
        samples_root
        / "pdf-papers"
        / "On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.pdf"
    )

    assert book_pdf.exists()
    assert book_epub.exists()
    assert paper_pdf.exists()

    pdf_result = service.document_ingest_pdf_book(
        doc_id=None,
        path=str(book_pdf),
        force=True,
    )
    assert pdf_result["chunks_count"] > 0

    epub_result = service.document_ingest_epub_book(
        doc_id=None,
        path=str(book_epub),
        force=True,
    )
    assert epub_result["chunks_count"] > 0

    paper_result = service.document_ingest_pdf_paper(
        doc_id=None,
        path=str(paper_pdf),
        force=True,
    )
    assert "grobid" in paper_result["parser_chain"]

    catalog = service._catalog_for_doc_id(pdf_result["doc_id"])
    assert catalog is not None
    seed_chunk = catalog.get_chunks_window(
        pdf_result["doc_id"], center_order=0, before=0, after=0
    )
    assert seed_chunk
    query_text = " ".join(seed_chunk[0].search_text.split()[:8]).strip()
    assert query_text

    hits = service.search(query=query_text, doc_ids=[pdf_result["doc_id"]], top_k=3)[
        "hits"
    ]
    assert hits

    read = service.read(
        locator=hits[0]["locator"], before=0, after=1, out_format="text"
    )
    assert read["content"].strip()

    outline = service.get_outline(paper_result["doc_id"])
    assert isinstance(outline["nodes"], list)

    render = service.render_pdf_page(doc_id=pdf_result["doc_id"], page=1, dpi=96)
    assert Path(render["image_path"]).exists()
