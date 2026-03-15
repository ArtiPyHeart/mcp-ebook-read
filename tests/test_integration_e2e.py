from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from mcp_ebook_read.schema.models import IngestJobStatus
from mcp_ebook_read.service import AppService


pytestmark = pytest.mark.e2e


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AssertionError(f"RUN_MCP_E2E=1 requires {name}")
    return value


def _wait_for_ingest_job(
    service: AppService,
    *,
    doc_id: str,
    job_id: str,
    timeout_seconds: float = 1800.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, object] | None = None
    while time.monotonic() < deadline:
        last = service.document_ingest_status(doc_id=doc_id, job_id=job_id)
        if last["status"] in {
            IngestJobStatus.SUCCEEDED,
            IngestJobStatus.FAILED,
            IngestJobStatus.CANCELED,
        }:
            return last
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for ingest job {job_id}: {last}")


def _prepare_sample_library(samples_root: Path, target_root: Path) -> dict[str, Path]:
    target_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for relative in [
        Path("pdf-books")
        / "Topological Data Analysis in High-Frequency Trading With Python.pdf",
        Path("epub-books")
        / "Probabilistic Machine Learning (Adaptive Computation and Machine Learning series).epub",
        Path("pdf-papers")
        / "Optimal market-Making strategies under synchronised order arrivals with deep neural networks.pdf",
    ]:
        src = samples_root / relative
        dst = target_root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[str(relative)] = dst
    return copied


def _first_words(text: str, limit: int = 8) -> str:
    return " ".join(text.split()[:limit]).strip()


def test_e2e_scan_ingest_search_read_images_formulas_and_storage(
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

    library_root = tmp_path / "library"
    copied = _prepare_sample_library(samples_root, library_root)

    service = AppService.from_env()

    scan = service.library_scan(str(library_root), ["**/*.pdf", "**/*.epub"])
    discovered = len(scan["added"]) + len(scan["updated"]) + scan["unchanged_count"]
    assert discovered >= 3

    sidecars = service.storage_list_sidecars(root=str(library_root), limit=20)
    assert sidecars["documents_count"] == 3

    book_pdf = copied[
        "pdf-books/Topological Data Analysis in High-Frequency Trading With Python.pdf"
    ]
    book_epub = copied[
        "epub-books/Probabilistic Machine Learning (Adaptive Computation and Machine Learning series).epub"
    ]
    paper_pdf = copied[
        "pdf-papers/Optimal market-Making strategies under synchronised order arrivals with deep neural networks.pdf"
    ]

    assert book_pdf.exists()
    assert book_epub.exists()
    assert paper_pdf.exists()

    pdf_queued = service.document_ingest_pdf_book(
        doc_id=None,
        path=str(book_pdf),
        force=True,
    )
    pdf_status = service.document_ingest_status(
        doc_id=pdf_queued["doc_id"],
        job_id=pdf_queued["job_id"],
    )
    assert pdf_status["status"] in {
        IngestJobStatus.QUEUED,
        IngestJobStatus.RUNNING,
    }
    pdf_jobs = service.document_ingest_list_jobs(doc_id=pdf_queued["doc_id"], limit=5)
    assert pdf_jobs["jobs"][0]["job_id"] == pdf_queued["job_id"]
    pdf_result = _wait_for_ingest_job(
        service,
        doc_id=pdf_queued["doc_id"],
        job_id=pdf_queued["job_id"],
    )["result"]
    assert pdf_result["chunks_count"] > 0

    epub_queued = service.document_ingest_epub_book(
        doc_id=None,
        path=str(book_epub),
        force=True,
    )
    epub_result = _wait_for_ingest_job(
        service,
        doc_id=epub_queued["doc_id"],
        job_id=epub_queued["job_id"],
    )["result"]
    assert epub_result["chunks_count"] > 0

    paper_queued = service.document_ingest_pdf_paper(
        doc_id=None,
        path=str(paper_pdf),
        force=True,
    )
    paper_result = _wait_for_ingest_job(
        service,
        doc_id=paper_queued["doc_id"],
        job_id=paper_queued["job_id"],
    )["result"]
    assert "grobid" in paper_result["parser_chain"]

    pdf_catalog = service._catalog_for_doc_id(pdf_result["doc_id"])
    assert pdf_catalog is not None
    pdf_seed_chunk = pdf_catalog.get_chunks_window(
        pdf_result["doc_id"], center_order=0, before=0, after=0
    )
    assert pdf_seed_chunk
    query_text = _first_words(pdf_seed_chunk[0].search_text)
    assert query_text

    hits = service.search(query=query_text, doc_ids=[pdf_result["doc_id"]], top_k=3)[
        "hits"
    ]
    assert hits

    read = service.read(
        locator=hits[0]["locator"], before=0, after=1, out_format="text"
    )
    assert read["content"].strip()

    pdf_outline = service.get_outline(pdf_result["doc_id"])
    epub_outline = service.get_outline(epub_result["doc_id"])
    paper_outline = service.get_outline(paper_result["doc_id"])
    assert isinstance(pdf_outline["nodes"], list) and pdf_outline["nodes"]
    assert isinstance(epub_outline["nodes"], list) and epub_outline["nodes"]
    assert isinstance(paper_outline["nodes"], list) and paper_outline["nodes"]

    pdf_node_id = pdf_outline["nodes"][0]["id"]
    epub_node_id = epub_outline["nodes"][0]["id"]
    paper_node_id = paper_outline["nodes"][0]["id"]

    pdf_node_read = service.read_outline_node(
        doc_id=pdf_result["doc_id"],
        node_id=pdf_node_id,
        format="text",
        max_chunks=5,
    )
    assert pdf_node_read["content"].strip()

    paper_query = _first_words(paper_outline["nodes"][0]["title"])
    scoped_hits = service.search_in_outline_node(
        doc_id=paper_result["doc_id"],
        node_id=paper_node_id,
        query=paper_query,
        top_k=3,
    )["hits"]
    assert isinstance(scoped_hits, list)

    epub_images = service.epub_list_images(
        doc_id=epub_result["doc_id"],
        node_id=epub_node_id,
        limit=20,
    )
    assert epub_images["images_count"] >= 0
    if epub_images["images_count"] > 0:
        epub_image = service.epub_read_image(
            doc_id=epub_result["doc_id"],
            image_id=epub_images["images"][0]["image_id"],
        )
        assert Path(epub_image["image"]["image_path"]).exists()

    pdf_images = service.pdf_list_images(
        doc_id=pdf_result["doc_id"],
        node_id=pdf_node_id,
        limit=20,
    )
    assert pdf_images["images_count"] >= 0
    if pdf_images["images_count"] > 0:
        pdf_image = service.pdf_read_image(
            doc_id=pdf_result["doc_id"],
            image_id=pdf_images["images"][0]["image_id"],
        )
        assert Path(pdf_image["image"]["image_path"]).exists()

    pdf_book_formulas = service.pdf_book_list_formulas(
        doc_id=pdf_result["doc_id"],
        node_id=pdf_node_id,
        limit=20,
        status=None,
    )
    assert pdf_book_formulas["formulas_count"] >= 0
    if pdf_book_formulas["formulas_count"] > 0:
        pdf_book_formula = service.pdf_book_read_formula(
            doc_id=pdf_result["doc_id"],
            formula_id=pdf_book_formulas["formulas"][0]["formula_id"],
        )
        assert pdf_book_formula["formula"]["latex"]

    pdf_paper_formulas = service.pdf_paper_list_formulas(
        doc_id=paper_result["doc_id"],
        node_id=paper_node_id,
        limit=20,
        status=None,
    )
    assert pdf_paper_formulas["formulas_count"] >= 0
    if pdf_paper_formulas["formulas_count"] > 0:
        pdf_paper_formula = service.pdf_paper_read_formula(
            doc_id=paper_result["doc_id"],
            formula_id=pdf_paper_formulas["formulas"][0]["formula_id"],
        )
        assert pdf_paper_formula["formula"]["latex"]

    render = service.render_pdf_page(doc_id=pdf_result["doc_id"], page=1, dpi=96)
    assert Path(render["image_path"]).exists()

    deleted = service.storage_delete_document(
        doc_id=epub_result["doc_id"],
        path=None,
        remove_artifacts=True,
    )
    assert deleted["deleted_records"] == 1

    paper_pdf.unlink()
    cleaned = service.storage_cleanup_sidecars(
        root=str(library_root),
        remove_missing_documents=True,
        remove_orphan_artifacts=True,
        compact_catalog=False,
    )
    assert cleaned["removed_deleted_count"] >= 1

    final_sidecars = service.storage_list_sidecars(root=str(library_root), limit=20)
    assert final_sidecars["documents_count"] == 1
