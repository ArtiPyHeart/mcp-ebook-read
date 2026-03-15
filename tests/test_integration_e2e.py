from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import fitz
import pytest

from mcp_ebook_read.schema.models import IngestJobStatus
from mcp_ebook_read.service import AppService

pytestmark = pytest.mark.e2e

PDF_BOOK_RELATIVE = (
    Path("pdf-books")
    / "Topological Data Analysis in High-Frequency Trading With Python.pdf"
)
EPUB_BOOK_RELATIVE = (
    Path("epub-books")
    / "Probabilistic Machine Learning (Adaptive Computation and Machine Learning series).epub"
)
PDF_PAPER_RELATIVE = (
    Path("pdf-papers")
    / "Optimal market-Making strategies under synchronised order arrivals with deep neural networks.pdf"
)


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


def _prepare_sample_document(
    *,
    samples_root: Path,
    target_root: Path,
    relative_path: Path,
    max_pages: int | None = None,
) -> Path:
    src = samples_root / relative_path
    dst = target_root / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)

    if max_pages is None or src.suffix.lower() != ".pdf":
        shutil.copy2(src, dst)
        return dst

    with fitz.open(str(src)) as source:
        subset = fitz.open()
        try:
            last_page = min(max_pages, source.page_count) - 1
            subset.insert_pdf(source, from_page=0, to_page=last_page)
            subset.save(str(dst))
        finally:
            subset.close()
    return dst


def _bootstrap_e2e_service(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collection_suffix: str,
) -> tuple[Path, Path, AppService]:
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
    monkeypatch.setenv(
        "QDRANT_COLLECTION",
        f"mcp_ebook_read_e2e_{collection_suffix}_{int(time.time())}",
    )
    tuning_profile_path = tmp_path / "runtime" / "docling_tuning.json"
    monkeypatch.setenv("PDF_DOCLING_TUNING_PROFILE_PATH", str(tuning_profile_path))
    return samples_root, tuning_profile_path, AppService.from_env()


def _resolve_scanned_doc_id(service: AppService, path: Path) -> str:
    catalog = service._catalog_for_document_path(path)
    doc = catalog.get_document_by_path(str(path.resolve()))
    assert doc is not None
    return doc.doc_id


def _wait_for_successful_job(
    service: AppService, *, doc_id: str, job_id: str
) -> dict[str, object]:
    status = _wait_for_ingest_job(service, doc_id=doc_id, job_id=job_id)
    assert status["status"] == IngestJobStatus.SUCCEEDED, status
    result = status["result"]
    assert result is not None
    return result


def _first_words(text: str, limit: int = 8) -> str:
    return " ".join(text.split()[:limit]).strip()


def test_e2e_pdf_book_autotune_ingest_search_and_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples_root, tuning_profile_path, service = _bootstrap_e2e_service(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        collection_suffix="pdf_book",
    )
    try:
        library_root = tmp_path / "library"
        book_pdf = _prepare_sample_document(
            samples_root=samples_root,
            target_root=library_root,
            relative_path=PDF_BOOK_RELATIVE,
            max_pages=40,
        )

        scan = service.library_scan(str(library_root), ["**/*.pdf"])
        discovered = len(scan["added"]) + len(scan["updated"]) + scan["unchanged_count"]
        assert discovered == 1

        sidecars = service.storage_list_sidecars(root=str(library_root), limit=20)
        assert sidecars["documents_count"] == 1

        book_doc_id = _resolve_scanned_doc_id(service, book_pdf)
        autotune = service.document_autotune_pdf_parser(
            doc_id=book_doc_id,
            path=None,
            sample_pages=5,
        )
        assert autotune["doc_id"] == book_doc_id
        assert autotune["path"] == str(book_pdf.resolve())
        assert autotune["profile_path"] == str(tuning_profile_path)
        assert tuning_profile_path.exists()
        assert autotune["selected_config"]["num_threads"] >= 1
        assert autotune["selected_config"]["ocr_batch_size"] >= 1
        assert autotune["benchmarks"]

        queued = service.document_ingest_pdf_book(
            doc_id=None,
            path=str(book_pdf),
            force=True,
        )
        initial_status = service.document_ingest_status(
            doc_id=queued["doc_id"],
            job_id=queued["job_id"],
        )
        assert initial_status["status"] in {
            IngestJobStatus.QUEUED,
            IngestJobStatus.RUNNING,
        }
        jobs = service.document_ingest_list_jobs(doc_id=queued["doc_id"], limit=5)
        assert jobs["jobs"][0]["job_id"] == queued["job_id"]
        result = _wait_for_successful_job(
            service,
            doc_id=queued["doc_id"],
            job_id=queued["job_id"],
        )
        assert result["chunks_count"] > 0

        pdf_catalog = service._catalog_for_doc_id(result["doc_id"])
        assert pdf_catalog is not None
        seed_chunk = pdf_catalog.get_chunks_window(
            result["doc_id"], center_order=0, before=0, after=0
        )
        assert seed_chunk
        query_text = _first_words(seed_chunk[0].search_text)
        assert query_text

        hits = service.search(query=query_text, doc_ids=[result["doc_id"]], top_k=3)[
            "hits"
        ]
        assert hits

        read = service.read(
            locator=hits[0]["locator"], before=0, after=1, out_format="text"
        )
        assert read["content"].strip()

        outline = service.get_outline(result["doc_id"])
        assert isinstance(outline["nodes"], list) and outline["nodes"]
        node_id = outline["nodes"][0]["id"]

        node_read = service.read_outline_node(
            doc_id=result["doc_id"],
            node_id=node_id,
            out_format="text",
            max_chunks=5,
        )
        assert node_read["content"].strip()

        scoped_query = _first_words(node_read["content"])
        scoped_hits = service.search_in_outline_node(
            doc_id=result["doc_id"],
            node_id=node_id,
            query=scoped_query,
            top_k=3,
        )["hits"]
        assert isinstance(scoped_hits, list)

        pdf_images = service.pdf_list_images(
            doc_id=result["doc_id"],
            node_id=node_id,
            limit=20,
        )
        assert pdf_images["images_count"] >= 0
        if pdf_images["images_count"] > 0:
            pdf_image = service.pdf_read_image(
                doc_id=result["doc_id"],
                image_id=pdf_images["images"][0]["image_id"],
            )
            assert Path(pdf_image["image"]["image_path"]).exists()

        pdf_book_formulas = service.pdf_book_list_formulas(
            doc_id=result["doc_id"],
            node_id=node_id,
            limit=20,
            status=None,
        )
        assert pdf_book_formulas["formulas_count"] >= 0
        if pdf_book_formulas["formulas_count"] > 0:
            pdf_book_formula = service.pdf_book_read_formula(
                doc_id=result["doc_id"],
                formula_id=pdf_book_formulas["formulas"][0]["formula_id"],
            )
            assert pdf_book_formula["formula"]["latex"]

        render = service.render_pdf_page(doc_id=result["doc_id"], page=1, dpi=96)
        assert Path(render["image_path"]).exists()
    finally:
        service.close()


def test_e2e_epub_book_ingest_read_images_and_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples_root, _, service = _bootstrap_e2e_service(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        collection_suffix="epub_book",
    )
    try:
        library_root = tmp_path / "library"
        book_epub = _prepare_sample_document(
            samples_root=samples_root,
            target_root=library_root,
            relative_path=EPUB_BOOK_RELATIVE,
        )

        scan = service.library_scan(str(library_root), ["**/*.epub"])
        discovered = len(scan["added"]) + len(scan["updated"]) + scan["unchanged_count"]
        assert discovered == 1

        sidecars = service.storage_list_sidecars(root=str(library_root), limit=20)
        assert sidecars["documents_count"] == 1

        queued = service.document_ingest_epub_book(
            doc_id=None,
            path=str(book_epub),
            force=True,
        )
        result = _wait_for_successful_job(
            service,
            doc_id=queued["doc_id"],
            job_id=queued["job_id"],
        )
        assert result["chunks_count"] > 0

        outline = service.get_outline(result["doc_id"])
        assert isinstance(outline["nodes"], list) and outline["nodes"]
        node_id = outline["nodes"][0]["id"]

        node_read = service.read_outline_node(
            doc_id=result["doc_id"],
            node_id=node_id,
            out_format="text",
            max_chunks=5,
        )
        assert node_read["content"].strip()

        epub_images = service.epub_list_images(
            doc_id=result["doc_id"],
            node_id=node_id,
            limit=20,
        )
        assert epub_images["images_count"] >= 0
        if epub_images["images_count"] > 0:
            epub_image = service.epub_read_image(
                doc_id=result["doc_id"],
                image_id=epub_images["images"][0]["image_id"],
            )
            assert Path(epub_image["image"]["image_path"]).exists()

        deleted = service.storage_delete_document(
            doc_id=result["doc_id"],
            path=None,
            remove_artifacts=True,
        )
        assert deleted["deleted_records"] == 1

        final_sidecars = service.storage_list_sidecars(root=str(library_root), limit=20)
        assert final_sidecars["documents_count"] == 0
    finally:
        service.close()


def test_e2e_pdf_paper_ingest_outline_formulas_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples_root, _, service = _bootstrap_e2e_service(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        collection_suffix="pdf_paper",
    )
    try:
        library_root = tmp_path / "library"
        paper_pdf = _prepare_sample_document(
            samples_root=samples_root,
            target_root=library_root,
            relative_path=PDF_PAPER_RELATIVE,
        )

        scan = service.library_scan(str(library_root), ["**/*.pdf"])
        discovered = len(scan["added"]) + len(scan["updated"]) + scan["unchanged_count"]
        assert discovered == 1

        sidecars = service.storage_list_sidecars(root=str(library_root), limit=20)
        assert sidecars["documents_count"] == 1

        queued = service.document_ingest_pdf_paper(
            doc_id=None,
            path=str(paper_pdf),
            force=True,
        )
        result = _wait_for_successful_job(
            service,
            doc_id=queued["doc_id"],
            job_id=queued["job_id"],
        )
        assert "grobid" in result["parser_chain"]

        outline = service.get_outline(result["doc_id"])
        assert isinstance(outline["nodes"], list) and outline["nodes"]
        node_id = outline["nodes"][0]["id"]

        scoped_query = _first_words(outline["nodes"][0]["title"])
        scoped_hits = service.search_in_outline_node(
            doc_id=result["doc_id"],
            node_id=node_id,
            query=scoped_query,
            top_k=3,
        )["hits"]
        assert isinstance(scoped_hits, list)

        pdf_paper_formulas = service.pdf_paper_list_formulas(
            doc_id=result["doc_id"],
            node_id=node_id,
            limit=20,
            status=None,
        )
        assert pdf_paper_formulas["formulas_count"] >= 0
        if pdf_paper_formulas["formulas_count"] > 0:
            pdf_paper_formula = service.pdf_paper_read_formula(
                doc_id=result["doc_id"],
                formula_id=pdf_paper_formulas["formulas"][0]["formula_id"],
            )
            assert pdf_paper_formula["formula"]["latex"]

        paper_pdf.unlink()
        cleaned = service.storage_cleanup_sidecars(
            root=str(library_root),
            remove_missing_documents=True,
            remove_orphan_artifacts=True,
            compact_catalog=False,
        )
        assert cleaned["removed_deleted_count"] >= 1

        final_sidecars = service.storage_list_sidecars(root=str(library_root), limit=20)
        assert final_sidecars["documents_count"] == 0
    finally:
        service.close()
