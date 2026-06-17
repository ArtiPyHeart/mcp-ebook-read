from __future__ import annotations

import json
import logging
import os
import hashlib
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.render.pdf_visuals import DoclingPdfVisualExtractor
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    ExtractedImage,
    FormulaRecord,
    ImageRecord,
    IngestJobStatus,
    IngestStage,
    Locator,
    OutlineNode,
    ParsedDocument,
    PdfFigureRecord,
    PdfParserPerformanceConfig,
    PdfParserTuningProfile,
    PdfTableRecord,
    Profile,
    TableSegmentRecord,
)
from mcp_ebook_read.service import AppService
from mcp_ebook_read.store.catalog import CatalogStore


class RecordingParser:
    def __init__(
        self, result: ParsedDocument | None = None, error: Exception | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.close_calls = 0

    def parse(self, path: str, doc_id: str) -> ParsedDocument:
        self.calls.append((path, doc_id))
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("parser result not configured")
        return self.result

    def autotune(
        self,
        *,
        pdf_path: str,
        sample_pages: int,
        total_memory_bytes: int | None,
        candidate_configs=None,  # noqa: ANN001
    ) -> PdfParserTuningProfile:
        selected = PdfParserPerformanceConfig(
            num_threads=8,
            ocr_batch_size=6,
            layout_batch_size=6,
            table_batch_size=6,
        )
        return PdfParserTuningProfile(
            created_at="2026-03-15T00:00:00Z",
            source_path=pdf_path,
            sample_pages=sample_pages,
            cpu_count=8,
            total_memory_bytes=total_memory_bytes,
            selected_config=selected,
            benchmarks=[],
        )

    def set_performance_config(self, config: PdfParserPerformanceConfig) -> None:
        self.performance_config = config

    def close(self) -> None:
        self.close_calls += 1


class BlockingParser(RecordingParser):
    def __init__(self, result: ParsedDocument) -> None:
        super().__init__(result=result)
        self.started = threading.Event()
        self.release = threading.Event()

    def parse(self, path: str, doc_id: str) -> ParsedDocument:
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("blocking parser was not released")
        return super().parse(path, doc_id)


class RecordingGrobid:
    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        outline: list[OutlineNode] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.metadata = metadata or {}
        self.outline = outline or []
        self.error = error
        self.base_url = "http://grobid.test"
        self.calls: list[str] = []

    def assert_available(self) -> None:
        if self.error is not None:
            raise self.error

    def parse_fulltext(self, path: str):
        self.calls.append(path)
        if self.error is not None:
            raise self.error

        class Result:
            def __init__(
                self, metadata: dict[str, Any], outline: list[OutlineNode]
            ) -> None:
                self.metadata = metadata
                self.outline = outline

        return Result(self.metadata, self.outline)


class RecordingPdfImageExtractor:
    def __init__(self, images: list[ImageRecord] | None = None) -> None:
        self.images = images or []
        self.calls: list[dict[str, Any]] = []

    def extract(
        self,
        *,
        pdf_path: str,
        doc_id: str,
        chunks: list[ChunkRecord],
        out_dir: Path,
    ) -> list[ImageRecord]:
        self.calls.append(
            {
                "pdf_path": pdf_path,
                "doc_id": doc_id,
                "chunks_count": len(chunks),
                "out_dir": str(out_dir),
            }
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        generated: list[ImageRecord] = []
        for image in self.images:
            image_path = out_dir / f"{image.image_id}.png"
            image_path.write_bytes(b"\x89PNG")
            generated.append(image.model_copy(update={"file_path": str(image_path)}))
        return generated


class RecordingPdfVisualExtractor:
    def __init__(
        self,
        *,
        tables: list[PdfTableRecord] | None = None,
        figures: list[PdfFigureRecord] | None = None,
    ) -> None:
        self.tables = tables or []
        self.figures = figures or []
        self.calls: list[dict[str, Any]] = []

    def extract(
        self,
        *,
        pdf_path: str,
        doc_id: str,
        chunks: list[ChunkRecord],
        tables_dir: Path,
        figures_dir: Path,
    ):
        self.calls.append(
            {
                "pdf_path": pdf_path,
                "doc_id": doc_id,
                "chunks_count": len(chunks),
                "tables_dir": str(tables_dir),
                "figures_dir": str(figures_dir),
            }
        )
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        generated_tables: list[PdfTableRecord] = []
        generated_figures: list[PdfFigureRecord] = []

        for table in self.tables:
            image_path = tables_dir / f"{table.table_id}.png"
            image_path.write_bytes(b"\x89PNG")
            segments = []
            for idx, segment in enumerate(table.segments):
                segment_path = tables_dir / f"{table.table_id}_segment_{idx}.png"
                segment_path.write_bytes(b"\x89PNG")
                segments.append(
                    segment.model_copy(update={"file_path": str(segment_path)})
                )
            generated_tables.append(
                table.model_copy(
                    update={
                        "file_path": str(image_path),
                        "segments": segments,
                    }
                )
            )

        for figure in self.figures:
            image_path = figures_dir / f"{figure.figure_id}.png"
            image_path.write_bytes(b"\x89PNG")
            generated_figures.append(
                figure.model_copy(update={"file_path": str(image_path)})
            )

        class Result:
            def __init__(
                self,
                tables: list[PdfTableRecord],
                figures: list[PdfFigureRecord],
            ) -> None:
                self.tables = tables
                self.figures = figures

        return Result(generated_tables, generated_figures)


def _chunk(
    doc_id: str, chunk_id: str, idx: int, method: str = "docling"
) -> ChunkRecord:
    locator = Locator(
        doc_id=doc_id,
        chunk_id=chunk_id,
        section_path=[f"S{idx}"],
        page_range=[1, 1],
        method=method,
    )
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        order_index=idx,
        section_path=[f"S{idx}"],
        text=f"text-{idx}",
        search_text=f"text-{idx}",
        locator=locator,
        method=method,
    )


def _formula(
    *,
    doc_id: str,
    formula_id: str,
    chunk_id: str | None,
    page: int | None,
    bbox: list[float] | None,
    latex: str,
    status: str = "resolved",
    source: str = "pix2text",
) -> FormulaRecord:
    return FormulaRecord(
        formula_id=formula_id,
        doc_id=doc_id,
        chunk_id=chunk_id,
        section_path=["S0"],
        page=page,
        bbox=bbox,
        latex=latex,
        source=source,
        status=status,
    )


def _pdf_table(
    *,
    doc_id: str,
    table_id: str,
    page_range: list[int],
    section_path: list[str],
    caption: str | None = None,
    merged: bool = False,
    merge_confidence: float | None = None,
) -> PdfTableRecord:
    return PdfTableRecord(
        table_id=table_id,
        doc_id=doc_id,
        order_index=0,
        section_path=section_path,
        page_range=page_range,
        bbox=[10.0, 10.0, 100.0, 100.0] if not merged else None,
        caption=caption,
        headers=["Name", "Value"],
        rows=[["alpha", "1"], ["beta", "2"]],
        markdown="| Name | Value |\n| --- | --- |\n| alpha | 1 |\n| beta | 2 |",
        html="<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody><tr><td>alpha</td><td>1</td></tr><tr><td>beta</td><td>2</td></tr></tbody></table>",
        file_path="placeholder.png",
        width=400,
        height=300,
        merged=merged,
        merge_confidence=merge_confidence,
        segments=[
            TableSegmentRecord(
                page=page_range[0],
                bbox=[10.0, 10.0, 100.0, 100.0],
                caption=caption,
                file_path="placeholder-segment.png",
                width=400,
                height=300,
            )
        ],
    )


def _pdf_figure(
    *,
    doc_id: str,
    figure_id: str,
    page: int,
    section_path: list[str],
    caption: str | None = None,
    kind: str = "picture",
) -> PdfFigureRecord:
    return PdfFigureRecord(
        figure_id=figure_id,
        doc_id=doc_id,
        order_index=0,
        section_path=section_path,
        page=page,
        bbox=[5.0, 5.0, 90.0, 90.0],
        caption=caption,
        kind=kind,
        file_path="placeholder.png",
        width=240,
        height=160,
    )


def _parsed(
    doc_id: str,
    *,
    title: str,
    method: str,
    images: list[ExtractedImage] | None = None,
    formulas: list[FormulaRecord] | None = None,
    outline: list[OutlineNode] | None = None,
) -> ParsedDocument:
    chunks = [_chunk(doc_id, "c0", 0, method), _chunk(doc_id, "c1", 1, method)]
    effective_outline = outline or [OutlineNode(id="n1", title="Section", level=1)]
    return ParsedDocument(
        title=title,
        parser_chain=[method],
        metadata={"source": method},
        outline=effective_outline,
        chunks=chunks,
        formulas=formulas or [],
        images=images or [],
        reading_markdown="# Section\n\ntext",
        overall_confidence=0.9,
    )


def _build_service(
    tmp_path: Path,
    *,
    pdf_parser: Any,
    epub_parser: RecordingParser,
    grobid: RecordingGrobid,
    pdf_image_extractor: RecordingPdfImageExtractor | None = None,
    pdf_visual_extractor: RecordingPdfVisualExtractor | None = None,
    pdf_parse_timeout_seconds: int = 1800,
    ingest_worker_count: int = 1,
) -> AppService:
    return AppService(
        sidecar_dir_name=".mcp-ebook-read",
        default_library_root=tmp_path,
        pdf_parser=pdf_parser,
        pdf_image_extractor=pdf_image_extractor,
        pdf_visual_extractor=pdf_visual_extractor,
        grobid_client=grobid,
        epub_parser=epub_parser,
        pdf_parse_timeout_seconds=pdf_parse_timeout_seconds,
        ingest_worker_count=ingest_worker_count,
    )


def _register_doc(
    service: AppService,
    doc_id: str,
    path: Path,
    doc_type: DocumentType,
    *,
    root: Path | None = None,
    profile: Profile = Profile.BOOK,
) -> None:
    catalog = service._catalog_for_document_path(path, root=root)
    catalog.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path.resolve()),
            type=doc_type,
            sha256=("a" if doc_type == DocumentType.PDF else "b") * 64,
            mtime=1.0,
            profile=profile,
        )
    )
    service._bind_doc_catalog(doc_id, catalog)


def _wait_for_ingest_job(
    service: AppService,
    *,
    doc_id: str,
    job_id: str,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = service.document_ingest_status(doc_id=doc_id, job_id=job_id)
        if last["status"] in {
            IngestJobStatus.SUCCEEDED,
            IngestJobStatus.FAILED,
            IngestJobStatus.CANCELED,
        }:
            if (
                last["status"] != IngestJobStatus.SUCCEEDED
                or last.get("result") is not None
            ):
                return last
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for ingest job {job_id}: {last}")


def _wait_for_catalog_ingest_job(
    catalog,  # noqa: ANN001
    *,
    doc_id: str,
    job_id: str,
    timeout_seconds: float = 5.0,
):
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        last = catalog.get_ingest_job(doc_id, job_id)
        if last is not None and last.status in {
            IngestJobStatus.SUCCEEDED,
            IngestJobStatus.FAILED,
            IngestJobStatus.CANCELED,
        }:
            return last
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for catalog ingest job {job_id}: {last}")


def test_parse_pdf_document_uses_isolated_worker_for_docling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = "doc-worker-pdf"
    pdf_path = tmp_path / "worker.pdf"
    pdf_path.write_bytes(b"pdf")
    parsed = _parsed(doc_id, title="Worker PDF", method="docling")
    calls: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "command": command,
                "input": json.loads(input),
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"ok": True, "data": parsed.model_dump(mode="json")},
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    service = _build_service(
        tmp_path,
        pdf_parser=DoclingPdfParser(
            enable_docling_formula_enrichment=False,
            require_formula_engine=False,
            formula_batch_size=2,
            performance_config=PdfParserPerformanceConfig(num_threads=3),
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_parse_timeout_seconds=12,
    )

    result = service._parse_pdf_document(str(pdf_path), doc_id)

    assert result.title == "Worker PDF"
    assert calls[0]["timeout"] == 12
    assert calls[0]["command"][:3] == [
        sys.executable,
        "-m",
        "mcp_ebook_read.workers.pdf_parse",
    ]
    assert calls[0]["input"]["formula_batch_size"] == 2
    assert calls[0]["input"]["performance_config"]["num_threads"] == 3


def test_parse_pdf_document_embeds_docling_visual_extraction_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = "doc-worker-pdf-visuals"
    pdf_path = tmp_path / "worker-visuals.pdf"
    pdf_path.write_bytes(b"pdf")
    parsed = _parsed(doc_id, title="Worker PDF Visuals", method="docling")
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    calls: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "command": command,
                "input": json.loads(input),
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"ok": True, "data": parsed.model_dump(mode="json")},
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    service = _build_service(
        tmp_path,
        pdf_parser=DoclingPdfParser(
            enable_docling_formula_enrichment=False,
            require_formula_engine=False,
            performance_config=PdfParserPerformanceConfig(num_threads=4),
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_visual_extractor=DoclingPdfVisualExtractor(
            performance_config=PdfParserPerformanceConfig(num_threads=4),
            images_scale=3.5,
        ),
    )

    result = service._parse_pdf_document(
        str(pdf_path),
        doc_id,
        visual_tables_dir=tables_dir,
        visual_figures_dir=figures_dir,
    )

    assert result.title == "Worker PDF Visuals"
    worker_input = calls[0]["input"]
    assert worker_input["visual_extraction"] == {
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "images_scale": 3.5,
    }
    assert worker_input["resource_plan"]["docling_visuals_in_parse_worker"] is True
    assert (
        result.metadata["pdf_parse_resource_plan"]["docling_visuals_in_parse_worker"]
        is True
    )


def test_parse_pdf_document_timeout_returns_structured_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = "doc-worker-timeout"
    pdf_path = tmp_path / "timeout.pdf"
    pdf_path.write_bytes(b"pdf")

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    service = _build_service(
        tmp_path,
        pdf_parser=DoclingPdfParser(
            enable_docling_formula_enrichment=False,
            require_formula_engine=False,
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_parse_timeout_seconds=1,
    )

    with pytest.raises(AppError) as exc:
        service._parse_pdf_document(str(pdf_path), doc_id)

    assert exc.value.code == ErrorCode.INGEST_PDF_DOCLING_TIMEOUT
    assert exc.value.details["timeout_seconds"] == 1
    assert exc.value.details["doc_id"] == doc_id


def test_extract_pdf_visuals_uses_isolated_worker_for_docling_visuals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = "doc-worker-visuals"
    pdf_path = tmp_path / "visuals.pdf"
    pdf_path.write_bytes(b"pdf")
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    table = _pdf_table(
        doc_id=doc_id,
        table_id="worker-table",
        page_range=[1, 1],
        section_path=["Section"],
        caption="Worker table",
    )
    figure = _pdf_figure(
        doc_id=doc_id,
        figure_id="worker-figure",
        page=2,
        section_path=["Section"],
        caption="Worker figure",
    )
    calls: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        *,
        input: str,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "command": command,
                "input": json.loads(input),
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "data": {
                        "tables": [table.model_dump(mode="json")],
                        "figures": [figure.model_dump(mode="json")],
                        "diagnostics": {"summary": {"tables_returned": 1}},
                    },
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_visual_extractor=DoclingPdfVisualExtractor(
            performance_config=PdfParserPerformanceConfig(num_threads=5),
            images_scale=3.0,
        ),
        pdf_parse_timeout_seconds=13,
    )

    extracted = service._extract_pdf_visuals(
        pdf_path=str(pdf_path),
        doc_id=doc_id,
        chunks=[_chunk(doc_id, "chunk-worker-visuals", 0, "docling")],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )

    assert extracted.tables[0].table_id == "worker-table"
    assert extracted.figures[0].figure_id == "worker-figure"
    assert extracted.diagnostics["summary"]["tables_returned"] == 1
    assert calls[0]["timeout"] == 13
    assert calls[0]["command"][:3] == [
        sys.executable,
        "-m",
        "mcp_ebook_read.workers.pdf_visuals",
    ]
    assert str(tables_dir) in calls[0]["command"]
    assert str(figures_dir) in calls[0]["command"]
    assert calls[0]["input"]["performance_config"]["num_threads"] == 5
    assert calls[0]["input"]["images_scale"] == 3.0
    assert calls[0]["input"]["chunks"][0]["chunk_id"] == "chunk-worker-visuals"


def test_library_scan_invalid_root(tmp_path: Path) -> None:
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    with pytest.raises(AppError) as exc:
        service.library_scan(root=str(tmp_path / "missing"), patterns=["**/*.pdf"])

    assert exc.value.code == ErrorCode.SCAN_INVALID_ROOT


def test_library_scan_add_unchanged_removed(tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()
    pdf = root / "a.pdf"
    epub = root / "b.epub"
    txt = root / "c.txt"
    pdf.write_bytes(b"pdf-v1")
    epub.write_bytes(b"epub-v1")
    txt.write_text("ignored", encoding="utf-8")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    first = service.library_scan(str(root), ["**/*"])
    assert len(first["added"]) == 2
    assert first["unchanged_count"] == 0
    assert first["removed_deleted_count"] == 0
    assert first["storage_maintenance"]["catalogs"]
    assert first["scan_performance"]["candidate_documents"] == 2
    assert first["scan_performance"]["hash_workers"] >= 1
    assert first["scan_performance"]["hash_seconds"] >= 0
    assert first["scan_performance"]["total_seconds"] >= 0

    second = service.library_scan(str(root), ["**/*.pdf", "**/*.epub"])
    assert second["unchanged_count"] == 2
    assert second["removed_deleted_count"] == 0

    epub.unlink()
    third = service.library_scan(str(root), ["**/*.pdf", "**/*.epub"])
    assert len(third["removed"]) == 1
    assert third["removed"][0] == str(epub.resolve())
    assert third["removed_deleted_count"] == 1
    scan_catalog = service._catalog_for_document_path(epub)
    assert scan_catalog.get_document_by_path(str(epub.resolve())) is None


def test_library_scan_allows_duplicate_file_content(tmp_path: Path) -> None:
    root = tmp_path / "library"
    first_dir = root / "alpha"
    second_dir = root / "beta"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    payload = b"same-pdf-bytes"
    first_pdf = first_dir / "duplicate.pdf"
    second_pdf = second_dir / "duplicate.pdf"
    first_pdf.write_bytes(payload)
    second_pdf.write_bytes(payload)

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    first = service.library_scan(str(root), ["**/*.pdf"])
    expected_content_prefix = hashlib.sha256(payload).hexdigest()[:16]
    doc_ids = {row["doc_id"] for row in first["added"]}

    assert len(first["added"]) == 2
    assert first["scan_performance"]["candidate_documents"] == 2
    assert len(doc_ids) == 2
    assert all(doc_id.startswith(f"{expected_content_prefix}-") for doc_id in doc_ids)
    assert all(len(doc_id) == 25 for doc_id in doc_ids)

    catalog = service._catalog_for_library_root(root)
    documents = catalog.list_documents()
    assert len(documents) == 2
    assert {doc.path for doc in documents} == {
        str(first_pdf.resolve()),
        str(second_pdf.resolve()),
    }

    second = service.library_scan(str(root), ["**/*.pdf"])
    assert second["unchanged_count"] == 2
    assert not second["added"]


def test_library_scan_infers_paper_profile_from_papers_path(tmp_path: Path) -> None:
    root = tmp_path / "library"
    papers_dir = root / "papers" / "market-microstructure"
    books_dir = root / "books"
    papers_dir.mkdir(parents=True)
    books_dir.mkdir(parents=True)
    paper_pdf = papers_dir / "paper.pdf"
    book_pdf = books_dir / "book.pdf"
    paper_pdf.write_bytes(b"paper")
    book_pdf.write_bytes(b"book")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    result = service.library_scan(str(root), ["**/*.pdf"])
    payload_by_path = {row["path"]: row for row in result["added"]}

    assert payload_by_path[str(paper_pdf.resolve())]["profile"] == Profile.PAPER
    assert payload_by_path[str(book_pdf.resolve())]["profile"] == Profile.BOOK

    catalog = service._catalog_for_library_root(root)
    paper_doc = catalog.get_document_by_path(str(paper_pdf.resolve()))
    book_doc = catalog.get_document_by_path(str(book_pdf.resolve()))
    assert paper_doc is not None
    assert paper_doc.profile == Profile.PAPER
    assert book_doc is not None
    assert book_doc.profile == Profile.BOOK


def test_library_scan_hash_worker_count_scales_with_safe_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        ingest_worker_count=5,
    )

    assert service._scan_worker_count(0) == 1
    assert service._scan_worker_count(1) == 1
    assert service._scan_worker_count(6) == 6
    assert service._scan_worker_count(100) == 8


def test_library_scan_routes_nested_documents_to_root_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "library"
    books_dir = root / "books"
    papers_dir = root / "papers"
    books_dir.mkdir(parents=True)
    papers_dir.mkdir(parents=True)

    book_pdf = books_dir / "book.pdf"
    paper_pdf = papers_dir / "paper.pdf"
    book_pdf.write_bytes(b"book")
    paper_pdf.write_bytes(b"paper")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    result = service.library_scan(str(root), ["**/*.pdf"])
    assert len(result["added"]) == 2

    book_catalog = service._catalog_for_document_path(book_pdf)
    paper_catalog = service._catalog_for_document_path(paper_pdf)
    assert book_catalog.db_path.parent == root / ".mcp-ebook-read"
    assert paper_catalog.db_path.parent == root / ".mcp-ebook-read"
    assert not (books_dir / ".mcp-ebook-read").exists()
    assert not (papers_dir / ".mcp-ebook-read").exists()
    assert book_catalog.get_document_by_path(str(book_pdf.resolve())) is not None
    assert paper_catalog.get_document_by_path(str(paper_pdf.resolve())) is not None

    sidecars = service.storage_list_sidecars(root=str(root), limit=20)
    assert sidecars["sidecars_count"] == 1
    assert sidecars["documents_count"] == 2


def test_storage_list_defaults_to_root_sidecar_and_ignores_nested_legacy_sidecar(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    nested = root / "nested"
    nested.mkdir(parents=True)
    root_doc_path = root / "root-book.pdf"
    legacy_doc_path = nested / "legacy-book.pdf"
    root_doc_path.write_bytes(b"root")
    legacy_doc_path.write_bytes(b"legacy")

    service = _build_service(
        root,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    root_catalog = service._catalog_for_library_root()
    root_catalog.upsert_scanned_document(
        DocumentRecord(
            doc_id="doc-root-sidecar",
            path=str(root_doc_path.resolve()),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    legacy_catalog = CatalogStore(nested / ".mcp-ebook-read" / "catalog.db")
    legacy_catalog.upsert_scanned_document(
        DocumentRecord(
            doc_id="doc-legacy-nested-sidecar",
            path=str(legacy_doc_path.resolve()),
            type=DocumentType.PDF,
            sha256="b" * 64,
            mtime=1.0,
        )
    )

    sidecars = service.storage_list_sidecars(limit=20)

    assert sidecars["root"] == str(root.resolve())
    assert sidecars["sidecars_count"] == 1
    assert sidecars["documents_count"] == 1
    assert sidecars["sidecars"][0]["sidecar_path"] == str(root / ".mcp-ebook-read")
    assert sidecars["sidecars"][0]["documents"][0]["doc_id"] == "doc-root-sidecar"


def test_library_scan_cleans_catalog_when_folder_becomes_empty(tmp_path: Path) -> None:
    root = tmp_path / "library"
    books_dir = root / "books"
    books_dir.mkdir(parents=True)
    book_pdf = books_dir / "book.pdf"
    book_pdf.write_bytes(b"book")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    first = service.library_scan(str(root), ["**/*.pdf"])
    assert len(first["added"]) == 1
    doc_id = first["added"][0]["doc_id"]

    book_pdf.unlink()
    second = service.library_scan(str(root), ["**/*.pdf"])
    assert second["removed_deleted_count"] == 1
    catalog = service._catalog_for_document_path(book_pdf)
    assert catalog.get_document_by_id(doc_id) is None


def test_document_ingest_success(tmp_path: Path) -> None:
    doc_id = "docpdf1"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    parsed_pdf = _parsed(doc_id, title="PDF Book", method="docling")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed_pdf),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    queued = service.document_ingest(
        doc_id=doc_id,
        path=None,
        force=True,
    )
    result = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )["result"]

    assert result["doc_id"] == doc_id
    assert result["chunks_count"] == 2
    assert result["images_count"] == 0
    assert result["parser_chain"] == ["docling"]

    reading = tmp_path / ".mcp-ebook-read" / "docs" / doc_id / "reading" / "reading.md"
    assert reading.exists()

    loaded_catalog = service._catalog_for_document_path(pdf_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.status == DocumentStatus.READY
    assert loaded.title == "PDF Book"


def test_document_ingest_profile_override_paper_for_pdf_path_without_hint(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "research" / "attention.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"pdf")
    sha256 = hashlib.sha256(b"pdf").hexdigest()
    doc_id = AppService._scanned_doc_id(pdf_path, sha256)
    grobid = RecordingGrobid(metadata={"paper_title": "Explicit Paper"})
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF Document", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=grobid,
    )

    queued = service.document_ingest(
        path=str(pdf_path),
        force=True,
        profile="paper",
    )
    result = _wait_for_ingest_job(
        service,
        doc_id=queued["doc_id"],
        job_id=queued["job_id"],
    )["result"]

    assert queued["doc_id"] == doc_id
    assert queued["profile"] == Profile.PAPER
    assert result["profile"] == Profile.PAPER
    assert grobid.calls == [str(pdf_path)]
    catalog = service._catalog_for_document_path(pdf_path)
    loaded = catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.profile == Profile.PAPER
    assert loaded.title == "Explicit Paper"


def test_document_ingest_rejects_invalid_profile(tmp_path: Path) -> None:
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    with pytest.raises(AppError) as exc:
        service.document_ingest(path=str(pdf_path), profile="article")

    assert exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE
    assert exc.value.details["supported_profiles"] == ["auto", "book", "paper"]


def test_document_ingest_rejects_paper_profile_for_epub(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    with pytest.raises(AppError) as exc:
        service.document_ingest(path=str(epub_path), profile="paper")

    assert exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE
    assert exc.value.details["supported_profiles"] == ["auto", "book"]


def test_document_ingest_persists_fast_preflight_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = "docpdf-fast"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")
    parsed_pdf = _parsed(doc_id, title="PDF Book", method="docling")
    parsed_pdf.metadata = {
        "pages": 2,
        "formula_markers_total": 3,
        "formula_unresolved": 0,
        "pdf_tables_count": 1,
        "pdf_figures_count": 1,
        "pdf_parse_phase_seconds": {"total": 0.5, "docling_convert": 0.3},
    }
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed_pdf),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    class FakeFastParser:
        method = "pypdfium2-fast"

        def parse(self, path: str, parsed_doc_id: str) -> ParsedDocument:
            assert path == str(pdf_path.resolve())
            assert parsed_doc_id == doc_id
            chunks = [
                ChunkRecord(
                    chunk_id="fast-c0",
                    doc_id=parsed_doc_id,
                    order_index=0,
                    section_path=["Fast"],
                    text="fast parser text",
                    search_text="fast parser text",
                    locator=Locator(
                        doc_id=parsed_doc_id,
                        chunk_id="fast-c0",
                        section_path=["Fast"],
                        page_range=[1, 1],
                        method=self.method,
                    ),
                    method=self.method,
                )
            ]
            return ParsedDocument(
                title="Fast",
                parser_chain=[self.method],
                metadata={
                    "pages": 2,
                    "pdf_parse_phase_seconds": {"total": 0.01},
                },
                outline=[OutlineNode(id="fast-n1", title="Fast", level=1)],
                chunks=chunks,
                reading_markdown="fast parser text",
            )

    monkeypatch.setattr("mcp_ebook_read.service.Pypdfium2PdfParser", FakeFastParser)
    monkeypatch.setattr(
        AppService,
        "_pdf_diagnostic_preflight",
        lambda self, path, parsed_doc_id: {
            "status": "ok",
            "parser": "pymupdf",
            "pages": 2,
            "outline_nodes": 1,
            "images": 4,
            "text_blocks": 12,
            "page_samples": [
                {"page": 1, "images": 2, "text_blocks": 6},
                {"page": 2, "images": 2, "text_blocks": 6},
            ],
            "seconds": 0.02,
        },
    )

    queued = service.document_ingest(
        doc_id=doc_id,
        path=None,
        force=True,
    )
    completed = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )

    result = completed["result"]
    lanes = result["pdf_parser_lanes"]
    assert result["pdf_parse_phase_seconds"] == {
        "total": 0.5,
        "docling_convert": 0.3,
    }
    assert lanes["strategy"]["fast_lane"] == "pypdfium2_pdf"
    assert lanes["strategy"]["diagnostic_lane"] == "pymupdf"
    assert lanes["strategy"]["canonical_content_source"] == "docling"
    assert lanes["fast_preflight"]["status"] == "ok"
    assert lanes["fast_preflight"]["parser"] == "pypdfium2-fast"
    assert lanes["fast_preflight"]["preflight_execution"]["mode"] == "parallel"
    assert lanes["fast_preflight"]["preflight_execution"]["lanes"] == [
        "pypdfium2_fast",
        "pymupdf_diagnostic",
    ]
    assert lanes["diagnostic_preflight"]["status"] == "ok"
    assert lanes["diagnostic_preflight"]["preflight_execution"]["mode"] == "parallel"
    assert lanes["diagnostic_preflight"]["images"] == 4
    assert lanes["diagnostic_preflight"]["text_blocks"] == 12
    assert lanes["canonical_fidelity"]["parser_chain"] == ["docling"]
    assert lanes["canonical_fidelity"]["formula_markers_total"] == 3
    assert lanes["fast_text_ratio_to_fidelity"] is not None

    loaded = service._catalog_for_document_path(pdf_path).get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.metadata["pdf_parser_lanes"]["fast_preflight"]["status"] == "ok"
    assert (
        loaded.metadata["pdf_parser_lanes"]["diagnostic_preflight"]["parser"]
        == "pymupdf"
    )
    cached = service.document_ingest(doc_id=doc_id, path=None, force=False)
    assert cached["cached"] is True
    assert cached["result"]["pdf_parse_phase_seconds"] == {
        "total": 0.5,
        "docling_convert": 0.3,
    }


def test_document_ingest_keeps_docling_when_fast_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = "docpdf-fast-error"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF Book", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    class BrokenFastParser:
        method = "pypdfium2-fast"

        def parse(self, path: str, parsed_doc_id: str) -> ParsedDocument:
            raise RuntimeError("fast parser failed")

    monkeypatch.setattr("mcp_ebook_read.service.Pypdfium2PdfParser", BrokenFastParser)

    queued = service.document_ingest(
        doc_id=doc_id,
        path=None,
        force=True,
    )
    completed = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )

    assert completed["status"] == IngestJobStatus.SUCCEEDED
    lanes = completed["result"]["pdf_parser_lanes"]
    assert lanes["fast_preflight"]["status"] == "error"
    assert lanes["fast_preflight"]["error"]["message"] == "fast parser failed"
    assert lanes["canonical_fidelity"]["parser_chain"] == ["docling"]


def test_document_ingest_pdf_visual_extractors_run_concurrently(
    tmp_path: Path,
) -> None:
    doc_id = "docpdf-visual-concurrent"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")
    image_started = threading.Event()
    visual_started = threading.Event()

    class CoordinatedImageExtractor(RecordingPdfImageExtractor):
        def extract(self, **kwargs: Any) -> list[ImageRecord]:
            image_started.set()
            if not visual_started.wait(timeout=2.0):
                raise AssertionError("visual extractor did not start concurrently")
            return super().extract(**kwargs)

    class CoordinatedVisualExtractor(RecordingPdfVisualExtractor):
        def extract(self, **kwargs: Any):  # noqa: ANN201
            visual_started.set()
            if not image_started.wait(timeout=2.0):
                raise AssertionError("image extractor did not start concurrently")
            return super().extract(**kwargs)

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF Book", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_image_extractor=CoordinatedImageExtractor(),
        pdf_visual_extractor=CoordinatedVisualExtractor(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    queued = service.document_ingest(
        doc_id=doc_id,
        path=None,
        force=True,
    )
    completed = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )

    assert completed["status"] == IngestJobStatus.SUCCEEDED
    assert image_started.is_set()
    assert visual_started.is_set()
    catalog = service._catalog_for_document_path(pdf_path)
    doc = catalog.get_document_by_id(doc_id)
    assert doc is not None
    workspace_dir = service._doc_workspace_dir(doc, catalog)
    images_manifest = json.loads(
        service._pdf_images_manifest_path(workspace_dir).read_text(encoding="utf-8")
    )
    visuals_manifest = json.loads(
        service._pdf_visuals_manifest_path(workspace_dir).read_text(encoding="utf-8")
    )
    assert images_manifest["execution"]["mode"] == "parallel"
    assert visuals_manifest["execution"]["mode"] == "parallel"
    assert set(visuals_manifest["execution"]["lanes"]) == {
        "pdf_images",
        "docling_tables_figures",
    }


def test_document_ingest_pdf_uses_embedded_docling_visuals_without_second_convert(
    tmp_path: Path,
) -> None:
    doc_id = "docpdf-embedded-visuals"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")
    table_path = tmp_path / "embedded-table.png"
    figure_path = tmp_path / "embedded-figure.png"
    table_path.write_bytes(b"\x89PNG")
    figure_path.write_bytes(b"\x89PNG")
    parsed_pdf = _parsed(doc_id, title="PDF Book", method="docling")
    parsed_pdf.metadata["_pdf_visuals_from_docling_document"] = {
        "tables": [
            PdfTableRecord(
                table_id="tbl-embedded",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter"],
                page_range=[1, 1],
                markdown="| A |\n| - |\n| 1 |",
                html="<table><tr><td>1</td></tr></table>",
                file_path=str(table_path),
            ).model_dump(mode="json")
        ],
        "figures": [
            PdfFigureRecord(
                figure_id="fig-embedded",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter"],
                page=1,
                file_path=str(figure_path),
            ).model_dump(mode="json")
        ],
        "diagnostics": {
            "extractor": "docling-visuals",
            "summary": {"tables_returned": 1, "figures_returned": 1},
        },
        "execution": {
            "mode": "single_docling_convert",
            "source": "docling_parse_worker",
        },
    }

    class FailingVisualExtractor(RecordingPdfVisualExtractor):
        def extract(self, **kwargs: Any):  # noqa: ANN201
            raise AssertionError("second Docling visual extraction should not run")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed_pdf),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_visual_extractor=FailingVisualExtractor(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    queued = service.document_ingest(
        doc_id=doc_id,
        path=None,
        force=True,
    )
    completed = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )

    assert completed["status"] == IngestJobStatus.SUCCEEDED
    catalog = service._catalog_for_document_path(pdf_path)
    assert len(catalog.list_pdf_tables(doc_id)) == 1
    assert len(catalog.list_pdf_figures(doc_id)) == 1
    loaded = catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert "_pdf_visuals_from_docling_document" not in loaded.metadata
    workspace_dir = service._doc_workspace_dir(loaded, catalog)
    visuals_manifest = json.loads(
        service._pdf_visuals_manifest_path(workspace_dir).read_text(encoding="utf-8")
    )
    assert visuals_manifest["execution"]["mode"] == "single_docling_convert"


def test_pdf_diagnostic_preflight_collects_pymupdf_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePage:
        def __init__(self, *, images: int, blocks: int) -> None:
            self.images = images
            self.blocks = blocks

        def get_images(self, *, full: bool) -> list[object]:
            assert full is True
            return [object()] * self.images

        def get_text(self, mode: str) -> list[tuple[object, ...]]:
            assert mode == "blocks"
            return [tuple()] * self.blocks

    class FakeDoc:
        metadata = {"title": "Diagnostic PDF"}
        page_count = 2

        def __enter__(self) -> "FakeDoc":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def __iter__(self):
            return iter(
                [
                    FakePage(images=2, blocks=3),
                    FakePage(images=1, blocks=4),
                ]
            )

        def get_toc(self) -> list[list[object]]:
            return [[1, "Intro", 1]]

    monkeypatch.setitem(
        sys.modules,
        "fitz",
        SimpleNamespace(open=lambda _path: FakeDoc()),
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    result = service._pdf_diagnostic_preflight("/tmp/x.pdf", "doc-diag")

    assert result["status"] == "ok"
    assert result["parser"] == "pymupdf"
    assert result["pages"] == 2
    assert result["outline_nodes"] == 1
    assert result["images"] == 3
    assert result["text_blocks"] == 7
    assert result["page_samples"] == [
        {"page": 1, "images": 2, "text_blocks": 3},
        {"page": 2, "images": 1, "text_blocks": 4},
    ]
    assert result["metadata"]["title"] == "Diagnostic PDF"


def test_pdf_diagnostic_preflight_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_open(_path: str) -> object:
        raise RuntimeError("broken pymupdf")

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=fail_open))
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    result = service._pdf_diagnostic_preflight("/tmp/x.pdf", "doc-diag")

    assert result["status"] == "error"
    assert result["parser"] == "pymupdf"
    assert result["error"]["message"] == "broken pymupdf"
    assert result["details"]["doc_id"] == "doc-diag"


def test_document_ingest_path_creates_sidecar_under_default_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "cwd"
    book_dir = tmp_path / "library" / "epub-books"
    cwd.mkdir()
    book_dir.mkdir(parents=True)
    epub_bytes = b"epub"
    epub_path = book_dir / "book.epub"
    epub_path.write_bytes(epub_bytes)
    doc_id = AppService._scanned_doc_id(
        epub_path, hashlib.sha256(epub_bytes).hexdigest()
    )

    epub_parser = RecordingParser(
        result=_parsed(doc_id, title="EPUB Book", method="ebooklib")
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=epub_parser,
        grobid=RecordingGrobid(),
    )

    monkeypatch.chdir(cwd)
    queued = service.document_ingest(
        doc_id=None,
        path=str(epub_path),
        force=True,
    )
    result = _wait_for_ingest_job(
        service,
        doc_id=queued["doc_id"],
        job_id=queued["job_id"],
    )["result"]

    assert result["doc_id"] == doc_id
    assert epub_parser.calls == [(str(epub_path), doc_id)]
    assert not (cwd / ".mcp-ebook-read").exists()
    assert not (book_dir / ".mcp-ebook-read").exists()
    assert (tmp_path / ".mcp-ebook-read" / "catalog.db").exists()
    assert (
        tmp_path / ".mcp-ebook-read" / "docs" / doc_id / "reading" / "reading.md"
    ).exists()


def test_document_ingest_path_uses_explicit_root_sidecar(tmp_path: Path) -> None:
    library_root = tmp_path / "library"
    book_dir = library_root / "nested" / "epub-books"
    book_dir.mkdir(parents=True)
    epub_bytes = b"epub"
    epub_path = book_dir / "book.epub"
    epub_path.write_bytes(epub_bytes)
    doc_id = AppService._scanned_doc_id(
        epub_path, hashlib.sha256(epub_bytes).hexdigest()
    )

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(
            result=_parsed(doc_id, title="EPUB Book", method="ebooklib")
        ),
        grobid=RecordingGrobid(),
    )

    queued = service.document_ingest(
        doc_id=None,
        path=str(epub_path),
        root=str(library_root),
        force=True,
    )
    _wait_for_ingest_job(service, doc_id=queued["doc_id"], job_id=queued["job_id"])

    assert (library_root / ".mcp-ebook-read" / "catalog.db").exists()
    assert not (book_dir / ".mcp-ebook-read").exists()


def test_document_ingest_stage_log_uses_safe_extra_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    doc_id = "docpdf-stage-log"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF Book", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    with caplog.at_level(logging.INFO, logger="mcp_ebook_read.service"):
        queued = service.document_ingest(
            doc_id=doc_id,
            path=None,
            force=True,
        )
        status = _wait_for_ingest_job(
            service,
            doc_id=doc_id,
            job_id=queued["job_id"],
        )

    assert status["status"] == IngestJobStatus.SUCCEEDED
    stage_records = [
        record for record in caplog.records if record.msg == "ingest_job_stage"
    ]
    assert stage_records
    assert any(
        getattr(record, "stage_message", None) == "Parsing PDF book with Docling."
        for record in stage_records
    )


def test_service_close_closes_pdf_parser_once(tmp_path: Path) -> None:
    parser = RecordingParser(result=_parsed("doc1", title="PDF", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(
            result=_parsed("doc2", title="EPUB", method="epub")
        ),
        grobid=RecordingGrobid(),
    )

    service.close()
    service.close()

    assert parser.close_calls == 1


def test_document_ingest_cached_ready_without_force(tmp_path: Path) -> None:
    doc_id = "docpdf2"
    pdf_path = tmp_path / "cached.pdf"
    pdf_path.write_bytes(b"pdf")

    parser = RecordingParser(result=_parsed(doc_id, title="Cached", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF, profile=Profile.PAPER)

    first = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=first["job_id"])
    second = service.document_ingest(doc_id=doc_id, path=None, force=False)

    assert second["doc_id"] == doc_id
    assert second["cached"] is True
    assert second["result"]["images_count"] == 0
    assert len(parser.calls) == 1


def test_document_ingest_merges_grobid(tmp_path: Path) -> None:
    doc_id = "docpaper1"
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"pdf")

    pdf_parsed = _parsed(doc_id, title="Paper", method="docling")
    grobid_outline = [OutlineNode(id="g1", title="Grobid Intro", level=1)]
    grobid = RecordingGrobid(
        metadata={"paper_title": "Paper Title", "doi": "10.1/abc"},
        outline=grobid_outline,
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=pdf_parsed),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=grobid,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF, profile=Profile.PAPER)

    queued = service.document_ingest(
        doc_id=doc_id,
        path=None,
        force=True,
    )
    result = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )["result"]

    assert result["parser_chain"] == ["docling", "grobid"]
    assert result["images_count"] == 0
    assert grobid.calls == [str(pdf_path.resolve())]

    loaded_catalog = service._catalog_for_document_path(pdf_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.title == "Paper Title"
    assert loaded.metadata["paper_title"] == "Paper Title"
    assert [node.title for node in loaded.outline] == ["Section"]


def test_document_ingest_without_grobid_succeeds_with_diagnostic(
    tmp_path: Path,
) -> None:
    doc_id = "docpaper-no-grobid"
    pdf_path = tmp_path / "paper-no-grobid.pdf"
    pdf_path.write_bytes(b"pdf")

    grobid = RecordingGrobid()
    grobid.base_url = ""
    fallback_chunks = [
        ChunkRecord(
            chunk_id="abstract-c0",
            doc_id=doc_id,
            order_index=0,
            section_path=["Abstract"],
            text="Abstract: This paper studies local metadata fallback.",
            search_text="local metadata fallback abstract",
            locator=Locator(
                doc_id=doc_id,
                chunk_id="abstract-c0",
                section_path=["Abstract"],
                page_range=[1, 1],
                method="docling",
            ),
            method="docling",
        ),
        ChunkRecord(
            chunk_id="refs-c0",
            doc_id=doc_id,
            order_index=1,
            section_path=["References"],
            text="References [1] Example local fallback paper.",
            search_text="references fallback paper",
            locator=Locator(
                doc_id=doc_id,
                chunk_id="refs-c0",
                section_path=["References"],
                page_range=[9, 9],
                method="docling",
            ),
            method="docling",
        ),
    ]
    fallback_parsed = ParsedDocument(
        title="Local Paper",
        parser_chain=["docling"],
        metadata={"source": "docling"},
        outline=[
            OutlineNode(id="abstract", title="Abstract", level=1),
            OutlineNode(id="references", title="References", level=1),
        ],
        chunks=fallback_chunks,
        reading_markdown="# Abstract\n\nThis paper studies local metadata fallback.",
        overall_confidence=0.9,
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=fallback_parsed),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=grobid,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF, profile=Profile.PAPER)

    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    result = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )["result"]

    assert result["parser_chain"] == ["docling"]
    assert grobid.calls == []

    loaded_catalog = service._catalog_for_document_path(pdf_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.title == "Local Paper"
    assert loaded.metadata["grobid_enrichment"]["status"] == "skipped"
    assert "GROBID_URL" in loaded.metadata["grobid_enrichment"]["reason"]
    assert loaded.metadata["local_paper_metadata_fallback"]["status"] == "used"
    assert loaded.metadata["paper_title"] == "Local Paper"
    assert loaded.metadata["abstract"] == "This paper studies local metadata fallback."
    assert loaded.metadata["references_text_evidence"][0]["chunk_id"] == "refs-c0"

    explored = service.document_explore(doc_id=doc_id, query="Local Paper", top_k=3)
    assert any(
        diagnostic["code"] == "GROBID_ENRICHMENT_SKIPPED"
        for diagnostic in explored["diagnostics"]
    )


def test_document_ingest_with_grobid_failure_uses_local_fallback(
    tmp_path: Path,
) -> None:
    doc_id = "docpaper-grobid-failed"
    pdf_path = tmp_path / "paper-grobid-failed.pdf"
    pdf_path.write_bytes(b"pdf")

    grobid = RecordingGrobid(error=RuntimeError("grobid unavailable"))
    fallback_parsed = _parsed(doc_id, title="Docling Fallback Paper", method="docling")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=fallback_parsed),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=grobid,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF, profile=Profile.PAPER)

    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    result = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )["result"]

    assert result["parser_chain"] == ["docling"]
    assert grobid.calls == [str(pdf_path.resolve())]

    loaded_catalog = service._catalog_for_document_path(pdf_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.title == "Docling Fallback Paper"
    assert loaded.metadata["grobid_enrichment"]["status"] == "failed"
    assert (
        loaded.metadata["grobid_enrichment"]["error"]["message"] == "grobid unavailable"
    )
    assert loaded.metadata["local_paper_metadata_fallback"]["status"] == "used"
    assert loaded.metadata["paper_title"] == "Docling Fallback Paper"

    explored = service.document_explore(
        doc_id=doc_id, query="Docling Fallback", top_k=3
    )
    assert any(
        diagnostic["code"] == "GROBID_ENRICHMENT_FAILED"
        and diagnostic["severity"] == "warning"
        for diagnostic in explored["diagnostics"]
    )


def test_document_ingest_failure_sets_failed_status(tmp_path: Path) -> None:
    doc_id = "docepub1"
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    parser_error = AppError(ErrorCode.INGEST_EPUB_PARSE_FAILED, "epub parse failed")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(error=parser_error),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, epub_path, DocumentType.EPUB)

    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    status = _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])
    assert status["status"] == IngestJobStatus.FAILED
    assert status["error"]["code"] == ErrorCode.INGEST_EPUB_PARSE_FAILED
    loaded_catalog = service._catalog_for_document_path(epub_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.status == DocumentStatus.FAILED


def test_document_ingest_validation_failure_preserves_previous_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = "doc-atomic-ingest"
    epub_path = tmp_path / "atomic.epub"
    epub_path.write_bytes(b"epub")

    old_parsed = _parsed(doc_id, title="Old Book", method="ebooklib").model_copy(
        update={
            "reading_markdown": "# Old\n\nold text",
            "raw_artifacts": {"chapter.xhtml": "<html>old</html>"},
        }
    )
    parser = RecordingParser(result=old_parsed)
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=parser,
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, epub_path, DocumentType.EPUB)

    first = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=first["job_id"])
    catalog = service._catalog_for_document_path(epub_path)
    loaded_before = catalog.get_document_by_id(doc_id)
    assert loaded_before is not None
    workspace_dir = service._doc_workspace_dir(loaded_before, catalog)
    old_raw_path = workspace_dir / "raw" / "chapter.xhtml"
    assert old_raw_path.read_text(encoding="utf-8") == "<html>old</html>"
    assert [chunk.chunk_id for chunk in catalog.list_chunks(doc_id)] == ["c0", "c1"]

    new_parsed = _parsed(doc_id, title="New Book", method="ebooklib").model_copy(
        update={
            "reading_markdown": "# New\n\nnew text",
            "raw_artifacts": {"chapter.xhtml": "<html>new</html>"},
        }
    )
    parser.result = new_parsed

    original_validate = CatalogStore.validate_document_graph

    def fail_validation(self: CatalogStore, target_doc_id: str) -> list[dict[str, Any]]:
        issues = original_validate(self, target_doc_id)
        if self.db_path.name.endswith(".tmp") and target_doc_id == doc_id:
            return [
                {
                    "severity": "error",
                    "code": "TEST_GRAPH_INVALID",
                    "message": "forced validation failure",
                    "metadata": {"doc_id": target_doc_id},
                }
            ]
        return issues

    monkeypatch.setattr(CatalogStore, "validate_document_graph", fail_validation)

    second = service.document_ingest(doc_id=doc_id, path=None, force=True)
    failed = _wait_for_ingest_job(service, doc_id=doc_id, job_id=second["job_id"])

    assert failed["status"] == IngestJobStatus.FAILED
    assert failed["error"]["code"] == ErrorCode.INGEST_GRAPH_VALIDATION_FAILED
    loaded_after = catalog.get_document_by_id(doc_id)
    assert loaded_after is not None
    assert loaded_after.status == DocumentStatus.FAILED
    assert loaded_after.title == "Old Book"
    assert old_raw_path.read_text(encoding="utf-8") == "<html>old</html>"
    assert [chunk.chunk_id for chunk in catalog.list_chunks(doc_id)] == ["c0", "c1"]
    assert not list(workspace_dir.parent.glob(f".{workspace_dir.name}.staging-*"))


def test_document_ingest_jobs_deduplicate_and_list_status(tmp_path: Path) -> None:
    doc_id = "doc-job-dedupe"
    pdf_path = tmp_path / "dedupe.pdf"
    pdf_path.write_bytes(b"pdf")

    parser = BlockingParser(result=_parsed(doc_id, title="Queued", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    first = service.document_ingest(doc_id=doc_id, path=None, force=True)
    assert first["progress"]["stage"] == IngestStage.QUEUED
    assert first["progress"]["done"] == 0
    assert first["progress"]["total"] == 6
    assert first["progress"]["current_item"]
    assert parser.started.wait(timeout=2.0)

    second = service.document_ingest(doc_id=doc_id, path=None, force=True)
    assert second["deduplicated"] is True
    assert second["job_id"] == first["job_id"]
    assert second["progress"]["total"] == 6

    in_progress = service.document_ingest_status(doc_id=doc_id, job_id=first["job_id"])
    assert in_progress["status"] in {
        IngestJobStatus.QUEUED,
        IngestJobStatus.RUNNING,
    }
    assert in_progress["progress"]["stage"] in {
        IngestStage.QUEUED,
        IngestStage.PARSE,
        IngestStage.GROBID,
    }
    assert 0 <= in_progress["progress"]["done"] <= in_progress["progress"]["total"]
    if in_progress["status"] == IngestJobStatus.RUNNING:
        assert isinstance(in_progress["progress"]["elapsed_ms"], int)

    parser.release.set()
    completed = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=first["job_id"],
    )
    assert completed["status"] == IngestJobStatus.SUCCEEDED
    assert completed["progress"]["stage"] == IngestStage.COMPLETED
    assert completed["progress"]["done"] == completed["progress"]["total"]
    assert completed["progress"]["pct"] == 100.0
    assert completed["progress"]["diagnostics"]["chunks_count"] == 2
    assert isinstance(completed["progress"]["elapsed_ms"], int)

    listed = service.document_ingest_list_jobs(doc_id=doc_id, limit=10)
    assert listed["doc_id"] == doc_id
    assert listed["jobs"][0]["job_id"] == first["job_id"]
    assert listed["jobs"][0]["progress"]["stage"] == IngestStage.COMPLETED


def test_document_ingest_worker_uses_queued_catalog_for_duplicate_doc_ids(
    tmp_path: Path,
) -> None:
    doc_id = "same-content-doc"
    first_path = tmp_path / "alpha" / "shared.pdf"
    second_path = tmp_path / "beta" / "shared.pdf"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_bytes(b"same pdf bytes")
    second_path.write_bytes(b"same pdf bytes")

    parser = RecordingParser(result=_parsed(doc_id, title="Shared", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, first_path, DocumentType.PDF, root=first_path.parent)
    first_catalog = service._catalog_for_document_path(
        first_path, root=first_path.parent
    )
    _register_doc(
        service, doc_id, second_path, DocumentType.PDF, root=second_path.parent
    )
    second_catalog = service._catalog_for_document_path(
        second_path, root=second_path.parent
    )

    first = service.document_ingest(
        path=str(first_path.resolve()),
        doc_id=None,
        root=str(first_path.parent),
        force=True,
    )
    second = service.document_ingest(
        path=str(second_path.resolve()),
        doc_id=None,
        root=str(second_path.parent),
        force=True,
    )

    first_job = _wait_for_catalog_ingest_job(
        first_catalog,
        doc_id=doc_id,
        job_id=first["job_id"],
    )
    second_job = _wait_for_catalog_ingest_job(
        second_catalog,
        doc_id=doc_id,
        job_id=second["job_id"],
    )

    assert first_job.status == IngestJobStatus.SUCCEEDED
    assert second_job.status == IngestJobStatus.SUCCEEDED
    assert first_catalog.get_document_by_path(str(first_path.resolve())).status == (
        DocumentStatus.READY
    )
    assert second_catalog.get_document_by_path(str(second_path.resolve())).status == (
        DocumentStatus.READY
    )
    assert parser.calls == [
        (str(first_path.resolve()), doc_id),
        (str(second_path.resolve()), doc_id),
    ]


def test_search_and_outline_and_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = "docpdf3"
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    search_data = service.search("text-0", [doc_id], 5)
    assert search_data["hits"][0]["doc_id"] == doc_id
    assert search_data["hits"][0]["chunk_id"] == "c0"
    assert search_data["retrieval"]["mode"] == "sqlite_fts5"
    assert search_data["hits"][0]["retrieval_sources"][0] == {
        "source": "sqlite_fts5",
        "rank": 1,
    }

    outline = service.get_outline(doc_id)
    assert outline["title"] == "PDF"
    assert len(outline["nodes"]) == 1
    assert outline["pipeline_status"]["is_stale"] is False

    monkeypatch.setattr(
        "mcp_ebook_read.service.render_pdf_page",
        lambda _path, _out, _page, _dpi: (800, 600),
    )
    render = service.render_pdf_page(doc_id=doc_id, page=1, dpi=150)
    assert render["width"] == 800
    assert render["height"] == 600


def test_stale_ready_document_reingests_instead_of_cached_reuse(
    tmp_path: Path,
) -> None:
    doc_id = "doc-stale-pipeline"
    pdf_path = tmp_path / "stale.pdf"
    pdf_path.write_bytes(b"pdf")
    parser = RecordingParser(result=_parsed(doc_id, title="Fresh", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    catalog = service._catalog_for_document_path(pdf_path)
    catalog.update_document_source_metadata(
        doc_id=doc_id,
        sha256=service._compute_sha256(pdf_path),
        mtime=pdf_path.stat().st_mtime,
    )
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="Old",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="old", title="Old", level=1)],
        overall_confidence=0.8,
        status=DocumentStatus.READY,
    )
    catalog.set_document_status(doc_id, DocumentStatus.READY, profile=Profile.BOOK)

    outline = service.get_outline(doc_id)
    assert outline["pipeline_status"]["is_stale"] is True
    assert outline["pipeline_status"]["reason"] == "missing_pipeline_metadata"

    queued = service.document_ingest(doc_id=doc_id, path=None, force=False)
    assert queued["cached"] is False
    completed = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )
    assert completed["status"] == IngestJobStatus.SUCCEEDED
    assert completed["pipeline_status"]["is_stale"] is False
    assert parser.calls


def test_source_file_change_marks_ready_document_stale_and_reingests(
    tmp_path: Path,
) -> None:
    doc_id = "doc-source-stale"
    pdf_path = tmp_path / "source-stale.pdf"
    pdf_path.write_bytes(b"old-pdf")
    parser = RecordingParser(result=_parsed(doc_id, title="Fresh", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])
    assert len(parser.calls) == 1

    pdf_path.write_bytes(b"new-pdf")
    os.utime(pdf_path, (time.time() + 10, time.time() + 10))

    outline = service.get_outline(doc_id)
    assert outline["pipeline_status"]["is_stale"] is True
    assert outline["pipeline_status"]["reason"] == "source_file_changed"
    assert outline["pipeline_status"]["freshness"] == "needs_reingest"

    reingest = service.document_ingest(doc_id=doc_id, path=None, force=False)
    assert reingest["cached"] is False
    completed = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=reingest["job_id"],
    )
    assert completed["status"] == IngestJobStatus.SUCCEEDED
    assert completed["pipeline_status"]["is_stale"] is False
    assert len(parser.calls) == 2


def test_search_uses_local_fts_without_external_vector_backend(tmp_path: Path) -> None:
    doc_id = "doc-local-search"
    pdf_path = tmp_path / "local-search.pdf"
    pdf_path.write_bytes(b"pdf")
    locator = Locator(
        doc_id=doc_id,
        chunk_id="local-c0",
        section_path=["Symbols"],
        page_range=[7, 8],
        method="docling",
    )
    chunk = ChunkRecord(
        chunk_id="local-c0",
        doc_id=doc_id,
        order_index=0,
        section_path=["Symbols"],
        text="Banach fixedpoint theorem with exactterm42.",
        search_text="banach fixedpoint exactterm42",
        locator=locator,
        method="docling",
    )
    parsed = ParsedDocument(
        title="Local Search",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="symbols", title="Symbols", level=1)],
        chunks=[chunk],
        formulas=[
            _formula(
                doc_id=doc_id,
                formula_id="local-f0",
                chunk_id="local-c0",
                page=7,
                bbox=None,
                latex=r"\alpha + \beta = \gamma",
            )
        ],
        reading_markdown="Symbols",
        overall_confidence=0.9,
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    chunk_search = service.search("exactterm42", [doc_id], 5)
    assert chunk_search["retrieval"]["local_hits_count"] >= 1
    assert chunk_search["hits"][0]["source_type"] == "chunk"
    assert chunk_search["hits"][0]["locator"] == locator.model_dump()
    assert chunk_search["hits"][0]["retrieval_sources"][0] == {
        "source": "sqlite_fts5",
        "rank": 1,
    }

    formula_search = service.search("alpha beta", [doc_id], 5)
    assert any(hit["source_id"] == "local-f0" for hit in formula_search["hits"])


def test_document_explore_and_node_use_document_graph(tmp_path: Path) -> None:
    doc_id = "doc-explore-book"
    pdf_path = tmp_path / "explore.pdf"
    pdf_path.write_bytes(b"pdf")
    locator = Locator(
        doc_id=doc_id,
        chunk_id="explore-c0",
        section_path=["Chapter 1"],
        page_range=[2, 2],
        method="docling",
    )
    chunk = ChunkRecord(
        chunk_id="explore-c0",
        doc_id=doc_id,
        order_index=0,
        section_path=["Chapter 1"],
        text="Persistent homology appears in this chapter.",
        search_text="persistent homology chapter",
        locator=locator,
        method="docling",
    )
    parsed = ParsedDocument(
        title="Explore Book",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="chapter-1", title="Chapter 1", level=1)],
        chunks=[chunk],
        reading_markdown="Persistent homology appears in this chapter.",
        overall_confidence=0.9,
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    explored = service.document_explore(
        doc_id=doc_id,
        query="persistent homology",
        top_k=5,
    )
    assert explored["retrieval"]["mode"] == "sqlite_fts5_plus_document_graph"
    assert explored["graph"]["nodes_count"] >= 3
    assert explored["selected_nodes"][0]["kind"] == "chunk"
    assert explored["selected_nodes"][0]["why_included"]["source"] == "sqlite_fts5"

    node = service.document_node(doc_id=doc_id, node_id="explore-c0")
    assert node["node"]["kind"] == "chunk"
    assert "Persistent homology" in node["read_result"]["content"]
    assert any(item["edge_kind"] == "contains" for item in node["neighbors"])

    sidecars = service.storage_list_sidecars(root=str(tmp_path), limit=20)
    assert sidecars["sidecars"][0]["node_count"] >= 4
    assert sidecars["sidecars"][0]["edge_count"] >= 3
    assert sidecars["sidecars"][0]["documents"][0]["graph"]["page_nodes_count"] == 1


def test_library_explore_searches_root_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "library"
    pdf_dir = root / "pdf-books"
    epub_dir = root / "epub-books"
    pdf_dir.mkdir(parents=True)
    epub_dir.mkdir(parents=True)
    pdf_doc_id = "doc-library-pdf"
    epub_doc_id = "doc-library-epub"
    pdf_path = pdf_dir / "tda.pdf"
    epub_path = epub_dir / "anchor.epub"
    pdf_path.write_bytes(b"pdf")
    epub_path.write_bytes(b"epub")
    pdf_parsed = ParsedDocument(
        title="TDA Book",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="tda", title="TDA", level=1)],
        chunks=[
            ChunkRecord(
                chunk_id="tda-c0",
                doc_id=pdf_doc_id,
                order_index=0,
                section_path=["TDA"],
                text="Persistent homology is the main library-wide match.",
                search_text="persistent homology library match",
                locator=Locator(
                    doc_id=pdf_doc_id,
                    chunk_id="tda-c0",
                    section_path=["TDA"],
                    page_range=[1, 1],
                    method="docling",
                ),
                method="docling",
            )
        ],
        reading_markdown="Persistent homology is the main library-wide match.",
        overall_confidence=0.9,
    )
    epub_parsed = ParsedDocument(
        title="Anchor Book",
        parser_chain=["ebooklib"],
        metadata={},
        outline=[OutlineNode(id="anchor", title="Anchor", level=1)],
        chunks=[
            ChunkRecord(
                chunk_id="anchor-c0",
                doc_id=epub_doc_id,
                order_index=0,
                section_path=["Anchor"],
                text="Anchor programs appear in this EPUB.",
                search_text="anchor programs epub",
                locator=Locator(
                    doc_id=epub_doc_id,
                    chunk_id="anchor-c0",
                    section_path=["Anchor"],
                    method="ebooklib",
                ),
                method="ebooklib",
            )
        ],
        reading_markdown="Anchor programs appear in this EPUB.",
        overall_confidence=0.9,
    )
    service = _build_service(
        root,
        pdf_parser=RecordingParser(result=pdf_parsed),
        epub_parser=RecordingParser(result=epub_parsed),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, pdf_doc_id, pdf_path, DocumentType.PDF, root=root)
    _register_doc(service, epub_doc_id, epub_path, DocumentType.EPUB, root=root)
    pdf_job = service.document_ingest(doc_id=pdf_doc_id, force=True)
    epub_job = service.document_ingest(doc_id=epub_doc_id, force=True)
    _wait_for_ingest_job(service, doc_id=pdf_doc_id, job_id=pdf_job["job_id"])
    _wait_for_ingest_job(service, doc_id=epub_doc_id, job_id=epub_job["job_id"])

    explored = service.library_explore(
        query="persistent homology",
        top_k=5,
    )

    assert explored["root"] == str(root.resolve())
    assert explored["retrieval"]["mode"] == (
        "root_sidecar_sqlite_fts5_plus_document_graph"
    )
    assert explored["retrieval"]["sidecars_count"] == 1
    assert explored["selected_results"][0]["document"]["doc_id"] == pdf_doc_id
    assert explored["selected_results"][0]["node"]["kind"] == "chunk"
    assert {
        call["tool"] for call in explored["selected_results"][0]["suggested_next_calls"]
    } == {"document_explore", "document_node"}
    assert explored["documents"][0]["doc_id"] == pdf_doc_id
    assert explored["documents"][0]["document_score"] > 0
    assert "ranking_signals" in explored["documents"][0]
    assert (
        explored["selected_results"][0]["why_included"]["library_document_score"]
        == explored["documents"][0]["document_score"]
    )


def test_document_explore_and_node_surface_references_and_citations(
    tmp_path: Path,
) -> None:
    doc_id = "doc-paper-refs"
    pdf_path = tmp_path / "paper-refs.pdf"
    pdf_path.write_bytes(b"pdf")
    chunk = ChunkRecord(
        chunk_id="paper-c0",
        doc_id=doc_id,
        order_index=0,
        section_path=["Introduction"],
        text="The paper cites prior graph retrieval work [1].",
        search_text="graph retrieval citation",
        locator=Locator(
            doc_id=doc_id,
            chunk_id="paper-c0",
            section_path=["Introduction"],
            page_range=[1, 1],
            method="docling",
        ),
        method="docling",
    )
    parsed = ParsedDocument(
        title="Paper",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="intro", title="Introduction", level=1)],
        chunks=[chunk],
        reading_markdown="The paper cites prior graph retrieval work [1].",
        overall_confidence=0.9,
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(
            metadata={
                "paper_title": "Reference Paper",
                "references": [
                    {
                        "reference_id": "b1",
                        "title": "Graph Retrieval Systems",
                        "raw_text": "Graph Retrieval Systems. 2026.",
                    }
                ],
                "citations": [
                    {
                        "citation_id": "cite-1",
                        "target": "#b1",
                        "text": "[1]",
                        "chunk_id": "paper-c0",
                        "section_path": ["Introduction"],
                    }
                ],
            }
        ),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF, profile=Profile.PAPER)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    explored = service.document_explore(
        doc_id=doc_id,
        query="reference Graph Retrieval Systems",
        top_k=5,
    )
    assert explored["graph"]["reference_nodes_count"] == 1
    assert explored["graph"]["citation_nodes_count"] == 1
    assert any(node["kind"] == "reference" for node in explored["selected_nodes"])

    reference_node = service.document_node(doc_id=doc_id, node_id="b1")
    assert reference_node["node"]["kind"] == "reference"
    assert "Graph Retrieval Systems" in reference_node["read_result"]["text"]
    citation_node = service.document_node(doc_id=doc_id, node_id="cite-1")
    assert citation_node["node"]["kind"] == "citation"
    assert any(edge["edge_kind"] == "cites" for edge in citation_node["neighbors"])


def test_document_explore_reports_truncation_and_ambiguity(tmp_path: Path) -> None:
    doc_id = "doc-explore-budget"
    pdf_path = tmp_path / "explore-budget.pdf"
    pdf_path.write_bytes(b"pdf")
    chunks: list[ChunkRecord] = []
    for index in range(3):
        chunks.append(
            ChunkRecord(
                chunk_id=f"budget-c{index}",
                doc_id=doc_id,
                order_index=index,
                section_path=["Repeated"],
                text=f"Repeated topic evidence block {index}.",
                search_text="repeated topic evidence",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id=f"budget-c{index}",
                    section_path=["Repeated"],
                    page_range=[index + 1, index + 1],
                    method="docling",
                ),
                method="docling",
            )
        )
    parsed = ParsedDocument(
        title="Budget Book",
        parser_chain=["docling"],
        metadata={},
        outline=[
            OutlineNode(id="repeat-a", title="Repeated", level=1),
            OutlineNode(id="repeat-b", title="Repeated", level=1),
        ],
        chunks=chunks,
        reading_markdown="Repeated topic evidence.",
        overall_confidence=0.9,
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    explored = service.document_explore(
        doc_id=doc_id,
        query="Repeated",
        top_k=2,
    )

    assert explored["truncation"]["selected_nodes_truncated"] is True
    assert explored["truncation"]["next_call"]["args"]["top_k"] == 4
    assert explored["ambiguity_candidates"]
    assert any(
        len(candidate["candidates"]) > 1
        for candidate in explored["ambiguity_candidates"]
    )


def test_document_node_can_read_artifact_node(tmp_path: Path) -> None:
    doc_id = "doc-epub-artifact"
    epub_path = tmp_path / "artifact.epub"
    epub_path.write_bytes(b"epub")
    parsed = _parsed(
        doc_id,
        title="Artifact EPUB",
        method="ebooklib",
        images=[
            ExtractedImage(
                image_id="cover-image",
                doc_id=doc_id,
                order_index=0,
                section_path=["Section"],
                alt="Cover diagram",
                caption="Cover diagram",
                media_type="image/png",
                extension=".png",
                width=320,
                height=180,
                data=b"\x89PNG",
            )
        ],
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=parsed),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, epub_path, DocumentType.EPUB)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])
    _doc, catalog = service._require_doc(
        doc_id,
        error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
        message="missing",
    )
    artifact_node = catalog.list_graph_nodes(doc_id=doc_id, kind="artifact")[0]

    node = service.document_node(doc_id=doc_id, node_id=artifact_node["node_id"])
    assert node["node"]["kind"] == "artifact"
    assert node["read_result"]["media_type"] == "image/png"
    assert Path(node["read_result"]["file_path"]).exists()


def test_epub_ingest_persists_raw_artifacts_as_graph_nodes(tmp_path: Path) -> None:
    doc_id = "doc-epub-raw"
    epub_path = tmp_path / "raw.epub"
    epub_path.write_bytes(b"epub")
    parsed = ParsedDocument(
        title="Raw EPUB",
        parser_chain=["ebooklib"],
        metadata={},
        outline=[OutlineNode(id="section", title="Section", level=1)],
        chunks=[
            ChunkRecord(
                chunk_id="raw-c0",
                doc_id=doc_id,
                order_index=0,
                section_path=["Section"],
                text="Raw artifact chapter.",
                search_text="raw artifact chapter",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="raw-c0",
                    section_path=["Section"],
                    method="ebooklib",
                ),
                method="ebooklib",
            )
        ],
        reading_markdown="Raw artifact chapter.",
        raw_artifacts={"chapter.xhtml": "<html><body>Raw HTML</body></html>"},
        overall_confidence=0.9,
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=parsed),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, epub_path, DocumentType.EPUB)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])
    _doc, catalog = service._require_doc(
        doc_id,
        error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
        message="missing",
    )
    raw_node = next(
        node
        for node in catalog.list_graph_nodes(doc_id=doc_id, kind="artifact")
        if node["metadata"].get("source") == "raw_artifacts"
    )

    read = service.document_node(doc_id=doc_id, node_id=raw_node["node_id"])
    raw_path = Path(read["read_result"]["file_path"])
    assert raw_path.exists()
    assert raw_path.read_text(encoding="utf-8") == "<html><body>Raw HTML</body></html>"
    assert read["read_result"]["media_type"] == "text/plain"


def test_reading_session_capture_export_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id = "doc-capture"
    pdf_path = tmp_path / "capture.pdf"
    pdf_path.write_bytes(b"pdf")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    ingest = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=ingest["job_id"])

    monkeypatch.setenv("MCP_EBOOK_CAPTURE_READING_SESSION", "1")
    monkeypatch.delenv("MCP_EBOOK_CAPTURE_INCLUDE_QUERY", raising=False)
    captured = service.capture_tool_call(
        tool_name="search",
        use_case="search",
        kwargs={"query": "private user question", "doc_ids": [doc_id], "top_k": 5},
        result={"hits": [{"doc_id": doc_id, "chunk_id": "c0"}]},
        latency_ms=12,
    )
    assert captured["eval_capture"]["captured"] is True

    export = service.eval_export_reading_sessions(root=str(tmp_path), limit=10)
    assert export["events_count"] == 1
    event = export["events"][0]
    assert event["doc_id"] == doc_id
    assert event["input"]["query_sha256"]
    assert "query" not in event["input"]
    assert event["returned_ids"]["chunk_ids"] == ["c0"]

    replay = service.eval_replay_reading_sessions(root=str(tmp_path), limit=10)
    assert replay["replayed_count"] == 0
    assert replay["skipped_count"] == 1
    assert replay["skipped"][0]["reason"] == "query_not_captured"

    monkeypatch.setenv("MCP_EBOOK_CAPTURE_INCLUDE_QUERY", "1")
    service.capture_tool_call(
        tool_name="search",
        use_case="search",
        kwargs={"query": "replayable drift query", "doc_ids": [doc_id], "top_k": 5},
        result={"hits": [{"doc_id": doc_id, "chunk_id": "old-hit"}]},
        latency_ms=9,
    )
    document_result = service.document_explore(doc_id=doc_id, query="Section", top_k=3)
    service.capture_tool_call(
        tool_name="document_explore",
        use_case="search",
        kwargs={"doc_id": doc_id, "query": "Section", "top_k": 3},
        result=document_result,
        latency_ms=7,
    )
    library_result = service.library_explore(
        root=str(tmp_path), query="Section", top_k=3
    )
    service.capture_tool_call(
        tool_name="library_explore",
        use_case="search",
        kwargs={"root": str(tmp_path), "query": "Section", "top_k": 3},
        result=library_result,
        latency_ms=8,
    )
    replay_with_query = service.eval_replay_reading_sessions(
        root=str(tmp_path),
        limit=10,
    )
    assert replay_with_query["replayed_count"] == 3
    assert replay_with_query["drifted_count"] == 1
    assert {item["tool_name"] for item in replay_with_query["replayed"]} == {
        "search",
        "document_explore",
        "library_explore",
    }


def test_doctor_health_check_reports_sidecar_findings(tmp_path: Path) -> None:
    doc_id = "doc-doctor"
    pdf_path = tmp_path / "doctor.pdf"
    pdf_path.write_bytes(b"pdf")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    catalog = service._catalog_for_document_path(pdf_path)
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="Doctor",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="n", title="N", level=1)],
        overall_confidence=0.8,
        status=DocumentStatus.READY,
    )
    catalog.set_document_status(doc_id, DocumentStatus.READY, profile=Profile.BOOK)
    catalog.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="missing-image",
                doc_id=doc_id,
                order_index=0,
                section_path=["N"],
                media_type="image/png",
                file_path=str(tmp_path / "missing.png"),
                source="test",
            )
        ],
    )
    orphan_dir = catalog.db_path.parent / "docs" / "orphan-doc"
    orphan_dir.mkdir(parents=True)

    result = service.doctor_health_check(root=str(tmp_path))
    assert result["ok"] is True
    assert {component["name"] for component in result["components"]} >= {
        "local_sqlite_index",
        "grobid_optional",
        "parsers",
        "sidecar_catalogs",
    }
    finding_codes = {finding["code"] for finding in result["findings"]}
    assert "READY_DOCUMENT_HAS_NO_CHUNKS" in finding_codes
    assert "DOCUMENT_PIPELINE_STALE" in finding_codes
    assert "MISSING_IMAGE_ARTIFACT" in finding_codes
    assert "ORPHAN_DOC_ARTIFACT_DIR" in finding_codes


def test_get_outline_and_render_errors(tmp_path: Path) -> None:
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    with pytest.raises(AppError) as outline_exc:
        service.get_outline("missing")
    assert outline_exc.value.code == ErrorCode.INGEST_DOC_NOT_FOUND
    assert "library_scan" in str(outline_exc.value.details["hint"])
    assert outline_exc.value.details["known_roots"] == []

    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")
    _register_doc(service, "epubdoc", epub_path, DocumentType.EPUB)

    with pytest.raises(AppError) as render_exc:
        service.render_pdf_page("epubdoc", page=1, dpi=120)
    assert render_exc.value.code == ErrorCode.RENDER_PAGE_FAILED


def test_read_outline_node_success(tmp_path: Path) -> None:
    doc_id = "doc-outline-1"
    pdf_path = tmp_path / "outline.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    catalog = service._catalog_for_document_path(pdf_path)

    outline = [
        OutlineNode(id="toc-0", title="Chapter A", level=1, page_start=10, page_end=12),
        OutlineNode(id="toc-1", title="Chapter B", level=1, page_start=20, page_end=22),
    ]
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="Outline PDF",
        parser_chain=["docling"],
        metadata={},
        outline=outline,
        overall_confidence=None,
        status=DocumentStatus.READY,
    )

    chunks = [
        ChunkRecord(
            chunk_id="ca-1",
            doc_id=doc_id,
            order_index=0,
            section_path=["Chapter A"],
            text="A-1",
            search_text="A-1",
            locator=Locator(
                doc_id=doc_id,
                chunk_id="ca-1",
                section_path=["Chapter A"],
                page_range=[10, 10],
                method="docling",
            ),
            method="docling",
        ),
        ChunkRecord(
            chunk_id="ca-2",
            doc_id=doc_id,
            order_index=1,
            section_path=["Chapter A"],
            text="A-2",
            search_text="A-2",
            locator=Locator(
                doc_id=doc_id,
                chunk_id="ca-2",
                section_path=["Chapter A"],
                page_range=[11, 11],
                method="docling",
            ),
            method="docling",
        ),
        ChunkRecord(
            chunk_id="cb-1",
            doc_id=doc_id,
            order_index=2,
            section_path=["Chapter B"],
            text="B-1",
            search_text="B-1",
            locator=Locator(
                doc_id=doc_id,
                chunk_id="cb-1",
                section_path=["Chapter B"],
                page_range=[20, 20],
                method="docling",
            ),
            method="docling",
        ),
    ]
    catalog.replace_chunks(doc_id, chunks)

    read = service.read_outline_node(
        doc_id=doc_id,
        node_id="toc-0",
        out_format="text",
        max_chunks=10,
    )
    assert "A-1" in read["content"]
    assert "A-2" in read["content"]
    assert "B-1" not in read["content"]
    assert read["chunks_count"] == 2
    assert read["truncated"] is False


def test_nested_outline_child_node_read_and_search(tmp_path: Path) -> None:
    doc_id = "doc-outline-nested"
    pdf_path = tmp_path / "nested.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    catalog = service._catalog_for_document_path(pdf_path)
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="Nested PDF",
        parser_chain=["docling"],
        metadata={},
        outline=[
            OutlineNode(
                id="toc-parent",
                title="Chapter A",
                level=1,
                page_start=10,
                page_end=14,
                children=[
                    OutlineNode(
                        id="toc-child",
                        title="Part 1",
                        level=2,
                        page_start=11,
                        page_end=12,
                    )
                ],
            )
        ],
        overall_confidence=None,
        status=DocumentStatus.READY,
    )
    catalog.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="inside",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter A", "Part 1"],
                text="Child section text",
                search_text="Child section text",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="inside",
                    section_path=["Chapter A", "Part 1"],
                    page_range=[11, 11],
                    method="docling",
                ),
                method="docling",
            ),
            ChunkRecord(
                chunk_id="sibling",
                doc_id=doc_id,
                order_index=1,
                section_path=["Chapter A", "Part 2"],
                text="Sibling section text",
                search_text="Sibling section text",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="sibling",
                    section_path=["Chapter A", "Part 2"],
                    page_range=[13, 13],
                    method="docling",
                ),
                method="docling",
            ),
        ],
    )

    read = service.read_outline_node(
        doc_id=doc_id,
        node_id="toc-child",
        out_format="text",
        max_chunks=10,
    )
    assert "Child section text" in read["content"]
    assert "Sibling section text" not in read["content"]

    result = service.search_in_outline_node(
        doc_id=doc_id,
        node_id="toc-child",
        query="part",
        top_k=5,
    )
    hit_ids = {hit["chunk_id"] for hit in result["hits"]}
    assert "inside" in hit_ids
    assert "sibling" not in hit_ids


def test_search_in_outline_node_filters_by_page_range(tmp_path: Path) -> None:
    doc_id = "doc-outline-2"
    pdf_path = tmp_path / "outline2.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    catalog = service._catalog_for_document_path(pdf_path)
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="Outline PDF",
        parser_chain=["docling"],
        metadata={},
        outline=[
            OutlineNode(
                id="toc-0",
                title="Chapter A",
                level=1,
                page_start=10,
                page_end=12,
            ),
            OutlineNode(
                id="toc-1",
                title="Chapter B",
                level=1,
                page_start=20,
                page_end=22,
            ),
        ],
        overall_confidence=None,
        status=DocumentStatus.READY,
    )
    catalog.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="inside",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter A", "Part 1"],
                text="Chapter A content",
                search_text="chapter content",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="inside",
                    section_path=["Chapter A", "Part 1"],
                    page_range=[10, 11],
                    method="docling",
                ),
                method="docling",
            ),
            ChunkRecord(
                chunk_id="outside",
                doc_id=doc_id,
                order_index=1,
                section_path=["Chapter B"],
                text="Chapter B content",
                search_text="chapter content",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="outside",
                    section_path=["Chapter B"],
                    page_range=[20, 21],
                    method="docling",
                ),
                method="docling",
            ),
        ],
    )

    result = service.search_in_outline_node(
        doc_id=doc_id,
        node_id="toc-0",
        query="chapter",
        top_k=5,
    )
    hit_ids = {hit["chunk_id"] for hit in result["hits"]}
    assert "inside" in hit_ids
    assert "outside" not in hit_ids


def test_read_outline_node_missing_node(tmp_path: Path) -> None:
    doc_id = "doc-outline-3"
    pdf_path = tmp_path / "outline3.pdf"
    pdf_path.write_bytes(b"pdf")
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    with pytest.raises(AppError) as exc:
        service.read_outline_node(
            doc_id=doc_id,
            node_id="missing-node",
            out_format="text",
        )
    assert exc.value.code == ErrorCode.READ_LOCATOR_NOT_FOUND


def test_document_ingest_epub_persists_images_and_read(tmp_path: Path) -> None:
    doc_id = "doc-epub-image-1"
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    image = ExtractedImage(
        image_id="img-1",
        doc_id=doc_id,
        order_index=0,
        section_path=["Chapter 1"],
        spine_id="chap1",
        href="images/fig1.png",
        anchor="fig1",
        alt="Architecture diagram",
        caption="System overview",
        media_type="image/png",
        extension=".png",
        width=320,
        height=240,
        data=b"\x89PNG\r\n\x1a\n\x00\x00",
    )
    parsed_epub = _parsed(
        doc_id,
        title="Image Book",
        method="ebooklib",
        images=[image],
        outline=[
            OutlineNode(id="toc-1", title="Chapter 1", level=1, spine_ref="chap1")
        ],
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=parsed_epub),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, epub_path, DocumentType.EPUB)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    result = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )["result"]
    assert result["images_count"] == 1

    catalog = service._catalog_for_document_path(epub_path)
    catalog.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="c-epub-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter 1"],
                text="Nearby text for figure",
                search_text="Nearby text for figure",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="c-epub-1",
                    section_path=["Chapter 1"],
                    epub_locator={
                        "spine_id": "chap1",
                        "href": "chap1.xhtml",
                        "anchor": "fig1",
                    },
                    method="ebooklib",
                ),
                method="ebooklib",
            )
        ],
    )

    listed = service.epub_list_images(doc_id=doc_id, node_id=None, limit=50)
    assert listed["images_count"] == 1
    assert listed["images"][0]["image_id"] == "img-1"
    assert Path(listed["images"][0]["image_path"]).exists()
    assert listed["images"][0]["semantic"]["diagnostics"]["ocr_performed"] is False

    node_scoped = service.epub_list_images(doc_id=doc_id, node_id="toc-1", limit=50)
    assert node_scoped["images_count"] == 1

    read = service.epub_read_image(doc_id=doc_id, image_id="img-1")
    assert read["image"]["alt"] == "Architecture diagram"
    assert read["image"]["caption_evidence"]["status"] == "resolved"
    assert read["image"]["caption_evidence"]["source"] == "epub_figcaption"
    assert "Architecture diagram" in read["image"]["semantic"]["summary"]
    assert read["context"] is not None
    assert "Nearby text" in read["context"]["text"]
    assert "Nearby text" in read["image"]["semantic"]["signals"]["context_excerpt"]


def test_epub_node_scoping_uses_spine_and_full_path(tmp_path: Path) -> None:
    doc_id = "doc-epub-scoped"
    epub_path = tmp_path / "scoped.epub"
    epub_path.write_bytes(b"epub")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, epub_path, DocumentType.EPUB)
    catalog = service._catalog_for_document_path(epub_path)
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="Scoped EPUB",
        parser_chain=["ebooklib"],
        metadata={},
        outline=[
            OutlineNode(
                id="toc-ch1",
                title="Chapter One",
                level=1,
                spine_ref="chap1",
                children=[
                    OutlineNode(
                        id="toc-ch1-summary",
                        title="Summary",
                        level=2,
                        spine_ref="chap1",
                    )
                ],
            ),
            OutlineNode(
                id="toc-ch2",
                title="Chapter Two",
                level=1,
                spine_ref="chap2",
                children=[
                    OutlineNode(
                        id="toc-ch2-summary",
                        title="Summary",
                        level=2,
                        spine_ref="chap2",
                    )
                ],
            ),
        ],
        overall_confidence=None,
        status=DocumentStatus.READY,
    )
    catalog.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="ch1-summary",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter One", "Summary"],
                text="Chapter one summary",
                search_text="Chapter one summary",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="ch1-summary",
                    section_path=["Chapter One", "Summary"],
                    epub_locator={"spine_id": "chap1", "href": "chap1.xhtml"},
                    method="ebooklib",
                ),
                method="ebooklib",
            ),
            ChunkRecord(
                chunk_id="ch2-summary",
                doc_id=doc_id,
                order_index=1,
                section_path=["Chapter Two", "Summary"],
                text="Chapter two summary",
                search_text="Chapter two summary",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="ch2-summary",
                    section_path=["Chapter Two", "Summary"],
                    epub_locator={"spine_id": "chap2", "href": "chap2.xhtml"},
                    method="ebooklib",
                ),
                method="ebooklib",
            ),
        ],
    )
    img1 = tmp_path / "epub1.png"
    img2 = tmp_path / "epub2.png"
    img1.write_bytes(b"\x89PNG")
    img2.write_bytes(b"\x89PNG")
    catalog.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="img-ch1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter One", "Summary"],
                spine_id="chap1",
                file_path=str(img1),
            ),
            ImageRecord(
                image_id="img-ch2",
                doc_id=doc_id,
                order_index=1,
                section_path=["Chapter Two", "Summary"],
                spine_id="chap2",
                file_path=str(img2),
            ),
        ],
    )

    read = service.read_outline_node(
        doc_id=doc_id,
        node_id="toc-ch1-summary",
        out_format="text",
        max_chunks=10,
    )
    assert "Chapter one summary" in read["content"]
    assert "Chapter two summary" not in read["content"]

    result = service.search_in_outline_node(
        doc_id=doc_id,
        node_id="toc-ch1-summary",
        query="summary",
        top_k=5,
    )
    hit_ids = {hit["chunk_id"] for hit in result["hits"]}
    assert "ch1-summary" in hit_ids
    assert "ch2-summary" not in hit_ids
    assert "img-ch2" not in hit_ids

    listed = service.epub_list_images(
        doc_id=doc_id,
        node_id="toc-ch1-summary",
        limit=20,
    )
    assert listed["images_count"] == 1
    assert listed["images"][0]["image_id"] == "img-ch1"


def test_epub_image_tools_reject_non_epub(tmp_path: Path) -> None:
    doc_id = "doc-pdf-image-tools"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    with pytest.raises(AppError) as list_exc:
        service.epub_list_images(doc_id=doc_id, node_id=None, limit=20)
    assert list_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE

    with pytest.raises(AppError) as read_exc:
        service.epub_read_image(doc_id=doc_id, image_id="missing")
    assert read_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE


def test_epub_read_image_missing(tmp_path: Path) -> None:
    doc_id = "doc-epub-image-missing"
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, epub_path, DocumentType.EPUB)

    with pytest.raises(AppError) as exc:
        service.epub_read_image(doc_id=doc_id, image_id="unknown")
    assert exc.value.code == ErrorCode.READ_IMAGE_NOT_FOUND


def test_pdf_list_images_and_read_image(tmp_path: Path) -> None:
    doc_id = "doc-pdf-images-1"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    catalog = service._catalog_for_document_path(pdf_path)
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="PDF Image Book",
        parser_chain=["docling"],
        metadata={},
        outline=[
            OutlineNode(
                id="toc-a",
                title="Chapter A",
                level=1,
                page_start=10,
                page_end=12,
            ),
            OutlineNode(
                id="toc-b",
                title="Chapter B",
                level=1,
                page_start=20,
                page_end=22,
            ),
        ],
        overall_confidence=None,
        status=DocumentStatus.READY,
    )
    catalog.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="ca",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter A"],
                text="Figure context A",
                search_text="Figure context A",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="ca",
                    section_path=["Chapter A"],
                    page_range=[10, 10],
                    method="docling",
                ),
                method="docling",
            ),
            ChunkRecord(
                chunk_id="cb",
                doc_id=doc_id,
                order_index=1,
                section_path=["Chapter B"],
                text="Figure context B",
                search_text="Figure context B",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="cb",
                    section_path=["Chapter B"],
                    page_range=[20, 20],
                    method="docling",
                ),
                method="docling",
            ),
        ],
    )

    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    img1.write_bytes(b"\x89PNG")
    img2.write_bytes(b"\x89PNG")
    catalog.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="pdf-img-a",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter A"],
                page=10,
                bbox=[10.0, 10.0, 100.0, 100.0],
                caption="Figure 1: A",
                media_type="image/png",
                file_path=str(img1),
                width=400,
                height=300,
                source="pdf-image-extractor",
            ),
            ImageRecord(
                image_id="pdf-img-b",
                doc_id=doc_id,
                order_index=1,
                section_path=["Chapter B"],
                page=20,
                bbox=[10.0, 10.0, 100.0, 100.0],
                caption="Figure 2: B",
                media_type="image/png",
                file_path=str(img2),
                width=400,
                height=300,
                source="pdf-image-extractor",
            ),
        ],
    )

    listed_all = service.pdf_list_images(doc_id=doc_id, node_id=None, limit=50)
    assert listed_all["images_count"] == 2

    listed_a = service.pdf_list_images(doc_id=doc_id, node_id="toc-a", limit=50)
    assert listed_a["images_count"] == 1
    assert listed_a["images"][0]["image_id"] == "pdf-img-a"
    assert listed_a["images"][0]["page"] == 10

    read = service.pdf_read_image(doc_id=doc_id, image_id="pdf-img-a")
    assert read["image"]["caption"] == "Figure 1: A"
    assert read["image"]["caption_evidence"]["status"] == "resolved"
    assert (
        read["image"]["caption_evidence"]["source"] == "pdf_image_caption_nearby_text"
    )
    assert "Figure 1: A" in read["image"]["semantic"]["summary"]
    assert read["context"] is not None
    assert "context A" in read["context"]["text"]


def test_pdf_images_are_extracted_eagerly_during_ingest(tmp_path: Path) -> None:
    doc_id = "doc-pdf-eager-images"
    pdf_path = tmp_path / "eager.pdf"
    pdf_path.write_bytes(b"pdf")

    extractor = RecordingPdfImageExtractor(
        images=[
            ImageRecord(
                image_id="eager-img-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["S0"],
                page=1,
                bbox=[1.0, 2.0, 3.0, 4.0],
                caption="Figure Eager",
                media_type="image/png",
                file_path="placeholder.png",
                width=200,
                height=100,
                source="pdf-image-extractor",
            )
        ]
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF Eager", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_image_extractor=extractor,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    ingest = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )["result"]
    assert ingest["images_count"] == 1
    assert len(extractor.calls) == 1

    listed = service.pdf_list_images(doc_id=doc_id, node_id=None, limit=20)
    assert listed["images_count"] == 1
    assert listed["images"][0]["image_id"] == "eager-img-1"
    assert Path(listed["images"][0]["image_path"]).exists()
    assert len(extractor.calls) == 1

    read = service.pdf_read_image(doc_id=doc_id, image_id="eager-img-1")
    assert read["image"]["caption"] == "Figure Eager"
    assert read["image"]["caption_evidence"]["status"] == "resolved"
    assert len(extractor.calls) == 1

    Path(read["image"]["image_path"]).unlink()
    with pytest.raises(AppError) as missing_evidence_exc:
        service.pdf_read_image(doc_id=doc_id, image_id="eager-img-1")
    assert missing_evidence_exc.value.code == ErrorCode.READ_IMAGE_NOT_FOUND
    assert "Re-run document_ingest" in missing_evidence_exc.value.details["hint"]
    assert len(extractor.calls) == 1

    catalog = service._catalog_for_document_path(pdf_path)
    workspace_dir = catalog.db_path.parent / "docs" / doc_id
    marker = workspace_dir / "assets" / "pdf-images" / ".extracted.json"
    assert marker.exists()


def test_pdf_image_tools_errors(tmp_path: Path) -> None:
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    epub_doc_id = "doc-epub-for-pdf-tools"
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")
    _register_doc(service, epub_doc_id, epub_path, DocumentType.EPUB)

    with pytest.raises(AppError) as list_exc:
        service.pdf_list_images(doc_id=epub_doc_id, node_id=None, limit=20)
    assert list_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE

    with pytest.raises(AppError) as read_exc:
        service.pdf_read_image(doc_id=epub_doc_id, image_id="x")
    assert read_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE

    pdf_doc_id = "doc-pdf-for-missing-image"
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"pdf")
    _register_doc(service, pdf_doc_id, pdf_path, DocumentType.PDF)
    with pytest.raises(AppError) as missing_exc:
        service.pdf_read_image(doc_id=pdf_doc_id, image_id="missing")
    assert missing_exc.value.code == ErrorCode.READ_IMAGE_NOT_FOUND


def test_pdf_list_tables_and_read_table(tmp_path: Path) -> None:
    doc_id = "doc-pdf-tables-1"
    pdf_path = tmp_path / "tables.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    catalog = service._catalog_for_document_path(pdf_path)
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="PDF Tables",
        parser_chain=["docling"],
        metadata={},
        outline=[
            OutlineNode(
                id="toc-a", title="Chapter A", level=1, page_start=10, page_end=12
            ),
            OutlineNode(
                id="toc-b", title="Chapter B", level=1, page_start=20, page_end=22
            ),
        ],
        overall_confidence=None,
        status=DocumentStatus.READY,
    )
    catalog.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="ca",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter A"],
                text="Table context A",
                search_text="Table context A",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="ca",
                    section_path=["Chapter A"],
                    page_range=[10, 10],
                    method="docling",
                ),
                method="docling",
            ),
            ChunkRecord(
                chunk_id="cb",
                doc_id=doc_id,
                order_index=1,
                section_path=["Chapter B"],
                text="Table context B",
                search_text="Table context B",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="cb",
                    section_path=["Chapter B"],
                    page_range=[20, 20],
                    method="docling",
                ),
                method="docling",
            ),
        ],
    )

    table_a_path = tmp_path / "table-a.png"
    table_b_path = tmp_path / "table-b.png"
    segment_a_path = tmp_path / "table-a-segment.png"
    segment_b_path = tmp_path / "table-b-segment.png"
    for path in (table_a_path, table_b_path, segment_a_path, segment_b_path):
        path.write_bytes(b"\x89PNG")

    table_a = _pdf_table(
        doc_id=doc_id,
        table_id="table-a",
        page_range=[10, 10],
        section_path=["Chapter A"],
        caption="Table 1: A",
    ).model_copy(
        update={
            "file_path": str(table_a_path),
            "segments": [
                TableSegmentRecord(
                    page=10,
                    bbox=[10.0, 10.0, 100.0, 100.0],
                    caption="Table 1: A",
                    file_path=str(segment_a_path),
                    width=400,
                    height=300,
                )
            ],
        }
    )
    table_b = _pdf_table(
        doc_id=doc_id,
        table_id="table-b",
        page_range=[20, 20],
        section_path=["Chapter B"],
        caption="Table 2: B",
    ).model_copy(
        update={
            "order_index": 1,
            "file_path": str(table_b_path),
            "segments": [
                TableSegmentRecord(
                    page=20,
                    bbox=[10.0, 10.0, 100.0, 100.0],
                    caption="Table 2: B",
                    file_path=str(segment_b_path),
                    width=400,
                    height=300,
                )
            ],
        }
    )
    catalog.replace_pdf_tables(doc_id, [table_a, table_b])

    listed_all = service.pdf_list_tables(doc_id=doc_id, node_id=None, limit=50)
    assert listed_all["tables_count"] == 2

    listed_a = service.pdf_list_tables(doc_id=doc_id, node_id="toc-a", limit=50)
    assert listed_a["tables_count"] == 1
    assert listed_a["tables"][0]["table_id"] == "table-a"

    read = service.pdf_read_table(doc_id=doc_id, table_id="table-a")
    assert read["table"]["caption"] == "Table 1: A"
    assert read["table"]["caption_evidence"]["status"] == "resolved"
    assert read["table"]["rows"][0] == ["alpha", "1"]
    assert read["context"] is not None
    assert "context A" in read["context"]["text"]


def test_pdf_visuals_are_extracted_eagerly_during_ingest(tmp_path: Path) -> None:
    doc_id = "doc-pdf-visuals-eager"
    pdf_path = tmp_path / "visuals.pdf"
    pdf_path.write_bytes(b"pdf")

    visual_extractor = RecordingPdfVisualExtractor(
        tables=[
            _pdf_table(
                doc_id=doc_id,
                table_id="eager-table-1",
                page_range=[1, 2],
                section_path=["S0"],
                caption="Table Eager",
                merged=True,
                merge_confidence=0.92,
            )
        ],
        figures=[
            _pdf_figure(
                doc_id=doc_id,
                figure_id="eager-figure-1",
                page=1,
                section_path=["S0"],
                caption="Figure Eager",
                kind="chart",
            )
        ],
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF Eager", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_visual_extractor=visual_extractor,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    ingest = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["job_id"],
    )["result"]
    assert ingest["images_count"] == 0
    assert ingest["pdf_tables_count"] == 1
    assert ingest["pdf_figures_count"] == 1
    assert len(visual_extractor.calls) == 1

    tables = service.pdf_list_tables(doc_id=doc_id, node_id=None, limit=20)
    assert tables["tables_count"] == 1
    assert tables["tables"][0]["table_id"] == "eager-table-1"
    assert tables["tables"][0]["merged"] is True
    assert len(visual_extractor.calls) == 1

    figures = service.pdf_list_figures(doc_id=doc_id, node_id=None, limit=20)
    assert figures["figures_count"] == 1
    assert figures["figures"][0]["figure_id"] == "eager-figure-1"
    assert len(visual_extractor.calls) == 1

    table_read = service.pdf_read_table(doc_id=doc_id, table_id="eager-table-1")
    assert table_read["table"]["segments"]
    assert table_read["table"]["caption_evidence"]["status"] == "resolved"
    assert table_read["context"] is not None

    figure_read = service.pdf_read_figure(doc_id=doc_id, figure_id="eager-figure-1")
    assert figure_read["figure"]["caption"] == "Figure Eager"
    assert figure_read["figure"]["caption_evidence"]["status"] == "resolved"
    assert figure_read["context"] is not None

    Path(table_read["table"]["image_path"]).unlink()
    with pytest.raises(AppError) as missing_table_evidence_exc:
        service.pdf_read_table(doc_id=doc_id, table_id="eager-table-1")
    assert missing_table_evidence_exc.value.code == ErrorCode.READ_TABLE_NOT_FOUND
    assert "Re-run document_ingest" in missing_table_evidence_exc.value.details["hint"]
    assert len(visual_extractor.calls) == 1

    Path(figure_read["figure"]["image_path"]).unlink()
    with pytest.raises(AppError) as missing_figure_evidence_exc:
        service.pdf_read_figure(doc_id=doc_id, figure_id="eager-figure-1")
    assert missing_figure_evidence_exc.value.code == ErrorCode.READ_FIGURE_NOT_FOUND
    assert "Re-run document_ingest" in missing_figure_evidence_exc.value.details["hint"]
    assert len(visual_extractor.calls) == 1

    catalog = service._catalog_for_document_path(pdf_path)
    workspace_dir = catalog.db_path.parent / "docs" / doc_id
    marker = workspace_dir / "assets" / "pdf-visuals" / ".extracted.json"
    assert marker.exists()


def test_pdf_table_and_figure_tools_errors(tmp_path: Path) -> None:
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    epub_doc_id = "doc-epub-for-visual-tools"
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")
    _register_doc(service, epub_doc_id, epub_path, DocumentType.EPUB)

    with pytest.raises(AppError) as list_table_exc:
        service.pdf_list_tables(doc_id=epub_doc_id, node_id=None, limit=20)
    assert list_table_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE

    with pytest.raises(AppError) as read_table_exc:
        service.pdf_read_table(doc_id=epub_doc_id, table_id="x")
    assert read_table_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE

    with pytest.raises(AppError) as list_figure_exc:
        service.pdf_list_figures(doc_id=epub_doc_id, node_id=None, limit=20)
    assert list_figure_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE

    with pytest.raises(AppError) as read_figure_exc:
        service.pdf_read_figure(doc_id=epub_doc_id, figure_id="x")
    assert read_figure_exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE

    pdf_doc_id = "doc-pdf-visual-tools"
    pdf_path = tmp_path / "visuals_missing.pdf"
    pdf_path.write_bytes(b"pdf")
    _register_doc(service, pdf_doc_id, pdf_path, DocumentType.PDF)

    with pytest.raises(AppError) as missing_table_exc:
        service.pdf_read_table(doc_id=pdf_doc_id, table_id="missing")
    assert missing_table_exc.value.code == ErrorCode.READ_TABLE_NOT_FOUND

    with pytest.raises(AppError) as missing_figure_exc:
        service.pdf_read_figure(doc_id=pdf_doc_id, figure_id="missing")
    assert missing_figure_exc.value.code == ErrorCode.READ_FIGURE_NOT_FOUND


def test_pdf_book_formula_tools_list_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = "doc-pdf-formulas-book"
    pdf_path = tmp_path / "formula_book.pdf"
    pdf_path.write_bytes(b"pdf")

    parsed_pdf = _parsed(
        doc_id,
        title="Formula Book",
        method="docling",
        outline=[
            OutlineNode(
                id="toc-1", title="Chapter 1", level=1, page_start=1, page_end=5
            )
        ],
        formulas=[
            _formula(
                doc_id=doc_id,
                formula_id="f-book-1",
                chunk_id="c0",
                page=2,
                bbox=[10.0, 20.0, 120.0, 80.0],
                latex=r"\frac{a}{b}",
                status="resolved",
            )
        ],
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed_pdf),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    listed = service.pdf_list_formulas(
        doc_id=doc_id, node_id=None, limit=20, status=None
    )
    assert listed["formulas_count"] == 1
    assert listed["formulas"][0]["formula_id"] == "f-book-1"

    def fake_render_region(
        _pdf_path: str,
        output_path: Path,
        *,
        page: int,
        bbox: list[float],
    ) -> tuple[int, int]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        assert page == 2
        assert bbox == [10.0, 20.0, 120.0, 80.0]
        return 320, 120

    monkeypatch.setattr(
        "mcp_ebook_read.service.render_pdf_region",
        fake_render_region,
    )
    read = service.pdf_read_formula(doc_id=doc_id, formula_id="f-book-1")
    assert read["formula"]["latex"] == r"\frac{a}{b}"
    assert read["evidence"] is not None
    assert read["evidence"]["type"] == "formula_region"
    assert read["evidence"]["artifact_node_id"]
    assert read["context"] is not None

    artifact_node = service.document_node(
        doc_id=doc_id,
        node_id=read["evidence"]["artifact_node_id"],
    )
    assert artifact_node["node"]["kind"] == "artifact"
    assert artifact_node["read_result"]["media_type"] == "image/png"
    assert Path(artifact_node["read_result"]["file_path"]).exists()
    assert any(
        edge["edge_kind"] == "renders_from" for edge in artifact_node["neighbors"]
    )


def test_pdf_paper_formula_tools_filter_by_node_and_status(tmp_path: Path) -> None:
    doc_id = "doc-pdf-formulas-paper"
    pdf_path = tmp_path / "formula_paper.pdf"
    pdf_path.write_bytes(b"pdf")

    parsed_pdf = _parsed(
        doc_id,
        title="Formula Paper",
        method="docling",
        outline=[
            OutlineNode(
                id="toc-intro", title="Intro", level=1, page_start=1, page_end=2
            ),
            OutlineNode(
                id="toc-method", title="Method", level=1, page_start=3, page_end=6
            ),
        ],
        formulas=[
            _formula(
                doc_id=doc_id,
                formula_id="f-paper-1",
                chunk_id="c0",
                page=1,
                bbox=[10.0, 20.0, 120.0, 80.0],
                latex=r"a=b",
                status="resolved",
            ),
            _formula(
                doc_id=doc_id,
                formula_id="f-paper-2",
                chunk_id="c1",
                page=4,
                bbox=None,
                latex="fallback formula",
                status="fallback_text",
                source="page_text_fallback",
            ),
        ],
    )
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed_pdf),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(metadata={"paper_title": "Formula Paper"}),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF, profile=Profile.PAPER)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    method_only = service.pdf_list_formulas(
        doc_id=doc_id,
        node_id="toc-method",
        limit=20,
        status="fallback_text",
    )
    assert method_only["formulas_count"] == 1
    assert method_only["formulas"][0]["formula_id"] == "f-paper-2"


def test_storage_delete_document_removes_catalog_and_artifacts(
    tmp_path: Path,
) -> None:
    doc_id = "doc-delete-1"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])

    catalog = service._catalog_for_document_path(pdf_path)
    workspace_dir = catalog.db_path.parent / "docs" / doc_id
    assert workspace_dir.exists()

    deleted = service.storage_delete_document(
        doc_id=doc_id,
        path=None,
        remove_artifacts=True,
    )
    assert deleted["deleted_records"] == 1
    assert deleted["artifacts_removed"] is True
    assert catalog.get_document_by_id(doc_id) is None
    assert not workspace_dir.exists()


def test_storage_cleanup_sidecars_removes_missing_docs_and_orphans(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    books_dir = root / "books"
    books_dir.mkdir(parents=True)
    pdf_path = books_dir / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    scan = service.library_scan(str(root), ["**/*.pdf"])
    doc_id = scan["added"][0]["doc_id"]
    catalog = service._catalog_for_document_path(pdf_path)

    orphan_dir = catalog.db_path.parent / "docs" / "orphan-doc"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / "x.txt").write_text("orphan", encoding="utf-8")

    pdf_path.unlink()
    cleaned = service.storage_cleanup_sidecars(
        root=str(root),
        remove_missing_documents=True,
        remove_orphan_artifacts=True,
        compact_catalog=False,
    )
    assert cleaned["removed_deleted_count"] == 1
    assert str(pdf_path.resolve()) in cleaned["removed_paths"]
    assert cleaned["orphan_artifacts_deleted"] >= 1
    assert catalog.get_document_by_id(doc_id) is None


def test_library_ingest_dashboard_preserves_running_job_across_service_open(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    books_dir = root / "books"
    books_dir.mkdir(parents=True)
    pdf_path = books_dir / "running.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    parser = BlockingParser(_parsed("placeholder", title="PDF", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    scan = service.library_scan(str(root), ["**/*.pdf"])
    doc_id = scan["added"][0]["doc_id"]
    parser.result = _parsed(doc_id, title="Running PDF", method="docling")

    queued = service.library_ingest_documents(root=str(root), force=True)

    assert queued["documents_total"] == 1
    assert queued["selected_count"] == 1
    assert queued["queued_count"] == 1
    assert queued["jobs"][0]["relative_path"] == "books/running.pdf"
    assert parser.started.wait(timeout=1.0)

    status = service.library_ingest_status(root=str(root), limit_running=5)
    assert status["progress"]["running_count"] == 1
    assert status["running"][0]["doc_id"] == doc_id
    assert status["running"][0]["owner_id"] == service._ingest_owner_id
    assert status["running"][0]["relative_path"] == "books/running.pdf"

    second_service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed("x", title="Second", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    try:
        second_status = second_service.library_ingest_status(
            root=str(root),
            limit_running=5,
        )
        assert second_status["progress"]["running_count"] == 1
        assert second_status["progress"]["failed_latest_count"] == 0
        assert second_status["running"][0]["owner_id"] == service._ingest_owner_id
    finally:
        second_service.close()

    parser.release.set()
    final = _wait_for_ingest_job(
        service,
        doc_id=doc_id,
        job_id=queued["jobs"][0]["job_id"],
    )
    assert final["status"] == IngestJobStatus.SUCCEEDED
    service.close()


def test_doc_id_tools_require_discovery_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "library"
    books_dir = root / "books"
    books_dir.mkdir(parents=True)
    pdf_path = books_dir / "restart.pdf"
    pdf_path.write_bytes(b"pdf")

    parser = RecordingParser(result=_parsed("x", title="placeholder", method="docling"))
    service = _build_service(
        tmp_path,
        pdf_parser=parser,
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    scan = service.library_scan(str(root), ["**/*.pdf"])
    doc_id = scan["added"][0]["doc_id"]
    parser.result = _parsed(
        doc_id,
        title="Restart PDF",
        method="docling",
        outline=[
            OutlineNode(
                id="toc-restart", title="Chapter 1", level=1, page_start=1, page_end=1
            )
        ],
        formulas=[
            _formula(
                doc_id=doc_id,
                formula_id="f-restart",
                chunk_id="c0",
                page=1,
                bbox=[1.0, 2.0, 3.0, 4.0],
                latex="x=y",
            )
        ],
    )
    queued = service.document_ingest(doc_id=doc_id, path=None, force=True)
    _wait_for_ingest_job(service, doc_id=doc_id, job_id=queued["job_id"])
    catalog = service._catalog_for_document_path(pdf_path)
    chunk = catalog.list_chunks(doc_id)[0]
    image_path = tmp_path / "restart-image.png"
    image_path.write_bytes(b"\x89PNG")
    catalog.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="img-restart",
                doc_id=doc_id,
                order_index=0,
                section_path=["S0"],
                page=1,
                file_path=str(image_path),
            )
        ],
    )

    restarted = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )

    with pytest.raises(AppError) as exc:
        restarted.get_outline(doc_id)
    assert exc.value.code == ErrorCode.INGEST_DOC_NOT_FOUND
    assert exc.value.details["known_roots"] == []
    assert "storage_list_sidecars" in str(exc.value.details["hint"])

    rebound = restarted.storage_list_sidecars(root=str(root), limit=20)
    assert rebound["documents_count"] == 1

    outline = restarted.get_outline(doc_id)
    assert outline["title"] == "Restart PDF"

    read = restarted.read(
        locator=chunk.locator.model_dump(),
        before=0,
        after=0,
        out_format="text",
    )
    assert "text-0" in read["content"]

    formulas = restarted.pdf_list_formulas(
        doc_id=doc_id,
        node_id=None,
        limit=20,
        status=None,
    )
    assert formulas["formulas_count"] == 1

    images = restarted.pdf_list_images(doc_id=doc_id, node_id=None, limit=20)
    assert images["images_count"] == 1

    with pytest.raises(AppError) as missing_exc:
        restarted.get_outline("missing-doc")
    assert missing_exc.value.details["reason"] == (
        "Document id was not found under discovered roots."
    )
