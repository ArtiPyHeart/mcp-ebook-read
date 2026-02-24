from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    ExtractedImage,
    ImageRecord,
    Locator,
    OutlineNode,
    ParsedDocument,
)
from mcp_ebook_read.service import AppService
from mcp_ebook_read.store.catalog import CatalogStore


class RecordingVectorIndex:
    def __init__(self) -> None:
        self.rebuild_calls: list[tuple[str, str, list[ChunkRecord]]] = []
        self.search_calls: list[tuple[str, int, list[str] | None]] = []
        self.search_result: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []

    def rebuild_document(
        self, doc_id: str, title: str, chunks: list[ChunkRecord]
    ) -> None:
        self.rebuild_calls.append((doc_id, title, chunks))

    def search(self, query: str, top_k: int = 20, doc_ids: list[str] | None = None):
        self.search_calls.append((query, top_k, doc_ids))
        return self.search_result

    def delete_document(self, doc_id: str) -> None:
        self.delete_calls.append(doc_id)


class RecordingParser:
    def __init__(
        self, result: ParsedDocument | None = None, error: Exception | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def parse(self, path: str, doc_id: str) -> ParsedDocument:
        self.calls.append((path, doc_id))
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise RuntimeError("parser result not configured")
        return self.result


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
        self.calls: list[str] = []

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


def _parsed(
    doc_id: str,
    *,
    title: str,
    method: str,
    images: list[ExtractedImage] | None = None,
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
        images=images or [],
        reading_markdown="# Section\n\ntext",
        overall_confidence=0.9,
    )


def _build_service(
    tmp_path: Path,
    *,
    pdf_parser: RecordingParser,
    epub_parser: RecordingParser,
    grobid: RecordingGrobid,
    vector: RecordingVectorIndex | None = None,
    pdf_image_extractor: RecordingPdfImageExtractor | None = None,
) -> AppService:
    sidecar_dir = tmp_path / ".mcp-ebook-read"
    return AppService(
        data_dir=sidecar_dir,
        catalog=CatalogStore(sidecar_dir / "catalog.db"),
        sidecar_dir_name=".mcp-ebook-read",
        vector_index=vector or RecordingVectorIndex(),
        pdf_parser=pdf_parser,
        pdf_image_extractor=pdf_image_extractor,
        grobid_client=grobid,
        epub_parser=epub_parser,
    )


def _register_doc(
    service: AppService, doc_id: str, path: Path, doc_type: DocumentType
) -> None:
    catalog = service._catalog_for_document_path(path)
    catalog.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path.resolve()),
            type=doc_type,
            sha256=("a" if doc_type == DocumentType.PDF else "b") * 64,
            mtime=1.0,
        )
    )
    service._bind_doc_catalog(doc_id, catalog)


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
    assert first["storage_maintenance"]["auto_compaction_enabled"] is False

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


def test_library_scan_routes_documents_to_per_folder_sidecars(tmp_path: Path) -> None:
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
    assert book_catalog.db_path.parent == books_dir / ".mcp-ebook-read"
    assert paper_catalog.db_path.parent == papers_dir / ".mcp-ebook-read"
    assert book_catalog.get_document_by_path(str(book_pdf.resolve())) is not None
    assert paper_catalog.get_document_by_path(str(paper_pdf.resolve())) is not None

    sidecars = service.storage_list_sidecars(root=str(root), limit=20)
    assert sidecars["sidecars_count"] == 2
    assert sidecars["documents_count"] == 2


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


def test_document_ingest_pdf_book_success(tmp_path: Path) -> None:
    doc_id = "docpdf1"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    parsed_pdf = _parsed(doc_id, title="PDF Book", method="docling")
    vector = RecordingVectorIndex()
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=parsed_pdf),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        vector=vector,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    result = service.document_ingest_pdf_book(
        doc_id=doc_id,
        path=None,
        force=True,
    )

    assert result["doc_id"] == doc_id
    assert result["chunks_count"] == 2
    assert result["images_count"] == 0
    assert result["parser_chain"] == ["docling"]
    assert len(vector.rebuild_calls) == 1

    reading = tmp_path / ".mcp-ebook-read" / "docs" / doc_id / "reading" / "reading.md"
    assert reading.exists()

    loaded_catalog = service._catalog_for_document_path(pdf_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.status == DocumentStatus.READY
    assert loaded.title == "PDF Book"


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
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    service.document_ingest_pdf_book(doc_id=doc_id, path=None, force=True)
    second = service.document_ingest_pdf_book(doc_id=doc_id, path=None, force=False)

    assert second["doc_id"] == doc_id
    assert second["images_count"] == 0
    assert len(parser.calls) == 1


def test_document_ingest_pdf_paper_merges_grobid(tmp_path: Path) -> None:
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
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    result = service.document_ingest_pdf_paper(
        doc_id=doc_id,
        path=None,
        force=True,
    )

    assert result["parser_chain"] == ["docling", "grobid"]
    assert result["images_count"] == 0
    assert grobid.calls == [str(pdf_path.resolve())]

    loaded_catalog = service._catalog_for_document_path(pdf_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.metadata["paper_title"] == "Paper Title"
    assert [node.title for node in loaded.outline] == ["Grobid Intro"]


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

    with pytest.raises(AppError) as exc:
        service.document_ingest_epub_book(doc_id=doc_id, path=None, force=True)

    assert exc.value.code == ErrorCode.INGEST_EPUB_PARSE_FAILED
    loaded_catalog = service._catalog_for_document_path(epub_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.status == DocumentStatus.FAILED


def test_document_ingest_pdf_book_rejects_epub(tmp_path: Path) -> None:
    doc_id = "docepub2"
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
        service.document_ingest_pdf_book(doc_id=doc_id, path=None, force=True)

    assert exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE
    loaded_catalog = service._catalog_for_document_path(epub_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.status == DocumentStatus.FAILED


def test_document_ingest_epub_book_rejects_pdf(tmp_path: Path) -> None:
    doc_id = "docpdf4"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    with pytest.raises(AppError) as exc:
        service.document_ingest_epub_book(doc_id=doc_id, path=None, force=True)

    assert exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE
    loaded_catalog = service._catalog_for_document_path(pdf_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.status == DocumentStatus.FAILED


def test_document_ingest_pdf_paper_rejects_epub(tmp_path: Path) -> None:
    doc_id = "docepub2"
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
        service.document_ingest_pdf_paper(doc_id=doc_id, path=None, force=True)

    assert exc.value.code == ErrorCode.INGEST_UNSUPPORTED_TYPE
    loaded_catalog = service._catalog_for_document_path(epub_path)
    loaded = loaded_catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.status == DocumentStatus.FAILED


def test_search_and_outline_and_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_id = "docpdf3"
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"pdf")

    vector = RecordingVectorIndex()
    vector.search_result = [{"doc_id": doc_id, "chunk_id": "c0"}]
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        vector=vector,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    service.document_ingest_pdf_book(doc_id=doc_id, path=None, force=True)

    search_data = service.search("query", [doc_id], 5)
    assert search_data["hits"] == [{"doc_id": doc_id, "chunk_id": "c0"}]
    assert vector.search_calls == [("query", 5, [doc_id])]

    outline = service.get_outline(doc_id)
    assert outline["title"] == "PDF"
    assert len(outline["nodes"]) == 1

    monkeypatch.setattr(
        "mcp_ebook_read.service.render_pdf_page",
        lambda _path, _out, _page, _dpi: (800, 600),
    )
    render = service.render_pdf_page(doc_id=doc_id, page=1, dpi=150)
    assert render["width"] == 800
    assert render["height"] == 600


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


def test_search_in_outline_node_filters_by_page_range(tmp_path: Path) -> None:
    doc_id = "doc-outline-2"
    pdf_path = tmp_path / "outline2.pdf"
    pdf_path.write_bytes(b"pdf")

    vector = RecordingVectorIndex()
    vector.search_result = [
        {
            "doc_id": doc_id,
            "chunk_id": "inside",
            "locator": {
                "doc_id": doc_id,
                "chunk_id": "inside",
                "section_path": ["Chapter A", "Part 1"],
                "page_range": [10, 11],
                "method": "docling",
            },
        },
        {
            "doc_id": doc_id,
            "chunk_id": "outside",
            "locator": {
                "doc_id": doc_id,
                "chunk_id": "outside",
                "section_path": ["Chapter B"],
                "page_range": [20, 21],
                "method": "docling",
            },
        },
    ]
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        vector=vector,
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

    result = service.search_in_outline_node(
        doc_id=doc_id,
        node_id="toc-0",
        query="chapter",
        top_k=5,
    )
    assert [hit["chunk_id"] for hit in result["hits"]] == ["inside"]


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
    result = service.document_ingest_epub_book(doc_id=doc_id, path=None, force=True)
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

    node_scoped = service.epub_list_images(doc_id=doc_id, node_id="toc-1", limit=50)
    assert node_scoped["images_count"] == 1

    read = service.epub_read_image(doc_id=doc_id, image_id="img-1")
    assert read["image"]["alt"] == "Architecture diagram"
    assert read["context"] is not None
    assert "Nearby text" in read["context"]["text"]


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
    assert read["context"] is not None
    assert "context A" in read["context"]["text"]


def test_pdf_images_are_extracted_on_demand(tmp_path: Path) -> None:
    doc_id = "doc-pdf-lazy-images"
    pdf_path = tmp_path / "lazy.pdf"
    pdf_path.write_bytes(b"pdf")

    extractor = RecordingPdfImageExtractor(
        images=[
            ImageRecord(
                image_id="lazy-img-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["S0"],
                page=1,
                bbox=[1.0, 2.0, 3.0, 4.0],
                caption="Figure Lazy",
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
            result=_parsed(doc_id, title="PDF Lazy", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        pdf_image_extractor=extractor,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)

    ingest = service.document_ingest_pdf_book(doc_id=doc_id, path=None, force=True)
    assert ingest["images_count"] == 0
    assert extractor.calls == []

    listed = service.pdf_list_images(doc_id=doc_id, node_id=None, limit=20)
    assert listed["images_count"] == 1
    assert listed["images"][0]["image_id"] == "lazy-img-1"
    assert Path(listed["images"][0]["image_path"]).exists()
    assert len(extractor.calls) == 1

    read = service.pdf_read_image(doc_id=doc_id, image_id="lazy-img-1")
    assert read["image"]["caption"] == "Figure Lazy"
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


def test_storage_delete_document_removes_catalog_vector_and_artifacts(
    tmp_path: Path,
) -> None:
    doc_id = "doc-delete-1"
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    vector = RecordingVectorIndex()
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(
            result=_parsed(doc_id, title="PDF", method="docling")
        ),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        vector=vector,
    )
    _register_doc(service, doc_id, pdf_path, DocumentType.PDF)
    service.document_ingest_pdf_book(doc_id=doc_id, path=None, force=True)

    catalog = service._catalog_for_document_path(pdf_path)
    workspace_dir = catalog.db_path.parent / "docs" / doc_id
    assert workspace_dir.exists()

    deleted = service.storage_delete_document(
        doc_id=doc_id,
        path=None,
        remove_artifacts=True,
    )
    assert deleted["deleted_records"] == 1
    assert deleted["vector_deleted"] is True
    assert deleted["artifacts_removed"] is True
    assert catalog.get_document_by_id(doc_id) is None
    assert doc_id in vector.delete_calls
    assert not workspace_dir.exists()


def test_storage_cleanup_sidecars_removes_missing_docs_and_orphans(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    books_dir = root / "books"
    books_dir.mkdir(parents=True)
    pdf_path = books_dir / "book.pdf"
    pdf_path.write_bytes(b"pdf")

    vector = RecordingVectorIndex()
    service = _build_service(
        tmp_path,
        pdf_parser=RecordingParser(result=_parsed("x", title="X", method="docling")),
        epub_parser=RecordingParser(result=_parsed("x", title="X", method="ebooklib")),
        grobid=RecordingGrobid(),
        vector=vector,
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
    assert doc_id in vector.delete_calls
    assert catalog.get_document_by_id(doc_id) is None
