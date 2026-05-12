from __future__ import annotations

from pathlib import Path

from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentType,
    FormulaRecord,
    ImageRecord,
    IngestJobRecord,
    IngestJobStatus,
    IngestStage,
    Locator,
    PdfTableRecord,
    Profile,
    TableSegmentRecord,
)
from mcp_ebook_read.store.catalog import CatalogStore


def test_upsert_and_get_document(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc = DocumentRecord(
        doc_id="abcd1234abcd1234",
        path=str((tmp_path / "book.pdf").resolve()),
        type=DocumentType.PDF,
        sha256="a" * 64,
        mtime=1.0,
    )

    state = store.upsert_scanned_document(doc)
    assert state == "added"

    loaded = store.get_document_by_id(doc.doc_id)
    assert loaded is not None
    assert loaded.path == doc.path


def test_ingest_job_progress_round_trips_and_updates(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "progress-doc"
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str((tmp_path / "progress.pdf").resolve()),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )

    queued_progress = {
        "stage": "queued",
        "done": 0,
        "total": 6,
        "pct": 0.0,
        "current_item": "queued",
        "diagnostics": {},
    }
    job = IngestJobRecord(
        job_id="job-progress",
        doc_id=doc_id,
        path=str((tmp_path / "progress.pdf").resolve()),
        profile=Profile.BOOK,
        status=IngestJobStatus.QUEUED,
        stage=IngestStage.QUEUED,
        message="queued",
        progress=queued_progress,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    store.create_ingest_job(job)
    loaded = store.get_ingest_job(doc_id, "job-progress")
    assert loaded is not None
    assert loaded.progress == queued_progress

    index_progress = {
        "stage": "index",
        "done": 4,
        "total": 6,
        "pct": 66.7,
        "current_item": "indexing chunks",
        "diagnostics": {"chunks_count": 12},
    }
    store.update_ingest_job(
        "job-progress",
        status=IngestJobStatus.RUNNING,
        stage=IngestStage.INDEX,
        message="indexing chunks",
        progress=index_progress,
    )
    updated = store.get_ingest_job(doc_id, "job-progress")
    assert updated is not None
    assert updated.status == IngestJobStatus.RUNNING
    assert updated.stage == IngestStage.INDEX
    assert updated.progress == index_progress


def test_replace_chunks_and_window(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "abcd1234abcd1234"
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str((tmp_path / "book.epub").resolve()),
            type=DocumentType.EPUB,
            sha256="b" * 64,
            mtime=2.0,
        )
    )

    chunks = []
    for i in range(3):
        locator = Locator(
            doc_id=doc_id,
            chunk_id=f"c{i}",
            section_path=[f"S{i}"],
            method="ebooklib",
        )
        chunks.append(
            ChunkRecord(
                chunk_id=f"c{i}",
                doc_id=doc_id,
                order_index=i,
                section_path=[f"S{i}"],
                text=f"text-{i}",
                search_text=f"text-{i}",
                locator=locator,
                method="ebooklib",
            )
        )

    store.replace_chunks(doc_id, chunks)
    window = store.get_chunks_window(doc_id, center_order=1, before=1, after=1)
    assert [c.chunk_id for c in window] == ["c0", "c1", "c2"]
    listed = store.list_chunks(doc_id)
    assert [c.chunk_id for c in listed] == ["c0", "c1", "c2"]


def test_local_search_indexes_chunks_formulas_and_tables(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "local-search-doc"
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str((tmp_path / "search.pdf").resolve()),
            type=DocumentType.PDF,
            sha256="c" * 64,
            mtime=1.0,
        )
    )
    locator = Locator(
        doc_id=doc_id,
        chunk_id="chunk-1",
        section_path=["Methods"],
        page_range=[3, 4],
        method="docling",
    )
    store.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="chunk-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Methods"],
                text="Persistent homology exact keyword lives here.",
                search_text="persistent homology exactkeyword",
                locator=locator,
                method="docling",
            )
        ],
    )
    store.replace_formulas(
        doc_id,
        [
            FormulaRecord(
                formula_id="formula-1",
                doc_id=doc_id,
                chunk_id="chunk-1",
                section_path=["Methods"],
                page=3,
                bbox=None,
                latex=r"\alpha + \beta = \gamma",
                source="pix2text",
                status="resolved",
            )
        ],
    )
    store.replace_pdf_tables(
        doc_id,
        [
            PdfTableRecord(
                table_id="table-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Results"],
                page_range=[5, 6],
                bbox=None,
                caption="Cryptocurrency Sharpe table",
                headers=["Asset", "Sharpe"],
                rows=[["SOL", "1.7"]],
                markdown="| Asset | Sharpe |\n| --- | --- |\n| SOL | 1.7 |",
                html="<table><tr><td>SOL</td><td>1.7</td></tr></table>",
                file_path="table.png",
                width=400,
                height=300,
                merged=False,
                merge_confidence=None,
                segments=[
                    TableSegmentRecord(
                        page=5,
                        bbox=[1.0, 2.0, 3.0, 4.0],
                        caption="Cryptocurrency Sharpe table",
                        file_path="table.png",
                        width=400,
                        height=300,
                    )
                ],
            )
        ],
    )

    chunk_hits = store.search_local(query="exactkeyword", doc_ids=[doc_id], top_k=5)
    assert chunk_hits[0]["source_type"] == "chunk"
    assert chunk_hits[0]["locator"] == locator.model_dump()

    formula_hits = store.search_local(query="alpha beta", doc_ids=[doc_id], top_k=5)
    assert any(hit["source_id"] == "formula-1" for hit in formula_hits)

    table_hits = store.search_local(query="sharpe sol", doc_ids=[doc_id], top_k=5)
    assert any(hit["source_id"] == "table-1" for hit in table_hits)


def test_delete_documents_by_paths_cascades_related_rows(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "doc-delete-1"
    path = (tmp_path / "to-delete.pdf").resolve()
    path.write_bytes(b"pdf")

    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.PDF,
            sha256="d" * 64,
            mtime=1.0,
        )
    )
    chunk = ChunkRecord(
        chunk_id="chunk-delete-1",
        doc_id=doc_id,
        order_index=0,
        section_path=["S"],
        text="text",
        search_text="text",
        locator=Locator(
            doc_id=doc_id,
            chunk_id="chunk-delete-1",
            section_path=["S"],
            method="docling",
        ),
        method="docling",
    )
    store.replace_chunks(doc_id, [chunk])
    store.replace_formulas(
        doc_id,
        [
            FormulaRecord(
                formula_id="formula-delete-1",
                doc_id=doc_id,
                chunk_id="chunk-delete-1",
                section_path=["S"],
                page=1,
                bbox=[1.0, 2.0, 3.0, 4.0],
                latex="x=y",
                source="pix2text",
            )
        ],
    )
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"\x89PNG")
    store.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="image-delete-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["S"],
                spine_id="chap1",
                href="images/img.png",
                media_type="image/png",
                file_path=str(image_path),
            )
        ],
    )

    deleted = store.delete_documents_by_paths([str(path)])
    assert deleted == 1
    assert store.get_document_by_id(doc_id) is None
    assert store.list_chunks(doc_id) == []
    assert store.list_formulas(doc_id) == []
    assert store.list_images(doc_id) == []


def test_compact_runs_vacuum(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "doc-vacuum-1"
    path = (tmp_path / "vacuum.pdf").resolve()
    path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.PDF,
            sha256="e" * 64,
            mtime=1.0,
        )
    )
    chunk = ChunkRecord(
        chunk_id="chunk-vacuum-1",
        doc_id=doc_id,
        order_index=0,
        section_path=["S"],
        text="x" * 20000,
        search_text="x" * 20000,
        locator=Locator(
            doc_id=doc_id,
            chunk_id="chunk-vacuum-1",
            section_path=["S"],
            method="docling",
        ),
        method="docling",
    )
    store.replace_chunks(doc_id, [chunk])

    stats = store.compact()
    assert stats["before_bytes"] >= stats["after_bytes"]


def test_replace_and_get_images(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "doc-image-1"
    path = (tmp_path / "book.epub").resolve()
    path.write_bytes(b"epub")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.EPUB,
            sha256="f" * 64,
            mtime=1.0,
        )
    )
    img = tmp_path / "img1.png"
    img.write_bytes(b"\x89PNG")
    store.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="img-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter 1"],
                spine_id="chap1",
                href="images/img1.png",
                anchor="fig-1",
                alt="diagram",
                caption="system diagram",
                media_type="image/png",
                file_path=str(img),
                width=200,
                height=100,
            )
        ],
    )

    listed = store.list_images(doc_id)
    assert len(listed) == 1
    assert listed[0].image_id == "img-1"
    assert listed[0].anchor == "fig-1"
    loaded = store.get_image("img-1")
    assert loaded is not None
    assert loaded.file_path == str(img)
