from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    FormulaRecord,
    ImageRecord,
    IngestJobRecord,
    IngestJobStatus,
    IngestStage,
    Locator,
    OutlineNode,
    PdfFigureRecord,
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


def test_catalog_metadata_records_schema_versions(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    metadata = store.catalog_metadata()

    assert metadata["schema_version"] == str(CatalogStore.SCHEMA_VERSION)
    assert metadata["document_graph_schema_version"] == str(
        CatalogStore.DOCUMENT_GRAPH_SCHEMA_VERSION
    )


def test_local_search_boosts_multi_token_cooccurrence(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "doc-cooccur"
    doc_path = (tmp_path / "cooccur.pdf").resolve()
    doc_path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(doc_path),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    store.save_document_parse_output(
        doc_id=doc_id,
        title="Cooccurrence",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="n1", title="Section", level=1)],
        overall_confidence=0.9,
        status="ready",
    )
    chunks = [
        ChunkRecord(
            chunk_id="single-token",
            doc_id=doc_id,
            order_index=0,
            section_path=["Section"],
            text="alpha alpha alpha alpha alpha",
            search_text="alpha alpha alpha alpha alpha",
            locator=Locator(
                doc_id=doc_id,
                chunk_id="single-token",
                section_path=["Section"],
                page_range=[1, 1],
                method="docling",
            ),
            method="docling",
        ),
        ChunkRecord(
            chunk_id="multi-token",
            doc_id=doc_id,
            order_index=1,
            section_path=["Section"],
            text="alpha and beta co-occur in this evidence paragraph.",
            search_text="alpha beta cooccurrence evidence",
            locator=Locator(
                doc_id=doc_id,
                chunk_id="multi-token",
                section_path=["Section"],
                page_range=[1, 1],
                method="docling",
            ),
            method="docling",
        ),
    ]
    store.replace_chunks(doc_id, chunks)

    hits = store.search_local(query="alpha beta", doc_ids=[doc_id], top_k=5)

    assert hits[0]["source_id"] == "multi-token"
    assert hits[0]["why_included"]["token_cooccurrence"]["coverage"] == 1.0


def test_staging_copy_replaces_active_catalog_atomically(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "doc-staging-copy"
    path = (tmp_path / "book.pdf").resolve()
    path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
            title="Before",
        )
    )

    staged = store.create_staging_copy(suffix="unit-test")
    staged.save_document_parse_output(
        doc_id=doc_id,
        title="After",
        parser_chain=["docling"],
        metadata={},
        outline=[],
        overall_confidence=0.9,
        status="ready",
    )

    assert store.get_document_by_id(doc_id).title == "Before"

    store.replace_with_staging_copy(staged)

    loaded = store.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.title == "After"
    assert loaded.status == "ready"
    assert not staged.db_path.exists()


def test_incompatible_catalog_without_metadata_is_replaced(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE legacy_table (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO legacy_table (id) VALUES ('old')")

    store = CatalogStore(db_path)

    assert store.catalog_metadata()["schema_version"] == str(
        CatalogStore.SCHEMA_VERSION
    )
    assert store.list_documents() == []
    backups = list(
        tmp_path.glob("catalog.db.incompatible-missing_catalog_metadata-*.bak")
    )
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute("SELECT id FROM legacy_table").fetchone()[0] == "old"


def test_incompatible_catalog_version_is_replaced(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE catalog_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO catalog_metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", "999"),
                ("document_graph_schema_version", "999"),
            ],
        )

    store = CatalogStore(db_path)

    metadata = store.catalog_metadata()
    assert metadata["schema_version"] == str(CatalogStore.SCHEMA_VERSION)
    assert metadata["document_graph_schema_version"] == str(
        CatalogStore.DOCUMENT_GRAPH_SCHEMA_VERSION
    )
    assert list(tmp_path.glob("catalog.db.incompatible-schema_version_mismatch-*.bak"))


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


def test_catalog_reopen_preserves_active_ingest_jobs(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.db"
    store = CatalogStore(db_path)
    doc_id = "active-doc"
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str((tmp_path / "active.pdf").resolve()),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    store.create_ingest_job(
        IngestJobRecord(
            job_id="job-active",
            doc_id=doc_id,
            path=str((tmp_path / "active.pdf").resolve()),
            profile=Profile.BOOK,
            status=IngestJobStatus.QUEUED,
            stage=IngestStage.QUEUED,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    reopened = CatalogStore(db_path)
    loaded = reopened.get_ingest_job(doc_id, "job-active")

    assert loaded is not None
    assert loaded.status == IngestJobStatus.QUEUED
    assert loaded.stage == IngestStage.QUEUED


def test_ingest_job_claim_and_expired_lease(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "claim-doc"
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str((tmp_path / "claim.pdf").resolve()),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    store.create_ingest_job(
        IngestJobRecord(
            job_id="job-claim",
            doc_id=doc_id,
            path=str((tmp_path / "claim.pdf").resolve()),
            profile=Profile.BOOK,
            status=IngestJobStatus.QUEUED,
            stage=IngestStage.QUEUED,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    claimed = store.claim_ingest_job(
        "job-claim",
        owner_id="owner-a",
        lease_expires_at="2026-01-01T00:10:00+00:00",
        now="2026-01-01T00:00:01+00:00",
    )

    assert claimed is not None
    assert claimed.status == IngestJobStatus.RUNNING
    assert claimed.stage == IngestStage.PARSE
    assert claimed.owner_id == "owner-a"
    assert claimed.claimed_at == "2026-01-01T00:00:01+00:00"
    assert claimed.heartbeat_at == "2026-01-01T00:00:01+00:00"
    assert claimed.lease_expires_at == "2026-01-01T00:10:00+00:00"
    assert (
        store.claim_ingest_job(
            "job-claim",
            owner_id="owner-b",
            lease_expires_at="2026-01-01T00:20:00+00:00",
            now="2026-01-01T00:00:02+00:00",
        )
        is None
    )

    expired = store.fail_expired_running_ingest_jobs(now="2026-01-01T00:11:00+00:00")
    loaded = store.get_ingest_job(doc_id, "job-claim")

    assert expired == 1
    assert loaded is not None
    assert loaded.status == IngestJobStatus.FAILED
    assert loaded.stage == IngestStage.FAILED
    assert loaded.message == "Ingest job lease expired before completion."


def test_document_graph_deduplicates_repeated_stable_refs(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "duplicate-outline-doc"
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str((tmp_path / "duplicate.epub").resolve()),
            type=DocumentType.EPUB,
            sha256="a" * 64,
            mtime=1.0,
        )
    )

    store.save_document_parse_output(
        doc_id=doc_id,
        title="Duplicate Outline",
        parser_chain=["test"],
        metadata={},
        outline=[
            OutlineNode(id="shared-anchor", title="First", level=1),
            OutlineNode(id="shared-anchor", title="Second", level=1),
        ],
        overall_confidence=None,
        status=DocumentStatus.READY,
    )

    nodes = store.list_graph_nodes(doc_id=doc_id, kind="outline_node")

    assert [node["stable_ref"] for node in nodes] == [
        "outline:shared-anchor",
        "outline:shared-anchor#dup-2",
    ]
    assert nodes[1]["metadata"]["original_stable_ref"] == "outline:shared-anchor"
    assert nodes[1]["metadata"]["stable_ref_collision_index"] == 2


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


def test_document_graph_rebuilds_from_persisted_evidence(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "graph-doc"
    doc_path = (tmp_path / "graph.pdf").resolve()
    doc_path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(doc_path),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    store.save_document_parse_output(
        doc_id=doc_id,
        title="Graph PDF",
        parser_chain=["docling"],
        metadata={
            "source": "docling",
            "references": [
                {
                    "reference_id": "b1",
                    "title": "A reference about graph retrieval",
                    "raw_text": "Author. A reference about graph retrieval. 2026.",
                }
            ],
            "citations": [
                {
                    "citation_id": "cite-1",
                    "target": "#b1",
                    "text": "[1]",
                    "chunk_id": "chunk-1",
                    "section_path": ["Chapter 1"],
                }
            ],
        },
        outline=[
            OutlineNode(
                id="chapter-1",
                title="Chapter 1",
                level=1,
                page_start=1,
                page_end=3,
            )
        ],
        overall_confidence=0.9,
        status="ready",
    )
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        doc_id=doc_id,
        order_index=0,
        section_path=["Chapter 1"],
        text="Chunk text with formula and visual evidence.",
        search_text="chunk formula visual",
        locator=Locator(
            doc_id=doc_id,
            chunk_id="chunk-1",
            section_path=["Chapter 1"],
            page_range=[1, 1],
            method="docling",
        ),
        method="docling",
    )
    store.replace_chunks(doc_id, [chunk])
    store.replace_formulas(
        doc_id,
        [
            FormulaRecord(
                formula_id="formula-1",
                doc_id=doc_id,
                chunk_id="chunk-1",
                section_path=["Chapter 1"],
                page=1,
                bbox=[1.0, 2.0, 3.0, 4.0],
                latex="x=y",
                source="pix2text",
            )
        ],
    )
    store.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="image-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter 1"],
                caption="Architecture figure",
                media_type="image/png",
                file_path=str(tmp_path / "image.png"),
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
                section_path=["Chapter 1"],
                page_range=[1, 1],
                bbox=None,
                caption="Metrics table",
                headers=["Metric"],
                rows=[["Value"]],
                markdown="| Metric |\n| --- |\n| Value |",
                html="<table><tr><td>Value</td></tr></table>",
                file_path=str(tmp_path / "table.png"),
                width=200,
                height=100,
                merged=False,
                segments=[],
            )
        ],
    )
    store.replace_pdf_figures(
        doc_id,
        [
            PdfFigureRecord(
                figure_id="figure-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter 1"],
                page=1,
                caption="Signal chart",
                kind="chart",
                file_path=str(tmp_path / "figure.png"),
            )
        ],
    )

    stats = store.graph_stats(doc_id)
    assert stats["nodes_count"] >= 13
    assert stats["edges_count"] >= 16
    assert stats["page_nodes_count"] == 3
    assert stats["artifact_nodes_count"] == 3
    assert stats["reference_nodes_count"] == 1
    assert stats["citation_nodes_count"] == 1
    assert stats["artifacts_count"] == 3
    assert stats["diagnostics_count"] == 0

    nodes = store.list_graph_nodes(doc_id=doc_id)
    kinds = {node["kind"] for node in nodes}
    assert {
        "document",
        "outline_node",
        "page",
        "chunk",
        "formula",
        "image",
        "table",
        "figure",
        "artifact",
        "reference",
        "citation",
    }.issubset(kinds)
    formula_node = next(node for node in nodes if node["kind"] == "formula")
    assert formula_node["text"] == "x=y"
    assert formula_node["locator"]["chunk_id"] == "chunk-1"
    page_node = next(node for node in nodes if node["kind"] == "page")
    assert page_node["locator"]["page"] in {1, 2, 3}
    reference_node = next(node for node in nodes if node["kind"] == "reference")
    assert "graph retrieval" in reference_node["text"]

    edges = store.list_graph_edges(doc_id=doc_id)
    edge_kinds = {edge["kind"] for edge in edges}
    assert {"contains", "mentions", "near", "cites", "renders_from"}.issubset(
        edge_kinds
    )

    summary = store.sidecar_summary()
    assert summary["documents_count"] == 1
    assert summary["artifacts_count"] == 3
    assert summary["nodes_count"] == stats["nodes_count"]
    assert summary["diagnostics_count"] == 0
    diagnostics = store.list_diagnostics(doc_id=doc_id)
    assert diagnostics == []
    reference_hits = store.search_local(
        query="graph retrieval", doc_ids=[doc_id], top_k=5
    )
    reference_hit = next(
        hit for hit in reference_hits if hit["source_type"] == "reference"
    )
    assert reference_hit["retrieval_sources"][0]["source"] == "sqlite_fts5"
    diagnostic_hits = store.search_local(
        query="ARTIFACT_FILE_MISSING", doc_ids=[doc_id], top_k=5
    )
    assert not any(hit["source_type"] == "diagnostic" for hit in diagnostic_hits)

    validation_issues = store.validate_document_graph(doc_id)
    missing_artifact_issues = [
        issue for issue in validation_issues if issue["code"] == "ARTIFACT_FILE_MISSING"
    ]
    assert len(missing_artifact_issues) == 3
    assert {issue["metadata"]["artifact_id"] for issue in missing_artifact_issues} == {
        f"{doc_id}:image:image-1",
        f"{doc_id}:table:table-1",
        f"{doc_id}:figure:figure-1",
    }


def test_validate_document_graph_reports_structural_errors(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "graph-validate-doc"
    path = (tmp_path / "validate.pdf").resolve()
    path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    store.save_document_parse_output(
        doc_id=doc_id,
        title="Validate PDF",
        parser_chain=["docling"],
        metadata={},
        outline=[],
        overall_confidence=0.9,
        status="ready",
    )
    chunk = ChunkRecord(
        chunk_id="chunk-validate-1",
        doc_id=doc_id,
        order_index=0,
        section_path=["Section"],
        text="validate graph chunk",
        search_text="validate graph chunk",
        locator=Locator(
            doc_id=doc_id,
            chunk_id="chunk-validate-1",
            section_path=["Section"],
            method="docling",
        ),
        method="docling",
    )
    store.replace_chunks(doc_id, [chunk])

    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "DELETE FROM graph_nodes WHERE doc_id = ? AND kind = 'chunk'",
            (doc_id,),
        )
        conn.execute(
            "UPDATE chunks SET locator_json = ? WHERE doc_id = ? AND chunk_id = ?",
            (
                json.dumps(
                    {
                        "doc_id": "other-doc",
                        "chunk_id": "other-chunk",
                        "section_path": ["Section"],
                        "method": "docling",
                    }
                ),
                doc_id,
                "chunk-validate-1",
            ),
        )

    issues = store.validate_document_graph(doc_id)

    assert any(
        issue["code"] == "CHUNK_NODE_COUNT_MISMATCH" and issue["severity"] == "error"
        for issue in issues
    )
    assert any(
        issue["code"] == "CHUNK_LOCATOR_MISMATCH" and issue["severity"] == "error"
        for issue in issues
    )


def test_local_search_indexes_outline_images_and_figures(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "visual-search-doc"
    path = (tmp_path / "visual.pdf").resolve()
    path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    store.save_document_parse_output(
        doc_id=doc_id,
        title="Visual Evidence Book",
        parser_chain=["docling"],
        metadata={},
        outline=[OutlineNode(id="figures", title="Figure Gallery", level=1)],
        overall_confidence=0.9,
        status="ready",
    )
    store.replace_images(
        doc_id,
        [
            ImageRecord(
                image_id="image-search-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Figure Gallery"],
                alt="Network topology diagram",
                caption="Network topology diagram",
                media_type="image/png",
                file_path=str(tmp_path / "image.png"),
            )
        ],
    )
    store.replace_pdf_figures(
        doc_id,
        [
            PdfFigureRecord(
                figure_id="figure-search-1",
                doc_id=doc_id,
                order_index=0,
                section_path=["Figure Gallery"],
                page=2,
                caption="Latency chart",
                kind="chart",
                file_path=str(tmp_path / "figure.png"),
            )
        ],
    )

    outline_hits = store.search_local(query="figure gallery", doc_ids=[doc_id], top_k=5)
    assert any(hit["source_type"] == "outline_node" for hit in outline_hits)

    image_hits = store.search_local(query="network topology", doc_ids=[doc_id], top_k=5)
    assert any(hit["source_id"] == "image-search-1" for hit in image_hits)

    figure_hits = store.search_local(query="latency chart", doc_ids=[doc_id], top_k=5)
    assert any(hit["source_id"] == "figure-search-1" for hit in figure_hits)


def test_local_search_supports_fuzzy_title_and_intent_ranking(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "fuzzy-search-doc"
    path = (tmp_path / "fuzzy.pdf").resolve()
    path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    store.save_document_parse_output(
        doc_id=doc_id,
        title="Fuzzy PDF",
        parser_chain=["docling"],
        metadata={},
        outline=[
            OutlineNode(
                id="crypto",
                title="Case Study: TDA in Cryptocurrency Trading",
                level=1,
            )
        ],
        overall_confidence=0.9,
        status="ready",
    )
    store.replace_formulas(
        doc_id,
        [
            FormulaRecord(
                formula_id="eq-alpha-beta",
                doc_id=doc_id,
                section_path=["Case Study: TDA in Cryptocurrency Trading"],
                page=4,
                bbox=None,
                latex=r"\alpha+\beta=\gamma",
                source="pix2text",
            )
        ],
    )
    store.replace_pdf_figures(
        doc_id,
        [
            PdfFigureRecord(
                figure_id="latency-plot",
                doc_id=doc_id,
                order_index=0,
                section_path=["Case Study: TDA in Cryptocurrency Trading"],
                page=5,
                caption="Latency chart",
                kind="chart",
                file_path=str(tmp_path / "latency.png"),
            )
        ],
    )

    fuzzy_hits = store.search_local(
        query="TDA cryptocurrency trade case", doc_ids=[doc_id], top_k=5
    )
    assert fuzzy_hits[0]["source_type"] == "outline_node"
    assert "fuzzy" in fuzzy_hits[0]["why_included"]["reason"]

    formula_hits = store.search_local(
        query="equation alpha beta", doc_ids=[doc_id], top_k=5
    )
    assert formula_hits[0]["source_type"] == "formula"
    assert "query_intent_formula" in formula_hits[0]["why_included"]["reason"]

    figure_hits = store.search_local(
        query="show chart latency", doc_ids=[doc_id], top_k=5
    )
    assert figure_hits[0]["source_type"] == "pdf_figure"
    assert "query_intent_figure" in figure_hits[0]["why_included"]["reason"]


def test_formula_fallback_text_is_not_unresolved_diagnostic(tmp_path: Path) -> None:
    store = CatalogStore(tmp_path / "catalog.db")
    doc_id = "formula-diagnostics-doc"
    path = (tmp_path / "formula.pdf").resolve()
    path.write_bytes(b"pdf")
    store.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(path),
            type=DocumentType.PDF,
            sha256="f" * 64,
            mtime=1.0,
        )
    )
    store.save_document_parse_output(
        doc_id=doc_id,
        title="Formula Diagnostics",
        parser_chain=["docling"],
        metadata={},
        outline=[],
        overall_confidence=0.9,
        status="ready",
    )

    store.replace_formulas(
        doc_id,
        [
            FormulaRecord(
                formula_id="formula-fallback",
                doc_id=doc_id,
                section_path=["S"],
                page=1,
                bbox=[1.0, 2.0, 3.0, 4.0],
                latex="x+y",
                source="docling_formula_text",
                status="fallback_text",
            ),
            FormulaRecord(
                formula_id="formula-unresolved",
                doc_id=doc_id,
                section_path=["S"],
                page=2,
                bbox=[2.0, 3.0, 4.0, 5.0],
                latex="[Formula unresolved. Use render_pdf_page for verification.]",
                source="none",
                status="unresolved",
            ),
        ],
    )

    diagnostics = store.list_diagnostics(doc_id=doc_id)
    unresolved_diagnostics = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic["code"] == "FORMULA_UNRESOLVED"
    ]
    assert len(unresolved_diagnostics) == 1
    assert unresolved_diagnostics[0]["metadata"]["formula_id"] == "formula-unresolved"


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
    assert stats["before_bytes"] > 0
    assert stats["after_bytes"] > 0
    assert stats["reclaimed_bytes"] == max(
        0, stats["before_bytes"] - stats["after_bytes"]
    )


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
