"""SQLite persistence for documents and chunks."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    FormulaRecord,
    IngestJobRecord,
    IngestJobStatus,
    IngestStage,
    ImageRecord,
    OutlineNode,
    PdfFigureRecord,
    PdfTableRecord,
    Profile,
)


class CatalogStore:
    """Catalog and chunk persistence backed by sqlite."""

    _FTS_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    parser_chain_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    outline_json TEXT NOT NULL,
                    overall_confidence REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    section_path_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    locator_json TEXT NOT NULL,
                    method TEXT NOT NULL,
                    confidence REAL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_order
                ON chunks(doc_id, order_index);

                CREATE TABLE IF NOT EXISTS formulas (
                    formula_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT,
                    section_path_json TEXT NOT NULL,
                    page INTEGER,
                    bbox_json TEXT,
                    latex TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_formulas_doc
                ON formulas(doc_id);

                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    section_path_json TEXT NOT NULL,
                    spine_id TEXT,
                    page INTEGER,
                    bbox_json TEXT,
                    href TEXT,
                    anchor TEXT,
                    alt TEXT,
                    caption TEXT,
                    media_type TEXT,
                    file_path TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_images_doc
                ON images(doc_id, order_index);

                CREATE TABLE IF NOT EXISTS pdf_tables (
                    table_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    section_path_json TEXT NOT NULL,
                    page_range_json TEXT,
                    bbox_json TEXT,
                    caption TEXT,
                    headers_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    html TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    merged INTEGER NOT NULL,
                    merge_confidence REAL,
                    segments_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_pdf_tables_doc
                ON pdf_tables(doc_id, order_index);

                CREATE TABLE IF NOT EXISTS pdf_figures (
                    figure_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    section_path_json TEXT NOT NULL,
                    page INTEGER,
                    bbox_json TEXT,
                    caption TEXT,
                    kind TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_pdf_figures_doc
                ON pdf_figures(doc_id, order_index);

                CREATE VIRTUAL TABLE IF NOT EXISTS local_search_fts USING fts5(
                    doc_id UNINDEXED,
                    source_type UNINDEXED,
                    source_id UNINDEXED,
                    chunk_id UNINDEXED,
                    section_path_json UNINDEXED,
                    order_index UNINDEXED,
                    text,
                    tokenize='unicode61'
                );

                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    force INTEGER NOT NULL,
                    message TEXT,
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ingest_jobs_doc_created
                ON ingest_jobs(doc_id, created_at DESC);
                """
            )
            image_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(images)").fetchall()
            }
            if "page" not in image_columns:
                conn.execute("ALTER TABLE images ADD COLUMN page INTEGER")
            if "bbox_json" not in image_columns:
                conn.execute("ALTER TABLE images ADD COLUMN bbox_json TEXT")
            ingest_job_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(ingest_jobs)").fetchall()
            }
            if "progress_json" not in ingest_job_columns:
                conn.execute(
                    "ALTER TABLE ingest_jobs "
                    "ADD COLUMN progress_json TEXT NOT NULL DEFAULT '{}'"
                )
            conn.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, stage = ?, message = ?, updated_at = ?, finished_at = ?
                WHERE status IN (?, ?)
                """,
                (
                    IngestJobStatus.FAILED,
                    IngestStage.FAILED,
                    "Job was interrupted by a previous server shutdown or restart.",
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                    IngestJobStatus.QUEUED,
                    IngestJobStatus.RUNNING,
                ),
            )

    def upsert_scanned_document(self, doc: DocumentRecord) -> str:
        """Upsert discovered document and return added|updated|unchanged."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE path = ?",
                (doc.path,),
            ).fetchone()

            now = datetime.now(UTC).isoformat()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, path, type, sha256, mtime, title, status, profile,
                        parser_chain_json, metadata_json, outline_json,
                        overall_confidence, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc.doc_id,
                        doc.path,
                        doc.type,
                        doc.sha256,
                        doc.mtime,
                        doc.title,
                        doc.status,
                        doc.profile,
                        json.dumps(doc.parser_chain),
                        json.dumps(doc.metadata),
                        json.dumps([node.model_dump() for node in doc.outline]),
                        doc.overall_confidence,
                        now,
                        now,
                    ),
                )
                return "added"

            changed = row["sha256"] != doc.sha256 or float(row["mtime"]) != float(
                doc.mtime
            )
            if not changed:
                return "unchanged"

            old_doc_id = row["doc_id"]
            if old_doc_id != doc.doc_id:
                conn.execute(
                    "DELETE FROM local_search_fts WHERE doc_id = ?",
                    (old_doc_id,),
                )
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (old_doc_id,))
                conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, path, type, sha256, mtime, title, status, profile,
                        parser_chain_json, metadata_json, outline_json,
                        overall_confidence, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc.doc_id,
                        doc.path,
                        doc.type,
                        doc.sha256,
                        doc.mtime,
                        doc.title,
                        DocumentStatus.DISCOVERED,
                        doc.profile,
                        json.dumps([]),
                        json.dumps({}),
                        json.dumps([]),
                        None,
                        now,
                        now,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE documents
                    SET sha256 = ?, mtime = ?, status = ?, updated_at = ?
                    WHERE path = ?
                    """,
                    (doc.sha256, doc.mtime, DocumentStatus.DISCOVERED, now, doc.path),
                )
            return "updated"

    def list_document_paths_under_root(self, root: str) -> list[str]:
        root_path = str(Path(root).resolve())
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT path FROM documents WHERE path LIKE ?",
                (f"{root_path}%",),
            ).fetchall()
            return [str(row["path"]) for row in rows]

    def delete_documents_by_paths(self, paths: list[str]) -> int:
        if not paths:
            return 0
        normalized = sorted({str(Path(path).resolve()) for path in paths})
        with self._conn() as conn:
            placeholders = ",".join("?" for _ in normalized)
            rows = conn.execute(
                f"SELECT doc_id FROM documents WHERE path IN ({placeholders})",
                normalized,
            ).fetchall()
            doc_ids = [row["doc_id"] for row in rows]
            if doc_ids:
                doc_placeholders = ",".join("?" for _ in doc_ids)
                conn.execute(
                    "DELETE FROM local_search_fts "
                    f"WHERE doc_id IN ({doc_placeholders})",
                    doc_ids,
                )
            conn.executemany(
                "DELETE FROM documents WHERE path = ?",
                [(path,) for path in normalized],
            )
            return len(doc_ids)

    def db_size_bytes(self) -> int:
        if not self.db_path.exists():
            return 0
        return self.db_path.stat().st_size

    def compact(self) -> dict[str, int]:
        """Compact sqlite storage and report reclaimed bytes."""
        before = self.db_size_bytes()
        with self._conn() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
        after = self.db_size_bytes()
        reclaimed = max(0, before - after)
        return {
            "before_bytes": before,
            "after_bytes": after,
            "reclaimed_bytes": reclaimed,
        }

    def get_document_by_id(self, doc_id: str) -> DocumentRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            return self._row_to_document(row) if row else None

    def get_document_by_path(self, path: str) -> DocumentRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE path = ?",
                (str(Path(path).resolve()),),
            ).fetchone()
            return self._row_to_document(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM documents
                ORDER BY path ASC
                """
            ).fetchall()
            return [self._row_to_document(row) for row in rows]

    def create_ingest_job(self, job: IngestJobRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ingest_jobs (
                    job_id, doc_id, path, profile, status, stage, force, message,
                    progress_json, result_json, error_json, created_at, updated_at,
                    started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.doc_id,
                    job.path,
                    job.profile,
                    job.status,
                    job.stage,
                    1 if job.force else 0,
                    job.message,
                    json.dumps(job.progress),
                    json.dumps(job.result) if job.result is not None else None,
                    json.dumps(job.error) if job.error is not None else None,
                    job.created_at,
                    job.updated_at,
                    job.started_at,
                    job.finished_at,
                ),
            )

    def get_ingest_job(
        self, doc_id: str, job_id: str | None = None
    ) -> IngestJobRecord | None:
        with self._conn() as conn:
            if job_id:
                row = conn.execute(
                    """
                    SELECT * FROM ingest_jobs
                    WHERE doc_id = ? AND job_id = ?
                    """,
                    (doc_id, job_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT * FROM ingest_jobs
                    WHERE doc_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (doc_id,),
                ).fetchone()
            return self._row_to_ingest_job(row) if row else None

    def list_ingest_jobs(self, doc_id: str, limit: int = 20) -> list[IngestJobRecord]:
        capped_limit = max(1, min(limit, 200))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ingest_jobs
                WHERE doc_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (doc_id, capped_limit),
            ).fetchall()
            return [self._row_to_ingest_job(row) for row in rows]

    def get_active_ingest_job(self, doc_id: str) -> IngestJobRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM ingest_jobs
                WHERE doc_id = ? AND status IN (?, ?)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    doc_id,
                    IngestJobStatus.QUEUED,
                    IngestJobStatus.RUNNING,
                ),
            ).fetchone()
            return self._row_to_ingest_job(row) if row else None

    def update_ingest_job(
        self,
        job_id: str,
        *,
        status: IngestJobStatus | None = None,
        stage: IngestStage | None = None,
        message: str | None = None,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        assignments: list[str] = ["updated_at = ?"]
        values: list[Any] = [datetime.now(UTC).isoformat()]
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if stage is not None:
            assignments.append("stage = ?")
            values.append(stage)
        if message is not None:
            assignments.append("message = ?")
            values.append(message)
        if progress is not None:
            assignments.append("progress_json = ?")
            values.append(json.dumps(progress))
        if result is not None:
            assignments.append("result_json = ?")
            values.append(json.dumps(result))
        if error is not None:
            assignments.append("error_json = ?")
            values.append(json.dumps(error))
        if started_at is not None:
            assignments.append("started_at = ?")
            values.append(started_at)
        if finished_at is not None:
            assignments.append("finished_at = ?")
            values.append(finished_at)
        values.append(job_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE ingest_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                values,
            )

    def set_document_status(
        self, doc_id: str, status: DocumentStatus, profile: Profile | None = None
    ) -> None:
        with self._conn() as conn:
            now = datetime.now(UTC).isoformat()
            if profile is None:
                conn.execute(
                    "UPDATE documents SET status = ?, updated_at = ? WHERE doc_id = ?",
                    (status, now, doc_id),
                )
            else:
                conn.execute(
                    "UPDATE documents SET status = ?, profile = ?, updated_at = ? WHERE doc_id = ?",
                    (status, profile, now, doc_id),
                )

    def save_document_parse_output(
        self,
        *,
        doc_id: str,
        title: str,
        parser_chain: list[str],
        metadata: dict[str, Any],
        outline: list[OutlineNode],
        overall_confidence: float | None,
        status: DocumentStatus,
    ) -> None:
        with self._conn() as conn:
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                UPDATE documents
                SET title = ?, parser_chain_json = ?, metadata_json = ?, outline_json = ?,
                    overall_confidence = ?, status = ?, updated_at = ?
                WHERE doc_id = ?
                """,
                (
                    title,
                    json.dumps(parser_chain),
                    json.dumps(metadata),
                    json.dumps([node.model_dump() for node in outline]),
                    overall_confidence,
                    status,
                    now,
                    doc_id,
                ),
            )

    def replace_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "DELETE FROM local_search_fts WHERE doc_id = ? AND source_type = ?",
                (doc_id, "chunk"),
            )
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, order_index, section_path_json,
                    text, search_text, locator_json, method, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.order_index,
                        json.dumps(chunk.section_path),
                        chunk.text,
                        chunk.search_text,
                        chunk.locator.model_dump_json(),
                        chunk.method,
                        chunk.confidence,
                    )
                    for chunk in chunks
                ],
            )
            conn.executemany(
                """
                INSERT INTO local_search_fts (
                    doc_id, source_type, source_id, chunk_id,
                    section_path_json, order_index, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.doc_id,
                        "chunk",
                        chunk.chunk_id,
                        chunk.chunk_id,
                        json.dumps(chunk.section_path),
                        str(chunk.order_index),
                        self._chunk_search_text(chunk),
                    )
                    for chunk in chunks
                ],
            )

    def replace_formulas(self, doc_id: str, formulas: list[FormulaRecord]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM formulas WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "DELETE FROM local_search_fts WHERE doc_id = ? AND source_type = ?",
                (doc_id, "formula"),
            )
            if not formulas:
                return
            conn.executemany(
                """
                INSERT INTO formulas (
                    formula_id, doc_id, chunk_id, section_path_json,
                    page, bbox_json, latex, source, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        formula.formula_id,
                        formula.doc_id,
                        formula.chunk_id,
                        json.dumps(formula.section_path),
                        formula.page,
                        json.dumps(formula.bbox) if formula.bbox is not None else None,
                        formula.latex,
                        formula.source,
                        formula.confidence,
                        formula.status,
                    )
                    for formula in formulas
                ],
            )
            conn.executemany(
                """
                INSERT INTO local_search_fts (
                    doc_id, source_type, source_id, chunk_id,
                    section_path_json, order_index, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        formula.doc_id,
                        "formula",
                        formula.formula_id,
                        formula.chunk_id,
                        json.dumps(formula.section_path),
                        str(formula.page or 0),
                        self._formula_search_text(formula),
                    )
                    for formula in formulas
                ],
            )

    def replace_images(self, doc_id: str, images: list[ImageRecord]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM images WHERE doc_id = ?", (doc_id,))
            if not images:
                return
            conn.executemany(
                """
                INSERT INTO images (
                    image_id, doc_id, order_index, section_path_json,
                    spine_id, page, bbox_json, href, anchor, alt, caption, media_type,
                    file_path, width, height, source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        image.image_id,
                        image.doc_id,
                        image.order_index,
                        json.dumps(image.section_path),
                        image.spine_id,
                        image.page,
                        json.dumps(image.bbox) if image.bbox is not None else None,
                        image.href,
                        image.anchor,
                        image.alt,
                        image.caption,
                        image.media_type,
                        image.file_path,
                        image.width,
                        image.height,
                        image.source,
                        image.status,
                    )
                    for image in images
                ],
            )

    def replace_pdf_tables(self, doc_id: str, tables: list[PdfTableRecord]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM pdf_tables WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "DELETE FROM local_search_fts WHERE doc_id = ? AND source_type = ?",
                (doc_id, "pdf_table"),
            )
            if not tables:
                return
            conn.executemany(
                """
                INSERT INTO pdf_tables (
                    table_id, doc_id, order_index, section_path_json,
                    page_range_json, bbox_json, caption, headers_json, rows_json,
                    markdown, html, file_path, width, height, merged,
                    merge_confidence, segments_json, source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        table.table_id,
                        table.doc_id,
                        table.order_index,
                        json.dumps(table.section_path),
                        json.dumps(table.page_range)
                        if table.page_range is not None
                        else None,
                        json.dumps(table.bbox) if table.bbox is not None else None,
                        table.caption,
                        json.dumps(table.headers),
                        json.dumps(table.rows),
                        table.markdown,
                        table.html,
                        table.file_path,
                        table.width,
                        table.height,
                        1 if table.merged else 0,
                        table.merge_confidence,
                        json.dumps(
                            [
                                segment.model_dump(mode="json")
                                for segment in table.segments
                            ]
                        ),
                        table.source,
                        table.status,
                    )
                    for table in tables
                ],
            )
            conn.executemany(
                """
                INSERT INTO local_search_fts (
                    doc_id, source_type, source_id, chunk_id,
                    section_path_json, order_index, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        table.doc_id,
                        "pdf_table",
                        table.table_id,
                        None,
                        json.dumps(table.section_path),
                        str(table.order_index),
                        self._pdf_table_search_text(table),
                    )
                    for table in tables
                ],
            )

    def replace_pdf_figures(self, doc_id: str, figures: list[PdfFigureRecord]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM pdf_figures WHERE doc_id = ?", (doc_id,))
            if not figures:
                return
            conn.executemany(
                """
                INSERT INTO pdf_figures (
                    figure_id, doc_id, order_index, section_path_json,
                    page, bbox_json, caption, kind, file_path, width, height,
                    source, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        figure.figure_id,
                        figure.doc_id,
                        figure.order_index,
                        json.dumps(figure.section_path),
                        figure.page,
                        json.dumps(figure.bbox) if figure.bbox is not None else None,
                        figure.caption,
                        figure.kind,
                        figure.file_path,
                        figure.width,
                        figure.height,
                        figure.source,
                        figure.status,
                    )
                    for figure in figures
                ],
            )

    def search_local(
        self,
        *,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Search local FTS evidence rows for exact terms and structured artifacts."""

        if top_k <= 0:
            return []
        match_query = self._fts_match_query(query)
        if match_query is None:
            return []

        doc_filter = ""
        values: list[Any] = [match_query]
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            doc_filter = f" AND local_search_fts.doc_id IN ({placeholders})"
            values.extend(doc_ids)
        values.append(top_k)

        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    local_search_fts.doc_id,
                    local_search_fts.source_type,
                    local_search_fts.source_id,
                    local_search_fts.chunk_id,
                    local_search_fts.section_path_json,
                    local_search_fts.order_index,
                    snippet(local_search_fts, 6, '[', ']', '...', 24) AS snippet,
                    bm25(local_search_fts) AS rank_score,
                    chunks.locator_json AS locator_json
                FROM local_search_fts
                LEFT JOIN chunks
                    ON chunks.doc_id = local_search_fts.doc_id
                    AND chunks.chunk_id = local_search_fts.chunk_id
                WHERE local_search_fts MATCH ?{doc_filter}
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                values,
            ).fetchall()

        hits: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            section_path = json.loads(row["section_path_json"])
            locator = (
                json.loads(row["locator_json"])
                if row["locator_json"]
                else {
                    "doc_id": row["doc_id"],
                    "chunk_id": row["chunk_id"] or row["source_id"],
                    "section_path": section_path,
                    "method": "sqlite_fts",
                }
            )
            source_type = row["source_type"]
            hit = {
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"] or row["source_id"],
                "source_type": source_type,
                "source_id": row["source_id"],
                "locator": locator,
                "section_path": section_path,
                "snippet": row["snippet"],
                "local_rank": index,
                "local_score": -float(row["rank_score"]),
                "retrieval_source": "sqlite_fts",
            }
            if source_type == "formula":
                hit["read_hint"] = (
                    "Use pdf_book_read_formula or pdf_paper_read_formula with source_id."
                )
            elif source_type == "pdf_table":
                hit["read_hint"] = "Use pdf_read_table with source_id."
            hits.append(hit)
        return hits

    @classmethod
    def _fts_match_query(cls, query: str) -> str | None:
        tokens = [item.lower() for item in cls._FTS_TOKEN_RE.findall(query)]
        unique_tokens = list(dict.fromkeys(token for token in tokens if token))
        if not unique_tokens:
            return None
        return " OR ".join(f'"{token}"' for token in unique_tokens[:32])

    @staticmethod
    def _chunk_search_text(chunk: ChunkRecord) -> str:
        return "\n".join(
            [
                " / ".join(chunk.section_path),
                chunk.search_text,
                chunk.text,
            ]
        )

    @staticmethod
    def _formula_search_text(formula: FormulaRecord) -> str:
        return "\n".join(
            [
                " / ".join(formula.section_path),
                formula.latex,
                formula.source,
                formula.status,
            ]
        )

    @staticmethod
    def _pdf_table_search_text(table: PdfTableRecord) -> str:
        return "\n".join(
            [
                " / ".join(table.section_path),
                table.caption or "",
                " ".join(table.headers),
                table.markdown,
            ]
        )

    def get_chunk(self, doc_id: str, chunk_id: str) -> ChunkRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND chunk_id = ?",
                (doc_id, chunk_id),
            ).fetchone()
            return self._row_to_chunk(row) if row else None

    def get_chunks_window(
        self, doc_id: str, center_order: int, before: int, after: int
    ) -> list[ChunkRecord]:
        low = max(0, center_order - before)
        high = center_order + after
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE doc_id = ? AND order_index BETWEEN ? AND ?
                ORDER BY order_index ASC
                """,
                (doc_id, low, high),
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def list_chunks(self, doc_id: str) -> list[ChunkRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE doc_id = ?
                ORDER BY order_index ASC
                """,
                (doc_id,),
            ).fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def get_formula(self, formula_id: str) -> FormulaRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM formulas WHERE formula_id = ?",
                (formula_id,),
            ).fetchone()
            return self._row_to_formula(row) if row else None

    def list_formulas(self, doc_id: str) -> list[FormulaRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM formulas
                WHERE doc_id = ?
                ORDER BY formula_id ASC
                """,
                (doc_id,),
            ).fetchall()
            return [self._row_to_formula(row) for row in rows]

    def get_image(self, image_id: str) -> ImageRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE image_id = ?",
                (image_id,),
            ).fetchone()
            return self._row_to_image(row) if row else None

    def list_images(self, doc_id: str) -> list[ImageRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM images
                WHERE doc_id = ?
                ORDER BY order_index ASC
                """,
                (doc_id,),
            ).fetchall()
            return [self._row_to_image(row) for row in rows]

    def get_pdf_table(self, table_id: str) -> PdfTableRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pdf_tables WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            return self._row_to_pdf_table(row) if row else None

    def list_pdf_tables(self, doc_id: str) -> list[PdfTableRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM pdf_tables
                WHERE doc_id = ?
                ORDER BY order_index ASC
                """,
                (doc_id,),
            ).fetchall()
            return [self._row_to_pdf_table(row) for row in rows]

    def get_pdf_figure(self, figure_id: str) -> PdfFigureRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pdf_figures WHERE figure_id = ?",
                (figure_id,),
            ).fetchone()
            return self._row_to_pdf_figure(row) if row else None

    def list_pdf_figures(self, doc_id: str) -> list[PdfFigureRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM pdf_figures
                WHERE doc_id = ?
                ORDER BY order_index ASC
                """,
                (doc_id,),
            ).fetchall()
            return [self._row_to_pdf_figure(row) for row in rows]

    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            path=row["path"],
            type=row["type"],
            sha256=row["sha256"],
            mtime=row["mtime"],
            title=row["title"],
            status=row["status"],
            profile=row["profile"],
            parser_chain=json.loads(row["parser_chain_json"]),
            metadata=json.loads(row["metadata_json"]),
            outline=[OutlineNode(**item) for item in json.loads(row["outline_json"])],
            overall_confidence=row["overall_confidence"],
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        locator = json.loads(row["locator_json"])
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            order_index=row["order_index"],
            section_path=json.loads(row["section_path_json"]),
            text=row["text"],
            search_text=row["search_text"],
            locator=locator,
            method=row["method"],
            confidence=row["confidence"],
        )

    def _row_to_formula(self, row: sqlite3.Row) -> FormulaRecord:
        return FormulaRecord(
            formula_id=row["formula_id"],
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            section_path=json.loads(row["section_path_json"]),
            page=row["page"],
            bbox=json.loads(row["bbox_json"]) if row["bbox_json"] is not None else None,
            latex=row["latex"],
            source=row["source"],
            confidence=row["confidence"],
            status=row["status"],
        )

    def _row_to_image(self, row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            image_id=row["image_id"],
            doc_id=row["doc_id"],
            order_index=row["order_index"],
            section_path=json.loads(row["section_path_json"]),
            spine_id=row["spine_id"],
            page=row["page"],
            bbox=json.loads(row["bbox_json"]) if row["bbox_json"] is not None else None,
            href=row["href"],
            anchor=row["anchor"],
            alt=row["alt"],
            caption=row["caption"],
            media_type=row["media_type"],
            file_path=row["file_path"],
            width=row["width"],
            height=row["height"],
            source=row["source"],
            status=row["status"],
        )

    def _row_to_pdf_table(self, row: sqlite3.Row) -> PdfTableRecord:
        return PdfTableRecord(
            table_id=row["table_id"],
            doc_id=row["doc_id"],
            order_index=row["order_index"],
            section_path=json.loads(row["section_path_json"]),
            page_range=json.loads(row["page_range_json"])
            if row["page_range_json"] is not None
            else None,
            bbox=json.loads(row["bbox_json"]) if row["bbox_json"] is not None else None,
            caption=row["caption"],
            headers=json.loads(row["headers_json"]),
            rows=json.loads(row["rows_json"]),
            markdown=row["markdown"],
            html=row["html"],
            file_path=row["file_path"],
            width=row["width"],
            height=row["height"],
            merged=bool(row["merged"]),
            merge_confidence=row["merge_confidence"],
            segments=json.loads(row["segments_json"]),
            source=row["source"],
            status=row["status"],
        )

    def _row_to_pdf_figure(self, row: sqlite3.Row) -> PdfFigureRecord:
        return PdfFigureRecord(
            figure_id=row["figure_id"],
            doc_id=row["doc_id"],
            order_index=row["order_index"],
            section_path=json.loads(row["section_path_json"]),
            page=row["page"],
            bbox=json.loads(row["bbox_json"]) if row["bbox_json"] is not None else None,
            caption=row["caption"],
            kind=row["kind"],
            file_path=row["file_path"],
            width=row["width"],
            height=row["height"],
            source=row["source"],
            status=row["status"],
        )

    def _row_to_ingest_job(self, row: sqlite3.Row) -> IngestJobRecord:
        return IngestJobRecord(
            job_id=row["job_id"],
            doc_id=row["doc_id"],
            path=row["path"],
            profile=row["profile"],
            status=row["status"],
            stage=row["stage"],
            force=bool(row["force"]),
            message=row["message"],
            progress=json.loads(row["progress_json"]) if row["progress_json"] else {},
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
