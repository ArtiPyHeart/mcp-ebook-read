"""SQLite persistence for documents and chunks."""

from __future__ import annotations

import json
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
    ImageRecord,
    OutlineNode,
    Profile,
)


class CatalogStore:
    """Catalog and chunk persistence backed by sqlite."""

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
            count = conn.execute(
                f"SELECT COUNT(*) AS c FROM documents WHERE path IN ({placeholders})",
                normalized,
            ).fetchone()["c"]
            conn.executemany(
                "DELETE FROM documents WHERE path = ?",
                [(path,) for path in normalized],
            )
            return int(count)

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

    def replace_formulas(self, doc_id: str, formulas: list[FormulaRecord]) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM formulas WHERE doc_id = ?", (doc_id,))
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
