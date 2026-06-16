"""SQLite persistence for documents and chunks."""

from __future__ import annotations

import json
import re
import sqlite3
from difflib import SequenceMatcher
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

    SCHEMA_VERSION = 2
    DOCUMENT_GRAPH_SCHEMA_VERSION = 1
    _FTS_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")
    _NON_ALNUM_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
    _FORMULA_QUERY_TERMS = {"formula", "equation", "latex", "证明", "公式", "方程"}
    _FIGURE_QUERY_TERMS = {
        "figure",
        "fig",
        "image",
        "picture",
        "chart",
        "diagram",
        "plot",
        "图",
        "图片",
        "图表",
        "示意图",
    }
    _TABLE_QUERY_TERMS = {"table", "tabular", "表", "表格"}
    _REFERENCE_QUERY_TERMS = {
        "citation",
        "cite",
        "reference",
        "bibliography",
        "引用",
        "参考文献",
    }
    _FRONT_MATTER_TERMS = {
        "preface",
        "foreword",
        "acknowledgement",
        "acknowledgments",
        "license",
        "copyright",
        "index",
        "appendix",
        "references",
        "bibliography",
    }

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._replace_incompatible_db_if_needed()
        self._init_db()

    def _replace_incompatible_db_if_needed(self) -> None:
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                metadata_exists = conn.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'catalog_metadata'
                    """
                ).fetchone()
                if metadata_exists is None:
                    self._backup_incompatible_db("missing_catalog_metadata")
                    return
                rows = conn.execute(
                    "SELECT key, value FROM catalog_metadata"
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            self._backup_incompatible_db("unreadable_sqlite")
            return

        metadata = {str(row["key"]): str(row["value"]) for row in rows}
        expected = {
            "schema_version": str(self.SCHEMA_VERSION),
            "document_graph_schema_version": str(self.DOCUMENT_GRAPH_SCHEMA_VERSION),
        }
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                self._backup_incompatible_db(f"{key}_mismatch")
                return

    def _backup_incompatible_db(self, reason: str) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = self.db_path.with_name(
            f"{self.db_path.name}.incompatible-{reason}-{timestamp}.bak"
        )
        self.db_path.replace(backup_path)
        return backup_path

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_staging_copy(self, *, suffix: str) -> "CatalogStore":
        """Create a same-directory database copy for atomic ingest finalization."""

        safe_suffix = re.sub(r"[^0-9A-Za-z_.-]+", "-", suffix).strip(".-")
        if not safe_suffix:
            safe_suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        staging_path = self.db_path.with_name(
            f"{self.db_path.name}.staging-{safe_suffix}.tmp"
        )
        staging_path.unlink(missing_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(self.db_path)
        target = sqlite3.connect(staging_path)
        try:
            source.backup(target)
            target.commit()
        finally:
            target.close()
            source.close()
        return CatalogStore(staging_path)

    def replace_with_staging_copy(self, staging: "CatalogStore") -> None:
        """Atomically replace the active SQLite database with a staged copy."""

        if staging.db_path == self.db_path:
            return
        if staging.db_path.parent != self.db_path.parent:
            raise ValueError("staging database must live beside the active catalog")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        staging.db_path.replace(self.db_path)

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

                CREATE TABLE IF NOT EXISTS catalog_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
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

                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    stable_ref TEXT NOT NULL,
                    title TEXT,
                    text TEXT,
                    locator_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    order_index INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_graph_nodes_doc_kind
                ON graph_nodes(doc_id, kind, order_index);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_nodes_doc_ref
                ON graph_nodes(doc_id, stable_ref);

                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_node_id) REFERENCES graph_nodes(node_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_graph_edges_doc_kind
                ON graph_edges(doc_id, kind);

                CREATE INDEX IF NOT EXISTS idx_graph_edges_source
                ON graph_edges(source_node_id, kind);

                CREATE INDEX IF NOT EXISTS idx_graph_edges_target
                ON graph_edges(target_node_id, kind);

                CREATE TABLE IF NOT EXISTS diagnostics (
                    diagnostic_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    code TEXT NOT NULL,
                    message TEXT NOT NULL,
                    node_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    FOREIGN KEY (node_id) REFERENCES graph_nodes(node_id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_diagnostics_doc
                ON diagnostics(doc_id, severity, code);

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source_node_id TEXT,
                    source_ref TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    media_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_doc_kind
                ON artifacts(doc_id, kind);

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
                    owner_id TEXT,
                    claimed_at TEXT,
                    heartbeat_at TEXT,
                    lease_expires_at TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ingest_jobs_doc_created
                ON ingest_jobs(doc_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status_updated
                ON ingest_jobs(status, updated_at DESC);
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
            for column in (
                "owner_id",
                "claimed_at",
                "heartbeat_at",
                "lease_expires_at",
            ):
                if column not in ingest_job_columns:
                    conn.execute(f"ALTER TABLE ingest_jobs ADD COLUMN {column} TEXT")
            now = datetime.now(UTC).isoformat()
            conn.executemany(
                """
                INSERT INTO catalog_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE
                SET value = excluded.value, updated_at = excluded.updated_at
                """,
                [
                    ("schema_version", str(self.SCHEMA_VERSION), now),
                    (
                        "document_graph_schema_version",
                        str(self.DOCUMENT_GRAPH_SCHEMA_VERSION),
                        now,
                    ),
                ],
            )

    def catalog_metadata(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM catalog_metadata").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    @staticmethod
    def _page_range_numbers(start: int | None, end: int | None) -> list[int]:
        if start is None or end is None:
            return []
        left = max(1, int(start))
        right = max(left, int(end))
        return list(range(left, right + 1))

    @staticmethod
    def _flatten_outline(outline: list[OutlineNode]) -> list[OutlineNode]:
        flattened: list[OutlineNode] = []

        def walk(nodes: list[OutlineNode]) -> None:
            for node in nodes:
                flattened.append(node)
                walk(node.children)

        walk(outline)
        return flattened

    @staticmethod
    def _metadata_list(metadata: dict[str, Any], key: str) -> list[Any]:
        value = metadata.get(key)
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _metadata_item_dict(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return item
        return {"text": str(item)}

    @classmethod
    def _metadata_item_id(
        cls,
        item: Any,
        *,
        id_keys: tuple[str, ...],
        fallback: str,
    ) -> str:
        item_dict = cls._metadata_item_dict(item)
        for key in id_keys:
            value = item_dict.get(key)
            if value is not None and str(value).strip():
                return str(value).strip().lstrip("#")
        return fallback

    @classmethod
    def _metadata_item_text(
        cls,
        item: Any,
        *,
        text_keys: tuple[str, ...],
        fallback: str,
    ) -> str:
        item_dict = cls._metadata_item_dict(item)
        for key in text_keys:
            value = item_dict.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return fallback

    def sidecar_summary(self) -> dict[str, int]:
        with self._conn() as conn:
            documents_count = int(
                conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            artifacts_count = int(
                conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
            )
            nodes_count = int(
                conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
            )
            edges_count = int(
                conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
            )
            diagnostics_count = int(
                conn.execute("SELECT COUNT(*) FROM diagnostics").fetchone()[0]
            )
        return {
            "documents_count": documents_count,
            "artifacts_count": artifacts_count,
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "diagnostics_count": diagnostics_count,
            "db_size_bytes": self.db_size_bytes(),
            "total_bytes": self.sidecar_total_bytes(),
        }

    def sidecar_total_bytes(self) -> int:
        root = self.db_path.parent
        if not root.exists():
            return 0
        total = 0
        for item in root.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    def _outline_search_rows(
        self,
        *,
        doc_id: str,
        title: str,
        outline: list[OutlineNode],
    ) -> list[tuple[str, str, str, str | None, str, str, str]]:
        rows: list[tuple[str, str, str, str | None, str, str, str]] = [
            (
                doc_id,
                "document",
                doc_id,
                None,
                json.dumps([]),
                "0",
                title,
            )
        ]

        def walk(nodes: list[OutlineNode], path: list[str]) -> None:
            for index, node in enumerate(nodes):
                current_path = [*path, node.title]
                rows.append(
                    (
                        doc_id,
                        "outline_node",
                        node.id,
                        None,
                        json.dumps(current_path),
                        str(index),
                        "\n".join(
                            [
                                " / ".join(current_path),
                                node.title,
                                node.spine_ref or "",
                                str(node.page_start or ""),
                                str(node.page_end or ""),
                            ]
                        ),
                    )
                )
                walk(node.children, current_path)

        walk(outline, [])
        return rows

    def _graph_search_rows(
        self,
        *,
        graph_nodes: list[tuple[Any, ...]],
        diagnostic_rows: list[tuple[Any, ...]],
    ) -> list[tuple[str, str, str, str | None, str, str, str]]:
        rows: list[tuple[str, str, str, str | None, str, str, str]] = []
        indexed_kinds = {"page", "reference", "citation", "artifact"}
        for node in graph_nodes:
            (
                _node_id,
                doc_id,
                kind,
                stable_ref,
                title,
                text,
                locator_json,
                metadata_json,
                order_index,
                _created_at,
            ) = node
            if kind not in indexed_kinds:
                continue
            locator = json.loads(locator_json)
            metadata = json.loads(metadata_json)
            rows.append(
                (
                    str(doc_id),
                    str(kind),
                    self._graph_source_id(str(kind), str(stable_ref)),
                    None,
                    json.dumps(locator.get("section_path") or []),
                    str(order_index or 0),
                    "\n".join(
                        [
                            str(stable_ref),
                            str(title or ""),
                            str(text or ""),
                            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                        ]
                    ),
                )
            )

        for diagnostic in diagnostic_rows:
            (
                diagnostic_id,
                doc_id,
                severity,
                code,
                message,
                _node_id,
                metadata_json,
                _created_at,
            ) = diagnostic
            rows.append(
                (
                    str(doc_id),
                    "diagnostic",
                    str(diagnostic_id),
                    None,
                    json.dumps([]),
                    "0",
                    "\n".join(
                        [
                            str(severity),
                            str(code),
                            str(message),
                            str(metadata_json),
                        ]
                    ),
                )
            )
        return rows

    @staticmethod
    def _graph_node_id(doc_id: str, kind: str, stable_ref: str) -> str:
        return f"{doc_id}:{kind}:{stable_ref}"

    @staticmethod
    def _graph_edge_id(
        doc_id: str,
        source_node_id: str,
        target_node_id: str,
        kind: str,
    ) -> str:
        return f"{doc_id}:{kind}:{source_node_id}->{target_node_id}"

    def rebuild_document_graph(self, doc_id: str) -> None:
        """Rebuild typed graph nodes/edges from persisted document evidence."""

        doc = self.get_document_by_id(doc_id)
        if doc is None:
            return
        chunks = self.list_chunks(doc_id)
        formulas = self.list_formulas(doc_id)
        images = self.list_images(doc_id)
        tables = self.list_pdf_tables(doc_id)
        figures = self.list_pdf_figures(doc_id)
        references = self._metadata_list(doc.metadata, "references")
        citations = self._metadata_list(doc.metadata, "citations")
        raw_artifacts = self._metadata_list(doc.metadata, "raw_artifacts")

        now = datetime.now(UTC).isoformat()
        graph_nodes: list[tuple[Any, ...]] = []
        graph_edges: list[tuple[Any, ...]] = []
        artifact_rows: list[tuple[Any, ...]] = []
        diagnostic_rows: list[tuple[Any, ...]] = []
        stable_ref_counts: dict[str, int] = {}
        used_stable_refs: set[str] = set()

        def unique_stable_ref(stable_ref: str) -> str:
            count = stable_ref_counts.get(stable_ref, 0) + 1
            stable_ref_counts[stable_ref] = count
            candidate = stable_ref if count == 1 else f"{stable_ref}#dup-{count}"
            while candidate in used_stable_refs:
                count += 1
                stable_ref_counts[stable_ref] = count
                candidate = f"{stable_ref}#dup-{count}"
            used_stable_refs.add(candidate)
            return candidate

        def add_node(
            *,
            kind: str,
            stable_ref: str,
            title: str | None,
            text: str | None,
            locator: dict[str, Any],
            metadata: dict[str, Any] | None = None,
            order_index: int | None = None,
        ) -> str:
            effective_stable_ref = unique_stable_ref(stable_ref)
            node_metadata = dict(metadata or {})
            if effective_stable_ref != stable_ref:
                node_metadata["original_stable_ref"] = stable_ref
                node_metadata["stable_ref_collision_index"] = stable_ref_counts[
                    stable_ref
                ]
            node_id = self._graph_node_id(doc_id, kind, effective_stable_ref)
            graph_nodes.append(
                (
                    node_id,
                    doc_id,
                    kind,
                    effective_stable_ref,
                    title,
                    text,
                    json.dumps(locator),
                    json.dumps(node_metadata),
                    order_index,
                    now,
                )
            )
            return node_id

        def add_artifact(
            *,
            kind: str,
            source_node_id: str,
            source_ref: str,
            file_path: str,
            media_type: str | None,
            width: int | None,
            height: int | None,
            title: str | None,
            locator: dict[str, Any],
            metadata: dict[str, Any] | None = None,
            order_index: int | None = None,
        ) -> str:
            artifact_id = f"{doc_id}:{kind}:{source_ref}"
            artifact_node_id = add_node(
                kind="artifact",
                stable_ref=f"artifact:{artifact_id}",
                title=title or Path(file_path).name,
                text=Path(file_path).name,
                locator={**locator, "artifact_id": artifact_id},
                metadata={
                    **(metadata or {}),
                    "kind": kind,
                    "source_ref": source_ref,
                    "file_path": file_path,
                    "media_type": media_type,
                    "width": width,
                    "height": height,
                },
                order_index=order_index,
            )
            artifact_rows.append(
                (
                    artifact_id,
                    doc_id,
                    kind,
                    source_node_id,
                    source_ref,
                    file_path,
                    media_type,
                    width,
                    height,
                    json.dumps(metadata or {}),
                    now,
                )
            )
            add_edge(artifact_node_id, source_node_id, "renders_from")
            add_edge(source_node_id, artifact_node_id, "contains")
            return artifact_node_id

        def add_diagnostic(
            *,
            severity: str,
            code: str,
            message: str,
            node_id: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            diagnostic_rows.append(
                (
                    f"{doc_id}:{code}:{len(diagnostic_rows)}",
                    doc_id,
                    severity,
                    code,
                    message,
                    node_id,
                    json.dumps(metadata or {}),
                    now,
                )
            )

        def add_edge(
            source_node_id: str,
            target_node_id: str,
            kind: str,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            graph_edges.append(
                (
                    self._graph_edge_id(doc_id, source_node_id, target_node_id, kind),
                    doc_id,
                    source_node_id,
                    target_node_id,
                    kind,
                    json.dumps(metadata or {}),
                    now,
                )
            )

        document_node_id = add_node(
            kind="document",
            stable_ref=doc_id,
            title=doc.title,
            text=None,
            locator={"doc_id": doc_id, "path": doc.path},
            metadata={
                "path": doc.path,
                "type": doc.type,
                "profile": doc.profile,
                "status": doc.status,
                "sha256": doc.sha256,
            },
            order_index=0,
        )

        for index, raw_artifact in enumerate(raw_artifacts):
            artifact = self._metadata_item_dict(raw_artifact)
            file_path = str(artifact.get("file_path") or "").strip()
            if not file_path:
                continue
            source_ref = str(artifact.get("artifact_id") or f"raw-{index}")
            add_artifact(
                kind="raw",
                source_node_id=document_node_id,
                source_ref=source_ref,
                file_path=file_path,
                media_type=str(artifact.get("media_type") or "text/plain"),
                width=None,
                height=None,
                title=str(artifact.get("name") or source_ref),
                locator={
                    "doc_id": doc_id,
                    "artifact_id": source_ref,
                    "section_path": [],
                },
                metadata={
                    "name": artifact.get("name"),
                    "bytes": artifact.get("bytes"),
                    "source": "raw_artifacts",
                },
                order_index=index,
            )

        outline_by_path: dict[tuple[str, ...], str] = {}

        def walk_outline(
            nodes: list[OutlineNode],
            *,
            parent_node_id: str,
            parent_path: tuple[str, ...],
        ) -> None:
            for index, outline in enumerate(nodes):
                stable_ref = f"outline:{outline.id}"
                current_path = (*parent_path, outline.title)
                node_id = add_node(
                    kind="outline_node",
                    stable_ref=stable_ref,
                    title=outline.title,
                    text=None,
                    locator={
                        "doc_id": doc_id,
                        "node_id": outline.id,
                        "section_path": list(current_path),
                        "page_range": [outline.page_start, outline.page_end]
                        if outline.page_start is not None
                        and outline.page_end is not None
                        else None,
                        "spine_ref": outline.spine_ref,
                    },
                    metadata={"level": outline.level},
                    order_index=index,
                )
                outline_by_path[current_path] = node_id
                add_edge(parent_node_id, node_id, "contains")
                if parent_node_id != document_node_id:
                    add_edge(parent_node_id, node_id, "child")
                    add_edge(node_id, parent_node_id, "parent")
                walk_outline(
                    outline.children,
                    parent_node_id=node_id,
                    parent_path=current_path,
                )

        walk_outline(doc.outline, parent_node_id=document_node_id, parent_path=())

        page_numbers: set[int] = set()
        if doc.type == "pdf":
            for outline in self._flatten_outline(doc.outline):
                page_numbers.update(
                    self._page_range_numbers(outline.page_start, outline.page_end)
                )
            for chunk in chunks:
                if chunk.locator.page_range:
                    page_numbers.update(
                        self._page_range_numbers(
                            chunk.locator.page_range[0],
                            chunk.locator.page_range[-1],
                        )
                    )
            for formula in formulas:
                if formula.page is not None:
                    page_numbers.add(formula.page)
            for image in images:
                if image.page is not None:
                    page_numbers.add(image.page)
            for table in tables:
                if table.page_range:
                    page_numbers.update(
                        self._page_range_numbers(
                            table.page_range[0],
                            table.page_range[-1],
                        )
                    )
                for segment in table.segments:
                    if segment.page is not None:
                        page_numbers.add(segment.page)
            for figure in figures:
                if figure.page is not None:
                    page_numbers.add(figure.page)

        page_node_by_page: dict[int, str] = {}
        previous_page_node_id: str | None = None
        for page in sorted(item for item in page_numbers if item > 0):
            page_node_id = add_node(
                kind="page",
                stable_ref=f"page:{page}",
                title=f"Page {page}",
                text=None,
                locator={"doc_id": doc_id, "page": page},
                metadata={"source": "derived_from_locators"},
                order_index=page,
            )
            page_node_by_page[page] = page_node_id
            add_edge(document_node_id, page_node_id, "contains")
            if previous_page_node_id is not None:
                add_edge(previous_page_node_id, page_node_id, "next")
                add_edge(page_node_id, previous_page_node_id, "previous")
            previous_page_node_id = page_node_id

        def nearest_outline_node(section_path: list[str]) -> str | None:
            for end in range(len(section_path), 0, -1):
                node_id = outline_by_path.get(tuple(section_path[:end]))
                if node_id is not None:
                    return node_id
            return None

        def add_page_edges(
            *,
            target_node_id: str,
            page: int | None = None,
            page_range: list[int] | None = None,
        ) -> None:
            pages: list[int] = []
            if page is not None:
                pages.append(page)
            if page_range:
                pages.extend(self._page_range_numbers(page_range[0], page_range[-1]))
            for page_number in dict.fromkeys(pages):
                page_node_id = page_node_by_page.get(page_number)
                if page_node_id is not None:
                    add_edge(page_node_id, target_node_id, "contains")

        chunk_node_by_chunk_id: dict[str, str] = {}
        previous_chunk_node_id: str | None = None
        for chunk in chunks:
            node_id = add_node(
                kind="chunk",
                stable_ref=f"chunk:{chunk.chunk_id}",
                title=" / ".join(chunk.section_path) or None,
                text=chunk.text,
                locator=chunk.locator.model_dump(mode="json"),
                metadata={
                    "method": chunk.method,
                    "confidence": chunk.confidence,
                    "search_text": chunk.search_text,
                },
                order_index=chunk.order_index,
            )
            chunk_node_by_chunk_id[chunk.chunk_id] = node_id
            add_edge(document_node_id, node_id, "contains")
            outline_node_id = nearest_outline_node(chunk.section_path)
            if outline_node_id is not None:
                add_edge(outline_node_id, node_id, "contains")
            add_page_edges(target_node_id=node_id, page_range=chunk.locator.page_range)
            if previous_chunk_node_id is not None:
                add_edge(previous_chunk_node_id, node_id, "next")
                add_edge(node_id, previous_chunk_node_id, "previous")
            previous_chunk_node_id = node_id

        for formula in formulas:
            node_id = add_node(
                kind="formula",
                stable_ref=f"formula:{formula.formula_id}",
                title=formula.formula_id,
                text=formula.latex,
                locator={
                    "doc_id": doc_id,
                    "formula_id": formula.formula_id,
                    "chunk_id": formula.chunk_id,
                    "section_path": formula.section_path,
                    "page": formula.page,
                    "bbox": formula.bbox,
                },
                metadata={
                    "source": formula.source,
                    "confidence": formula.confidence,
                    "status": formula.status,
                },
                order_index=formula.page,
            )
            add_edge(document_node_id, node_id, "contains")
            if formula.chunk_id and formula.chunk_id in chunk_node_by_chunk_id:
                chunk_node_id = chunk_node_by_chunk_id[formula.chunk_id]
                add_edge(chunk_node_id, node_id, "mentions")
                add_edge(node_id, chunk_node_id, "near")
            outline_node_id = nearest_outline_node(formula.section_path)
            if outline_node_id is not None:
                add_edge(outline_node_id, node_id, "contains")
            add_page_edges(target_node_id=node_id, page=formula.page)
            if formula.status == "unresolved":
                add_diagnostic(
                    severity="warning",
                    code="FORMULA_UNRESOLVED",
                    message="Formula was not resolved to high-confidence LaTeX.",
                    node_id=node_id,
                    metadata={
                        "formula_id": formula.formula_id,
                        "status": formula.status,
                        "source": formula.source,
                    },
                )

        for image in images:
            node_id = add_node(
                kind="image",
                stable_ref=f"image:{image.image_id}",
                title=image.caption or image.alt or image.image_id,
                text=image.caption or image.alt,
                locator={
                    "doc_id": doc_id,
                    "image_id": image.image_id,
                    "section_path": image.section_path,
                    "spine_id": image.spine_id,
                    "page": image.page,
                    "bbox": image.bbox,
                },
                metadata={
                    "file_path": image.file_path,
                    "media_type": image.media_type,
                    "href": image.href,
                    "anchor": image.anchor,
                    "width": image.width,
                    "height": image.height,
                    "source": image.source,
                    "status": image.status,
                },
                order_index=image.order_index,
            )
            add_edge(document_node_id, node_id, "contains")
            outline_node_id = nearest_outline_node(image.section_path)
            if outline_node_id is not None:
                add_edge(outline_node_id, node_id, "contains")
            add_page_edges(target_node_id=node_id, page=image.page)
            add_artifact(
                kind="image",
                source_node_id=node_id,
                source_ref=image.image_id,
                file_path=image.file_path,
                media_type=image.media_type,
                width=image.width,
                height=image.height,
                title=image.caption or image.alt or image.image_id,
                locator={
                    "doc_id": doc_id,
                    "image_id": image.image_id,
                    "section_path": image.section_path,
                    "page": image.page,
                    "bbox": image.bbox,
                },
                metadata={
                    "source": image.source,
                    "status": image.status,
                    "href": image.href,
                    "anchor": image.anchor,
                },
                order_index=image.order_index,
            )

        for table in tables:
            node_id = add_node(
                kind="table",
                stable_ref=f"table:{table.table_id}",
                title=table.caption or table.table_id,
                text=table.markdown or table.caption,
                locator={
                    "doc_id": doc_id,
                    "table_id": table.table_id,
                    "section_path": table.section_path,
                    "page_range": table.page_range,
                    "bbox": table.bbox,
                },
                metadata={
                    "file_path": table.file_path,
                    "headers": table.headers,
                    "rows_count": len(table.rows),
                    "merged": table.merged,
                    "merge_confidence": table.merge_confidence,
                    "source": table.source,
                    "status": table.status,
                },
                order_index=table.order_index,
            )
            add_edge(document_node_id, node_id, "contains")
            outline_node_id = nearest_outline_node(table.section_path)
            if outline_node_id is not None:
                add_edge(outline_node_id, node_id, "contains")
            add_page_edges(target_node_id=node_id, page_range=table.page_range)
            add_artifact(
                kind="table",
                source_node_id=node_id,
                source_ref=table.table_id,
                file_path=table.file_path,
                media_type="image/png",
                width=table.width,
                height=table.height,
                title=table.caption or table.table_id,
                locator={
                    "doc_id": doc_id,
                    "table_id": table.table_id,
                    "section_path": table.section_path,
                    "page_range": table.page_range,
                    "bbox": table.bbox,
                },
                metadata={
                    "source": table.source,
                    "status": table.status,
                    "merged": table.merged,
                    "merge_confidence": table.merge_confidence,
                },
                order_index=table.order_index,
            )
            for segment_index, segment in enumerate(table.segments):
                add_artifact(
                    kind="table_segment",
                    source_node_id=node_id,
                    source_ref=f"{table.table_id}:segment:{segment_index}",
                    file_path=segment.file_path,
                    media_type="image/png",
                    width=segment.width,
                    height=segment.height,
                    title=segment.caption
                    or f"{table.table_id} segment {segment_index}",
                    locator={
                        "doc_id": doc_id,
                        "table_id": table.table_id,
                        "segment_index": segment_index,
                        "section_path": table.section_path,
                        "page": segment.page,
                        "bbox": segment.bbox,
                    },
                    metadata={"source": table.source, "status": table.status},
                    order_index=table.order_index,
                )

        for figure in figures:
            node_id = add_node(
                kind="figure",
                stable_ref=f"figure:{figure.figure_id}",
                title=figure.caption or figure.figure_id,
                text=figure.caption,
                locator={
                    "doc_id": doc_id,
                    "figure_id": figure.figure_id,
                    "section_path": figure.section_path,
                    "page": figure.page,
                    "bbox": figure.bbox,
                },
                metadata={
                    "file_path": figure.file_path,
                    "kind": figure.kind,
                    "width": figure.width,
                    "height": figure.height,
                    "source": figure.source,
                    "status": figure.status,
                },
                order_index=figure.order_index,
            )
            add_edge(document_node_id, node_id, "contains")
            outline_node_id = nearest_outline_node(figure.section_path)
            if outline_node_id is not None:
                add_edge(outline_node_id, node_id, "contains")
            add_page_edges(target_node_id=node_id, page=figure.page)
            add_artifact(
                kind="figure",
                source_node_id=node_id,
                source_ref=figure.figure_id,
                file_path=figure.file_path,
                media_type="image/png",
                width=figure.width,
                height=figure.height,
                title=figure.caption or figure.figure_id,
                locator={
                    "doc_id": doc_id,
                    "figure_id": figure.figure_id,
                    "section_path": figure.section_path,
                    "page": figure.page,
                    "bbox": figure.bbox,
                },
                metadata={"source": figure.source, "status": figure.status},
                order_index=figure.order_index,
            )

        reference_node_by_key: dict[str, str] = {}
        for index, reference in enumerate(references):
            ref_id = self._metadata_item_id(
                reference,
                id_keys=("reference_id", "id", "target", "xml_id"),
                fallback=f"ref-{index + 1}",
            )
            title = self._metadata_item_text(
                reference,
                text_keys=("title", "raw_text", "text", "label"),
                fallback=ref_id,
            )
            text = self._metadata_item_text(
                reference,
                text_keys=("raw_text", "text", "title", "doi"),
                fallback=title,
            )
            node_id = add_node(
                kind="reference",
                stable_ref=f"reference:{ref_id}",
                title=title,
                text=text,
                locator={
                    "doc_id": doc_id,
                    "reference_id": ref_id,
                    "section_path": ["References"],
                },
                metadata=self._metadata_item_dict(reference),
                order_index=index,
            )
            reference_node_by_key[ref_id] = node_id
            reference_node_by_key[f"#{ref_id}"] = node_id
            add_edge(document_node_id, node_id, "contains")

        for index, citation in enumerate(citations):
            citation_id = self._metadata_item_id(
                citation,
                id_keys=("citation_id", "id", "target"),
                fallback=f"cite-{index + 1}",
            )
            citation_text = self._metadata_item_text(
                citation,
                text_keys=("text", "label", "raw_text", "target"),
                fallback=citation_id,
            )
            citation_dict = self._metadata_item_dict(citation)
            section_path = citation_dict.get("section_path")
            if not isinstance(section_path, list):
                section_path = []
            node_id = add_node(
                kind="citation",
                stable_ref=f"citation:{citation_id}",
                title=citation_text,
                text=citation_text,
                locator={
                    "doc_id": doc_id,
                    "citation_id": citation_id,
                    "target": citation_dict.get("target"),
                    "section_path": section_path,
                    "page": citation_dict.get("page"),
                },
                metadata=citation_dict,
                order_index=index,
            )
            add_edge(document_node_id, node_id, "contains")
            target = str(citation_dict.get("target") or "").strip()
            reference_node_id = reference_node_by_key.get(target)
            if reference_node_id is not None:
                add_edge(node_id, reference_node_id, "cites")
            chunk_id = citation_dict.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id in chunk_node_by_chunk_id:
                chunk_node_id = chunk_node_by_chunk_id[chunk_id]
                add_edge(chunk_node_id, node_id, "mentions")
                add_edge(node_id, chunk_node_id, "near")

        grobid_enrichment = doc.metadata.get("grobid_enrichment")
        if isinstance(grobid_enrichment, dict):
            grobid_status = grobid_enrichment.get("status")
            if grobid_status == "skipped":
                add_diagnostic(
                    severity="info",
                    code="GROBID_ENRICHMENT_SKIPPED",
                    message="Optional GROBID paper enrichment was skipped.",
                    metadata=grobid_enrichment,
                )
            elif grobid_status == "failed":
                add_diagnostic(
                    severity="warning",
                    code="GROBID_ENRICHMENT_FAILED",
                    message="Optional GROBID paper enrichment failed.",
                    metadata=grobid_enrichment,
                )

        with self._conn() as conn:
            conn.execute("DELETE FROM diagnostics WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM artifacts WHERE doc_id = ?", (doc_id,))
            conn.execute(
                """
                DELETE FROM local_search_fts
                WHERE doc_id = ? AND source_type IN (
                    'page', 'reference', 'citation', 'artifact', 'diagnostic'
                )
                """,
                (doc_id,),
            )
            conn.execute("DELETE FROM graph_edges WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM graph_nodes WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO graph_nodes (
                    node_id, doc_id, kind, stable_ref, title, text, locator_json,
                    metadata_json, order_index, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                graph_nodes,
            )
            conn.executemany(
                """
                INSERT INTO graph_edges (
                    edge_id, doc_id, source_node_id, target_node_id, kind,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                graph_edges,
            )
            conn.executemany(
                """
                INSERT INTO artifacts (
                    artifact_id, doc_id, kind, source_node_id, source_ref,
                    file_path, media_type, width, height, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                artifact_rows,
            )
            conn.executemany(
                """
                INSERT INTO diagnostics (
                    diagnostic_id, doc_id, severity, code, message, node_id,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                diagnostic_rows,
            )
            conn.executemany(
                """
                INSERT INTO local_search_fts (
                    doc_id, source_type, source_id, chunk_id,
                    section_path_json, order_index, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                self._graph_search_rows(
                    graph_nodes=graph_nodes,
                    diagnostic_rows=diagnostic_rows,
                ),
            )

    def graph_stats(self, doc_id: str) -> dict[str, int]:
        with self._conn() as conn:
            nodes_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM graph_nodes WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()[0]
            )
            edges_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM graph_edges WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()[0]
            )
            artifacts_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()[0]
            )
            diagnostics_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM diagnostics WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()[0]
            )
            kind_rows = conn.execute(
                """
                SELECT kind, COUNT(*) AS count
                FROM graph_nodes
                WHERE doc_id = ?
                GROUP BY kind
                """,
                (doc_id,),
            ).fetchall()
        stats: dict[str, int] = {
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "artifacts_count": artifacts_count,
            "diagnostics_count": diagnostics_count,
        }
        for row in kind_rows:
            stats[f"{row['kind']}_nodes_count"] = int(row["count"])
        return stats

    def list_graph_nodes(
        self,
        *,
        doc_id: str,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM graph_nodes WHERE doc_id = ?"
        values: list[Any] = [doc_id]
        if kind is not None:
            query += " AND kind = ?"
            values.append(kind)
        query += " ORDER BY kind ASC, order_index ASC, stable_ref ASC LIMIT ?"
        values.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            {
                "node_id": row["node_id"],
                "doc_id": row["doc_id"],
                "kind": row["kind"],
                "stable_ref": row["stable_ref"],
                "title": row["title"],
                "text": row["text"],
                "locator": json.loads(row["locator_json"]),
                "metadata": json.loads(row["metadata_json"]),
                "order_index": row["order_index"],
            }
            for row in rows
        ]

    def list_graph_edges(
        self,
        *,
        doc_id: str,
        kind: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM graph_edges WHERE doc_id = ?"
        values: list[Any] = [doc_id]
        if kind is not None:
            query += " AND kind = ?"
            values.append(kind)
        query += " ORDER BY kind ASC, source_node_id ASC, target_node_id ASC LIMIT ?"
        values.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            {
                "edge_id": row["edge_id"],
                "doc_id": row["doc_id"],
                "source_node_id": row["source_node_id"],
                "target_node_id": row["target_node_id"],
                "kind": row["kind"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def get_graph_node(self, *, doc_id: str, node_id: str) -> dict[str, Any] | None:
        candidates = [node_id]
        if not node_id.startswith(f"{doc_id}:"):
            candidates.extend(
                [
                    self._graph_node_id(doc_id, "chunk", f"chunk:{node_id}"),
                    self._graph_node_id(doc_id, "formula", f"formula:{node_id}"),
                    self._graph_node_id(doc_id, "image", f"image:{node_id}"),
                    self._graph_node_id(doc_id, "table", f"table:{node_id}"),
                    self._graph_node_id(doc_id, "figure", f"figure:{node_id}"),
                    self._graph_node_id(doc_id, "outline_node", f"outline:{node_id}"),
                    self._graph_node_id(doc_id, "page", f"page:{node_id}"),
                    self._graph_node_id(doc_id, "reference", f"reference:{node_id}"),
                    self._graph_node_id(doc_id, "citation", f"citation:{node_id}"),
                    self._graph_node_id(doc_id, "artifact", f"artifact:{node_id}"),
                ]
            )
        with self._conn() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM graph_nodes
                WHERE doc_id = ? AND node_id IN ({",".join("?" for _ in candidates)})
                ORDER BY CASE node_id
                    {" ".join(f"WHEN ? THEN {idx}" for idx, _ in enumerate(candidates))}
                END
                LIMIT 1
                """,
                [doc_id, *candidates, *candidates],
            ).fetchone()
        if row is None:
            return None
        return {
            "node_id": row["node_id"],
            "doc_id": row["doc_id"],
            "kind": row["kind"],
            "stable_ref": row["stable_ref"],
            "title": row["title"],
            "text": row["text"],
            "locator": json.loads(row["locator_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "order_index": row["order_index"],
        }

    def list_graph_neighbors(
        self,
        *,
        doc_id: str,
        node_id: str,
        limit: int = 80,
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    graph_edges.kind AS edge_kind,
                    graph_edges.source_node_id,
                    graph_edges.target_node_id,
                    graph_edges.metadata_json AS edge_metadata_json,
                    graph_nodes.*
                FROM graph_edges
                JOIN graph_nodes
                    ON graph_nodes.node_id = CASE
                        WHEN graph_edges.source_node_id = ? THEN graph_edges.target_node_id
                        ELSE graph_edges.source_node_id
                    END
                WHERE graph_edges.doc_id = ?
                    AND (graph_edges.source_node_id = ? OR graph_edges.target_node_id = ?)
                ORDER BY graph_edges.kind ASC, graph_nodes.order_index ASC
                LIMIT ?
                """,
                (node_id, doc_id, node_id, node_id, limit),
            ).fetchall()
        neighbors: list[dict[str, Any]] = []
        for row in rows:
            direction = "outgoing" if row["source_node_id"] == node_id else "incoming"
            neighbors.append(
                {
                    "edge_kind": row["edge_kind"],
                    "direction": direction,
                    "node": {
                        "node_id": row["node_id"],
                        "doc_id": row["doc_id"],
                        "kind": row["kind"],
                        "stable_ref": row["stable_ref"],
                        "title": row["title"],
                        "text": row["text"],
                        "locator": json.loads(row["locator_json"]),
                        "metadata": json.loads(row["metadata_json"]),
                        "order_index": row["order_index"],
                    },
                    "edge_metadata": json.loads(row["edge_metadata_json"]),
                }
            )
        return neighbors

    def list_diagnostics(
        self, *, doc_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM diagnostics
                WHERE doc_id = ?
                ORDER BY
                    CASE severity
                        WHEN 'error' THEN 0
                        WHEN 'warning' THEN 1
                        ELSE 2
                    END,
                    created_at ASC
                LIMIT ?
                """,
                (doc_id, limit),
            ).fetchall()
        return [
            {
                "diagnostic_id": row["diagnostic_id"],
                "doc_id": row["doc_id"],
                "severity": row["severity"],
                "code": row["code"],
                "message": row["message"],
                "node_id": row["node_id"],
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def validate_document_graph(self, doc_id: str) -> list[dict[str, Any]]:
        """Return structural graph issues that should be visible before finalizing."""

        issues: list[dict[str, Any]] = []
        with self._conn() as conn:
            doc_row = conn.execute(
                "SELECT doc_id FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if doc_row is None:
                return [
                    {
                        "severity": "error",
                        "code": "DOCUMENT_RECORD_MISSING",
                        "message": "Document record is missing from the sidecar.",
                        "metadata": {"doc_id": doc_id},
                    }
                ]

            document_nodes = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM graph_nodes
                    WHERE doc_id = ? AND kind = 'document'
                    """,
                    (doc_id,),
                ).fetchone()[0]
            )
            if document_nodes != 1:
                issues.append(
                    {
                        "severity": "error",
                        "code": "DOCUMENT_NODE_COUNT_INVALID",
                        "message": "Document graph must contain exactly one document node.",
                        "metadata": {
                            "doc_id": doc_id,
                            "document_nodes_count": document_nodes,
                        },
                    }
                )

            count_pairs = [
                ("chunks", "chunk", "CHUNK_NODE_COUNT_MISMATCH"),
                ("formulas", "formula", "FORMULA_NODE_COUNT_MISMATCH"),
                ("images", "image", "IMAGE_NODE_COUNT_MISMATCH"),
                ("pdf_tables", "table", "TABLE_NODE_COUNT_MISMATCH"),
                ("pdf_figures", "figure", "FIGURE_NODE_COUNT_MISMATCH"),
            ]
            for table_name, node_kind, code in count_pairs:
                record_count = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE doc_id = ?",
                        (doc_id,),
                    ).fetchone()[0]
                )
                node_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM graph_nodes
                        WHERE doc_id = ? AND kind = ?
                        """,
                        (doc_id, node_kind),
                    ).fetchone()[0]
                )
                if record_count != node_count:
                    issues.append(
                        {
                            "severity": "error",
                            "code": code,
                            "message": (
                                "Graph node count does not match persisted "
                                f"{table_name} records."
                            ),
                            "metadata": {
                                "doc_id": doc_id,
                                "records_count": record_count,
                                "nodes_count": node_count,
                                "node_kind": node_kind,
                            },
                        }
                    )

            dangling_edge_rows = conn.execute(
                """
                SELECT edge.edge_id, edge.source_node_id, edge.target_node_id, edge.kind
                FROM graph_edges AS edge
                LEFT JOIN graph_nodes AS source
                    ON source.node_id = edge.source_node_id
                LEFT JOIN graph_nodes AS target
                    ON target.node_id = edge.target_node_id
                WHERE edge.doc_id = ?
                    AND (source.node_id IS NULL OR target.node_id IS NULL)
                ORDER BY edge.edge_id
                LIMIT 50
                """,
                (doc_id,),
            ).fetchall()
            for row in dangling_edge_rows:
                issues.append(
                    {
                        "severity": "error",
                        "code": "GRAPH_EDGE_ENDPOINT_MISSING",
                        "message": "Graph edge references a missing endpoint node.",
                        "metadata": {
                            "doc_id": doc_id,
                            "edge_id": row["edge_id"],
                            "source_node_id": row["source_node_id"],
                            "target_node_id": row["target_node_id"],
                            "edge_kind": row["kind"],
                        },
                    }
                )

            formula_chunk_rows = conn.execute(
                """
                SELECT formula_id, chunk_id
                FROM formulas
                WHERE doc_id = ?
                    AND chunk_id IS NOT NULL
                    AND chunk_id NOT IN (
                        SELECT chunk_id FROM chunks WHERE doc_id = ?
                    )
                ORDER BY formula_id
                LIMIT 50
                """,
                (doc_id, doc_id),
            ).fetchall()
            for row in formula_chunk_rows:
                issues.append(
                    {
                        "severity": "warning",
                        "code": "FORMULA_CHUNK_MISSING",
                        "message": "Formula points to a chunk that is not present.",
                        "metadata": {
                            "doc_id": doc_id,
                            "formula_id": row["formula_id"],
                            "chunk_id": row["chunk_id"],
                        },
                    }
                )

            chunk_locator_rows = conn.execute(
                """
                SELECT chunk_id, locator_json
                FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_id
                LIMIT 500
                """,
                (doc_id,),
            ).fetchall()
            for row in chunk_locator_rows:
                try:
                    locator = json.loads(row["locator_json"])
                except json.JSONDecodeError:
                    issues.append(
                        {
                            "severity": "error",
                            "code": "CHUNK_LOCATOR_INVALID_JSON",
                            "message": "Chunk locator is not valid JSON.",
                            "metadata": {
                                "doc_id": doc_id,
                                "chunk_id": row["chunk_id"],
                            },
                        }
                    )
                    continue
                if (
                    locator.get("doc_id") != doc_id
                    or locator.get("chunk_id") != row["chunk_id"]
                ):
                    issues.append(
                        {
                            "severity": "error",
                            "code": "CHUNK_LOCATOR_MISMATCH",
                            "message": "Chunk locator does not match its persisted row.",
                            "metadata": {
                                "doc_id": doc_id,
                                "chunk_id": row["chunk_id"],
                                "locator_doc_id": locator.get("doc_id"),
                                "locator_chunk_id": locator.get("chunk_id"),
                            },
                        }
                    )

            graph_locator_rows = conn.execute(
                """
                SELECT node_id, kind, locator_json
                FROM graph_nodes
                WHERE doc_id = ?
                ORDER BY node_id
                LIMIT 1000
                """,
                (doc_id,),
            ).fetchall()
            for row in graph_locator_rows:
                try:
                    locator = json.loads(row["locator_json"])
                except json.JSONDecodeError:
                    issues.append(
                        {
                            "severity": "error",
                            "code": "GRAPH_NODE_LOCATOR_INVALID_JSON",
                            "message": "Graph node locator is not valid JSON.",
                            "metadata": {
                                "doc_id": doc_id,
                                "node_id": row["node_id"],
                                "kind": row["kind"],
                            },
                        }
                    )
                    continue
                locator_doc_id = locator.get("doc_id")
                if locator_doc_id is not None and locator_doc_id != doc_id:
                    issues.append(
                        {
                            "severity": "error",
                            "code": "GRAPH_NODE_LOCATOR_DOC_MISMATCH",
                            "message": "Graph node locator points at a different document.",
                            "metadata": {
                                "doc_id": doc_id,
                                "node_id": row["node_id"],
                                "kind": row["kind"],
                                "locator_doc_id": locator_doc_id,
                            },
                        }
                    )

            artifact_rows = conn.execute(
                """
                SELECT artifact_id, kind, file_path
                FROM artifacts
                WHERE doc_id = ?
                ORDER BY artifact_id
                LIMIT 200
                """,
                (doc_id,),
            ).fetchall()
            for row in artifact_rows:
                if not Path(str(row["file_path"])).exists():
                    issues.append(
                        {
                            "severity": "warning",
                            "code": "ARTIFACT_FILE_MISSING",
                            "message": "Artifact file is missing on disk.",
                            "metadata": {
                                "doc_id": doc_id,
                                "artifact_id": row["artifact_id"],
                                "kind": row["kind"],
                                "file_path": row["file_path"],
                            },
                        }
                    )

        return issues

    def upsert_formula_evidence_artifact(
        self,
        *,
        doc_id: str,
        formula: FormulaRecord,
        file_path: str,
        evidence_type: str,
        width: int | None,
        height: int | None,
        page: int | None,
        bbox: list[float] | None,
    ) -> str | None:
        """Register an on-demand formula render as an addressable graph artifact."""

        formula_node_id = self._graph_node_id(
            doc_id,
            "formula",
            f"formula:{formula.formula_id}",
        )
        artifact_id = f"{doc_id}:formula_evidence:{formula.formula_id}"
        artifact_node_id = self._graph_node_id(
            doc_id,
            "artifact",
            f"artifact:{artifact_id}",
        )
        now = datetime.now(UTC).isoformat()
        locator = {
            "doc_id": doc_id,
            "artifact_id": artifact_id,
            "formula_id": formula.formula_id,
            "section_path": formula.section_path,
            "page": page,
            "bbox": bbox,
        }
        metadata = {
            "kind": "formula_evidence",
            "source_ref": formula.formula_id,
            "file_path": file_path,
            "media_type": "image/png",
            "width": width,
            "height": height,
            "evidence_type": evidence_type,
        }
        stable_ref = f"artifact:{artifact_id}"
        source_id = self._graph_source_id("artifact", stable_ref)
        with self._conn() as conn:
            formula_node = conn.execute(
                """
                SELECT node_id
                FROM graph_nodes
                WHERE doc_id = ? AND node_id = ? AND kind = 'formula'
                """,
                (doc_id, formula_node_id),
            ).fetchone()
            if formula_node is None:
                return None
            conn.execute(
                """
                INSERT INTO graph_nodes (
                    node_id, doc_id, kind, stable_ref, title, text, locator_json,
                    metadata_json, order_index, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    title = excluded.title,
                    text = excluded.text,
                    locator_json = excluded.locator_json,
                    metadata_json = excluded.metadata_json,
                    order_index = excluded.order_index,
                    created_at = excluded.created_at
                """,
                (
                    artifact_node_id,
                    doc_id,
                    "artifact",
                    stable_ref,
                    f"Formula evidence {formula.formula_id}",
                    Path(file_path).name,
                    json.dumps(locator),
                    json.dumps(metadata),
                    page or formula.page or 0,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, doc_id, kind, source_node_id, source_ref,
                    file_path, media_type, width, height, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    source_node_id = excluded.source_node_id,
                    source_ref = excluded.source_ref,
                    file_path = excluded.file_path,
                    media_type = excluded.media_type,
                    width = excluded.width,
                    height = excluded.height,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (
                    artifact_id,
                    doc_id,
                    "formula_evidence",
                    formula_node_id,
                    formula.formula_id,
                    file_path,
                    "image/png",
                    width,
                    height,
                    json.dumps({"evidence_type": evidence_type}),
                    now,
                ),
            )
            for source_node_id, target_node_id, kind in (
                (artifact_node_id, formula_node_id, "renders_from"),
                (formula_node_id, artifact_node_id, "contains"),
            ):
                conn.execute(
                    """
                    INSERT INTO graph_edges (
                        edge_id, doc_id, source_node_id, target_node_id, kind,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(edge_id) DO UPDATE SET
                        metadata_json = excluded.metadata_json,
                        created_at = excluded.created_at
                    """,
                    (
                        self._graph_edge_id(
                            doc_id,
                            source_node_id,
                            target_node_id,
                            kind,
                        ),
                        doc_id,
                        source_node_id,
                        target_node_id,
                        kind,
                        json.dumps({"source": "formula_read"}),
                        now,
                    ),
                )
            conn.execute(
                """
                DELETE FROM local_search_fts
                WHERE doc_id = ? AND source_type = ? AND source_id = ?
                """,
                (doc_id, "artifact", source_id),
            )
            conn.execute(
                """
                INSERT INTO local_search_fts (
                    doc_id, source_type, source_id, chunk_id,
                    section_path_json, order_index, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    "artifact",
                    source_id,
                    None,
                    json.dumps(formula.section_path),
                    str(page or formula.page or 0),
                    "\n".join(
                        [
                            "formula evidence",
                            formula.formula_id,
                            formula.latex,
                            file_path,
                            evidence_type,
                        ]
                    ),
                ),
            )
        return artifact_node_id

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

            changed = (
                row["sha256"] != doc.sha256
                or float(row["mtime"]) != float(doc.mtime)
                or row["profile"] != doc.profile
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
                    SET sha256 = ?, mtime = ?, profile = ?, status = ?, updated_at = ?
                    WHERE path = ?
                    """,
                    (
                        doc.sha256,
                        doc.mtime,
                        doc.profile,
                        DocumentStatus.DISCOVERED,
                        now,
                        doc.path,
                    ),
                )
            return "updated"

    def update_document_source_metadata(
        self,
        *,
        doc_id: str,
        sha256: str,
        mtime: float,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE documents
                SET sha256 = ?, mtime = ?, updated_at = ?
                WHERE doc_id = ?
                """,
                (sha256, mtime, datetime.now(UTC).isoformat(), doc_id),
            )

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
                    started_at, finished_at, owner_id, claimed_at, heartbeat_at,
                    lease_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    job.owner_id,
                    job.claimed_at,
                    job.heartbeat_at,
                    job.lease_expires_at,
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

    def list_all_ingest_jobs(
        self,
        *,
        statuses: list[IngestJobStatus] | None = None,
        limit: int = 1000,
    ) -> list[IngestJobRecord]:
        capped_limit = max(1, min(limit, 10000))
        values: list[Any] = []
        where_clause = ""
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            where_clause = f"WHERE status IN ({placeholders})"
            values.extend(statuses)
        values.append(capped_limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM ingest_jobs
                {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                values,
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

    def get_next_queued_ingest_job(self) -> IngestJobRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM ingest_jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (IngestJobStatus.QUEUED,),
            ).fetchone()
            return self._row_to_ingest_job(row) if row else None

    def claim_ingest_job(
        self,
        job_id: str,
        *,
        owner_id: str,
        lease_expires_at: str,
        now: str | None = None,
    ) -> IngestJobRecord | None:
        claim_time = now or datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                UPDATE ingest_jobs
                SET status = ?,
                    stage = ?,
                    owner_id = ?,
                    claimed_at = COALESCE(claimed_at, ?),
                    heartbeat_at = ?,
                    lease_expires_at = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (
                    IngestJobStatus.RUNNING,
                    IngestStage.PARSE,
                    owner_id,
                    claim_time,
                    claim_time,
                    lease_expires_at,
                    claim_time,
                    claim_time,
                    job_id,
                    IngestJobStatus.QUEUED,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            return self._row_to_ingest_job(row) if row else None

    def fail_expired_running_ingest_jobs(
        self,
        *,
        now: str | None = None,
    ) -> int:
        finished_at = now or datetime.now(UTC).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """
                UPDATE ingest_jobs
                SET status = ?,
                    stage = ?,
                    message = ?,
                    updated_at = ?,
                    finished_at = ?
                WHERE status = ?
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (
                    IngestJobStatus.FAILED,
                    IngestStage.FAILED,
                    "Ingest job lease expired before completion.",
                    finished_at,
                    finished_at,
                    IngestJobStatus.RUNNING,
                    finished_at,
                ),
            )
            return cursor.rowcount

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
        owner_id: str | None = None,
        heartbeat_at: str | None = None,
        lease_expires_at: str | None = None,
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
        if owner_id is not None:
            assignments.append("owner_id = ?")
            values.append(owner_id)
        if heartbeat_at is not None:
            assignments.append("heartbeat_at = ?")
            values.append(heartbeat_at)
        if lease_expires_at is not None:
            assignments.append("lease_expires_at = ?")
            values.append(lease_expires_at)
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
            conn.execute(
                """
                DELETE FROM local_search_fts
                WHERE doc_id = ? AND source_type IN (?, ?)
                """,
                (doc_id, "document", "outline_node"),
            )
            conn.executemany(
                """
                INSERT INTO local_search_fts (
                    doc_id, source_type, source_id, chunk_id,
                    section_path_json, order_index, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                self._outline_search_rows(doc_id=doc_id, title=title, outline=outline),
            )
        self.rebuild_document_graph(doc_id)

    def replace_chunks(
        self,
        doc_id: str,
        chunks: list[ChunkRecord],
        *,
        rebuild_graph: bool = True,
    ) -> None:
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
        if rebuild_graph:
            self.rebuild_document_graph(doc_id)

    def replace_formulas(
        self,
        doc_id: str,
        formulas: list[FormulaRecord],
        *,
        rebuild_graph: bool = True,
    ) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM formulas WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "DELETE FROM local_search_fts WHERE doc_id = ? AND source_type = ?",
                (doc_id, "formula"),
            )
            if formulas:
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
                            json.dumps(formula.bbox)
                            if formula.bbox is not None
                            else None,
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
        if rebuild_graph:
            self.rebuild_document_graph(doc_id)

    def replace_images(
        self,
        doc_id: str,
        images: list[ImageRecord],
        *,
        rebuild_graph: bool = True,
    ) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM images WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "DELETE FROM local_search_fts WHERE doc_id = ? AND source_type = ?",
                (doc_id, "image"),
            )
            if images:
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
                conn.executemany(
                    """
                    INSERT INTO local_search_fts (
                        doc_id, source_type, source_id, chunk_id,
                        section_path_json, order_index, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            image.doc_id,
                            "image",
                            image.image_id,
                            None,
                            json.dumps(image.section_path),
                            str(image.order_index),
                            "\n".join(
                                [
                                    " / ".join(image.section_path),
                                    image.caption or "",
                                    image.alt or "",
                                    image.href or "",
                                    image.media_type or "",
                                ]
                            ),
                        )
                        for image in images
                    ],
                )
        if rebuild_graph:
            self.rebuild_document_graph(doc_id)

    def replace_pdf_tables(
        self,
        doc_id: str,
        tables: list[PdfTableRecord],
        *,
        rebuild_graph: bool = True,
    ) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM pdf_tables WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "DELETE FROM local_search_fts WHERE doc_id = ? AND source_type = ?",
                (doc_id, "pdf_table"),
            )
            if tables:
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
        if rebuild_graph:
            self.rebuild_document_graph(doc_id)

    def replace_pdf_figures(
        self,
        doc_id: str,
        figures: list[PdfFigureRecord],
        *,
        rebuild_graph: bool = True,
    ) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM pdf_figures WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "DELETE FROM local_search_fts WHERE doc_id = ? AND source_type = ?",
                (doc_id, "pdf_figure"),
            )
            if figures:
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
                            json.dumps(figure.bbox)
                            if figure.bbox is not None
                            else None,
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
                conn.executemany(
                    """
                    INSERT INTO local_search_fts (
                        doc_id, source_type, source_id, chunk_id,
                        section_path_json, order_index, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            figure.doc_id,
                            "pdf_figure",
                            figure.figure_id,
                            None,
                            json.dumps(figure.section_path),
                            str(figure.order_index),
                            "\n".join(
                                [
                                    " / ".join(figure.section_path),
                                    figure.caption or "",
                                    figure.kind,
                                    str(figure.page or ""),
                                ]
                            ),
                        )
                        for figure in figures
                    ],
                )
        if rebuild_graph:
            self.rebuild_document_graph(doc_id)

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
        query_tokens = [
            item.lower() for item in self._FTS_TOKEN_RE.findall(query) if item.strip()
        ]

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
                    local_search_fts.text AS search_text,
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
            cooccurrence_score, cooccurrence = self._token_cooccurrence_score(
                query_label=self._normalize_search_label(query),
                texts=[
                    " / ".join(str(item) for item in section_path),
                    str(row["search_text"] or ""),
                ],
            )
            hit = {
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"] or row["source_id"],
                "source_type": source_type,
                "source_id": row["source_id"],
                "locator": locator,
                "section_path": section_path,
                "snippet": row["snippet"],
                "local_rank": index,
                "local_score": -float(row["rank_score"]) + cooccurrence_score,
                "retrieval_source": "sqlite_fts",
                "retrieval_sources": [{"source": "sqlite_fts5", "rank": index}],
            }
            if cooccurrence:
                hit["why_included"] = {
                    "reason": "sqlite_fts5+token_cooccurrence",
                    "token_cooccurrence": cooccurrence,
                }
            if source_type == "formula":
                hit["read_hint"] = (
                    "Use pdf_read_formula or pdf_read_formula with source_id."
                )
            elif source_type == "pdf_table":
                hit["read_hint"] = "Use pdf_read_table with source_id."
            hits.append(hit)

        structured_hits = self._structured_graph_search(
            query=query,
            query_tokens=query_tokens,
            doc_ids=doc_ids,
            top_k=top_k,
        )
        by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for hit in [*hits, *structured_hits]:
            key = (
                str(hit.get("doc_id") or ""),
                str(hit.get("source_type") or ""),
                str(hit.get("source_id") or hit.get("chunk_id") or ""),
            )
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = hit
                continue

            existing_sources = {
                str(source.get("source"))
                for source in existing.get("retrieval_sources", [])
                if isinstance(source, dict)
            }
            for source in hit.get("retrieval_sources", []):
                if not isinstance(source, dict):
                    continue
                source_name = str(source.get("source"))
                if source_name in existing_sources:
                    continue
                existing.setdefault("retrieval_sources", []).append(source)
                existing_sources.add(source_name)

            existing_score = float(existing.get("local_score", 0))
            hit_score = float(hit.get("local_score", 0))
            existing["local_score"] = max(existing_score, hit_score)
            if (
                "why_included" in hit
                and hit_score > existing_score
                and "why_included" in existing
            ):
                existing.setdefault("additional_relevance", []).append(
                    existing["why_included"]
                )
                existing["why_included"] = hit["why_included"]
            elif "why_included" not in existing and "why_included" in hit:
                existing["why_included"] = hit["why_included"]
            elif "why_included" in hit:
                existing.setdefault("additional_relevance", []).append(
                    hit["why_included"]
                )
        merged = sorted(
            by_key.values(),
            key=lambda item: (
                float(item.get("local_score", 0)),
                -int(item.get("local_rank", 0)),
            ),
            reverse=True,
        )
        return merged[:top_k]

    @staticmethod
    def _graph_source_type(kind: str) -> str:
        if kind == "table":
            return "pdf_table"
        if kind == "figure":
            return "pdf_figure"
        return kind

    @staticmethod
    def _graph_source_id(kind: str, stable_ref: str) -> str:
        prefixes = {
            "chunk": "chunk:",
            "formula": "formula:",
            "image": "image:",
            "table": "table:",
            "figure": "figure:",
            "outline_node": "outline:",
            "page": "page:",
            "reference": "reference:",
            "citation": "citation:",
            "artifact": "artifact:",
        }
        prefix = prefixes.get(kind)
        if prefix and stable_ref.startswith(prefix):
            return stable_ref[len(prefix) :]
        return stable_ref

    @classmethod
    def _normalize_search_label(cls, value: str) -> str:
        lowered = value.lower().strip()
        lowered = cls._NON_ALNUM_RE.sub(" ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip()

    @classmethod
    def _query_intent_kinds(cls, normalized_query: str) -> set[str]:
        tokens = set(normalized_query.split())
        joined = normalized_query.replace(" ", "")
        intents: set[str] = set()
        if tokens & cls._FORMULA_QUERY_TERMS or "公式" in joined:
            intents.add("formula")
        if tokens & cls._FIGURE_QUERY_TERMS or any(
            term in joined for term in ("图片", "图表", "示意图")
        ):
            intents.update({"figure", "image", "artifact"})
        if tokens & cls._TABLE_QUERY_TERMS or "表格" in joined:
            intents.add("table")
        if tokens & cls._REFERENCE_QUERY_TERMS or any(
            term in joined for term in ("引用", "参考文献")
        ):
            intents.update({"citation", "reference"})
        return intents

    @classmethod
    def _is_front_matter_hit(cls, title: str, section_path: list[Any]) -> bool:
        haystack = cls._normalize_search_label(
            " ".join([title, *[str(item) for item in section_path]])
        )
        return any(term in haystack.split() for term in cls._FRONT_MATTER_TERMS)

    @classmethod
    def _token_cooccurrence_score(
        cls,
        *,
        query_label: str,
        texts: list[str],
    ) -> tuple[float, dict[str, Any]]:
        query_tokens = list(dict.fromkeys(query_label.split()))
        if not query_tokens:
            return 0.0, {}

        haystack_label = cls._normalize_search_label(" ".join(texts))
        haystack_tokens = haystack_label.split()
        if not haystack_tokens:
            return 0.0, {}

        haystack_token_set = set(haystack_tokens)
        matched_tokens = [
            token
            for token in query_tokens
            if token in haystack_token_set
            or (len(token) >= 4 and token in haystack_label)
        ]
        if not matched_tokens:
            return 0.0, {}

        coverage = len(matched_tokens) / len(query_tokens)
        score = coverage * 18.0
        phrase_match = bool(query_label and query_label in haystack_label)
        if phrase_match and len(query_tokens) > 1:
            score += 12.0
        elif coverage == 1.0 and len(query_tokens) > 1:
            score += 8.0

        token_positions: list[int] = []
        for token in query_tokens:
            try:
                token_positions.append(haystack_tokens.index(token))
            except ValueError:
                continue
        span = None
        if len(token_positions) >= 2:
            span = max(token_positions) - min(token_positions) + 1
            if span <= max(len(query_tokens) * 3, 8):
                score += 6.0

        return score, {
            "matched_tokens": matched_tokens,
            "query_tokens": query_tokens,
            "coverage": round(coverage, 4),
            "phrase_match": phrase_match,
            "token_span": span,
            "score_boost": round(score, 4),
        }

    @classmethod
    def _structured_match_score(
        cls,
        *,
        query: str,
        query_label: str,
        query_intent_kinds: set[str],
        kind: str,
        stable_ref: str,
        title: str,
        text: str,
        section_path: list[Any],
    ) -> tuple[float, str, dict[str, Any]] | None:
        stable_label = cls._normalize_search_label(stable_ref)
        title_label = cls._normalize_search_label(title)
        text_label = cls._normalize_search_label(text[:500])
        section_label = cls._normalize_search_label(
            " ".join(str(item) for item in section_path)
        )
        labels = [
            item
            for item in (stable_label, title_label, text_label, section_label)
            if item
        ]
        if not labels:
            return None

        cooccurrence_score, cooccurrence = cls._token_cooccurrence_score(
            query_label=query_label,
            texts=[
                stable_ref,
                title,
                text,
                " ".join(str(item) for item in section_path),
            ],
        )
        stable_ref_lower = stable_ref.lower()
        title_lower = title.lower()
        query_lower = query.strip().lower()
        reason = "fuzzy_graph_match"
        score = 0.0
        if stable_ref_lower == query_lower or title_lower == query_lower:
            score = 140.0
            reason = "exact_id_or_title_match"
        elif stable_ref_lower.startswith(query_lower) or title_lower.startswith(
            query_lower
        ):
            score = 110.0
            reason = "prefix_id_or_title_match"
        elif any(query_label and query_label in label for label in labels):
            score = 85.0
            reason = "contains_title_or_text_match"
        else:
            query_tokens = set(query_label.split())
            best_overlap = 0.0
            for label in labels:
                label_tokens = set(label.split())
                if not label_tokens or not query_tokens:
                    continue
                overlap = len(query_tokens & label_tokens) / len(label_tokens)
                best_overlap = max(best_overlap, overlap)
            if best_overlap >= 0.66:
                score = 78.0 + best_overlap * 10.0
                reason = "token_overlap_title_match"
            else:
                best_ratio = max(
                    SequenceMatcher(None, query_label, label).ratio()
                    for label in labels
                    if query_label and label
                )
                if best_ratio < 0.55:
                    if cooccurrence.get("coverage", 0.0) < 0.66:
                        return None
                    score = 70.0
                    reason = "token_cooccurrence_graph_match"
                else:
                    score = 60.0 + best_ratio * 20.0
                    reason = "fuzzy_title_match"

        if cooccurrence_score > 0:
            score += cooccurrence_score
            reason = f"{reason}+token_cooccurrence"

        if kind in query_intent_kinds:
            score += 35.0
            reason = f"{reason}+query_intent_{kind}"
        if kind == "outline_node":
            score += 12.0
        elif kind == "document":
            score += 8.0
        if cls._is_front_matter_hit(title, section_path) and not (
            query_intent_kinds & {"citation", "reference"}
        ):
            score -= 25.0
            reason = f"{reason}+front_matter_deprioritized"
        return score, reason, cooccurrence

    def _structured_graph_search(
        self,
        *,
        query: str,
        query_tokens: list[str],
        doc_ids: list[str] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []
        query_label = self._normalize_search_label(query)
        if not query_label:
            return []
        query_intent_kinds = self._query_intent_kinds(query_label)

        like_terms = [normalized_query, *query_tokens[:8]]
        like_terms = list(dict.fromkeys(term for term in like_terms if term))
        conditions: list[str] = []
        values: list[Any] = []
        for term in like_terms:
            pattern = f"%{term}%"
            conditions.append(
                """
                (
                    lower(COALESCE(title, '')) LIKE ?
                    OR lower(stable_ref) LIKE ?
                    OR lower(COALESCE(text, '')) LIKE ?
                )
                """
            )
            values.extend([pattern, pattern, pattern])

        doc_filter = ""
        doc_values: list[Any] = []
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            doc_filter = f" AND doc_id IN ({placeholders})"
            doc_values.extend(doc_ids)
        limit = max(top_k * 8, 80)

        with self._conn() as conn:
            like_rows = conn.execute(
                f"""
                SELECT * FROM graph_nodes
                WHERE ({" OR ".join(conditions)}){doc_filter}
                LIMIT ?
                """,
                [*values, *doc_values, limit],
            ).fetchall()
            fuzzy_rows = conn.execute(
                f"""
                SELECT * FROM graph_nodes
                WHERE kind IN (
                    'document', 'outline_node', 'page', 'formula', 'image',
                    'figure', 'table', 'citation', 'reference', 'artifact'
                ){doc_filter}
                ORDER BY kind ASC, order_index ASC
                LIMIT ?
                """,
                [*doc_values, limit],
            ).fetchall()

        hits: list[dict[str, Any]] = []
        rows_by_id = {row["node_id"]: row for row in [*like_rows, *fuzzy_rows]}
        ranked_rows: list[tuple[float, str, dict[str, Any], sqlite3.Row]] = []
        for row in rows_by_id.values():
            kind = str(row["kind"])
            stable_ref = str(row["stable_ref"])
            title = str(row["title"] or "")
            text = str(row["text"] or "")
            locator = json.loads(row["locator_json"])
            section_path = locator.get("section_path") or []
            match = self._structured_match_score(
                query=query,
                query_label=query_label,
                query_intent_kinds=query_intent_kinds,
                kind=kind,
                stable_ref=stable_ref,
                title=title,
                text=text,
                section_path=section_path,
            )
            if match is None:
                continue
            score, reason, cooccurrence = match
            ranked_rows.append((score, reason, cooccurrence, row))

        ranked_rows.sort(
            key=lambda item: (
                item[0],
                -int(item[3]["order_index"] or 0),
                str(item[3]["kind"]),
            ),
            reverse=True,
        )
        for index, (score, reason, cooccurrence, row) in enumerate(
            ranked_rows[:limit], start=1
        ):
            kind = str(row["kind"])
            stable_ref = str(row["stable_ref"])
            title = str(row["title"] or "")
            text = str(row["text"] or "")
            locator = json.loads(row["locator_json"])
            section_path = locator.get("section_path") or []
            source_type = self._graph_source_type(kind)
            source_id = self._graph_source_id(kind, stable_ref)
            hits.append(
                {
                    "doc_id": row["doc_id"],
                    "chunk_id": source_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "locator": locator,
                    "section_path": section_path,
                    "snippet": title or text[:160],
                    "local_rank": index,
                    "local_score": score,
                    "retrieval_source": "sqlite_structured_graph",
                    "retrieval_sources": [
                        {"source": "sqlite_structured_graph", "rank": index}
                    ],
                    "why_included": {
                        "reason": reason,
                        "matched_node_kind": kind,
                        "query_intent_kinds": sorted(query_intent_kinds),
                        "token_cooccurrence": cooccurrence,
                    },
                    "read_hint": "Use the matching *_node tool with source_id or node_id.",
                }
            )
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
            owner_id=row["owner_id"],
            claimed_at=row["claimed_at"],
            heartbeat_at=row["heartbeat_at"],
            lease_expires_at=row["lease_expires_at"],
        )
