"""Core application service for MCP tools."""

from __future__ import annotations

from collections import defaultdict
import json
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.index.vector import QdrantVectorIndex
from mcp_ebook_read.parsers.epub_ebooklib import EbooklibEpubParser
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.parsers.pdf_grobid import GrobidClient
from mcp_ebook_read.render.pdf_images import PdfImageExtractor
from mcp_ebook_read.render.pdf_render import render_pdf_page
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
    Profile,
)
from mcp_ebook_read.store.catalog import CatalogStore

logger = logging.getLogger(__name__)


class AppService:
    """Orchestrates scan, ingest, indexing and read/render operations."""

    def __init__(
        self,
        *,
        data_dir: Path,
        catalog: CatalogStore,
        sidecar_dir_name: str,
        vector_index: QdrantVectorIndex,
        pdf_parser: DoclingPdfParser,
        pdf_image_extractor: PdfImageExtractor | None = None,
        grobid_client: GrobidClient,
        epub_parser: EbooklibEpubParser,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.catalog = catalog
        self.sidecar_dir_name = sidecar_dir_name
        self.vector_index = vector_index
        self.pdf_parser = pdf_parser
        self.pdf_image_extractor = pdf_image_extractor
        self.grobid_client = grobid_client
        self.epub_parser = epub_parser
        self._catalogs: dict[str, CatalogStore] = {
            str(catalog.db_path.resolve()): catalog,
        }
        self._doc_catalog_index: dict[str, str] = {}

    @staticmethod
    def _detect_total_memory_bytes() -> int | None:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and pages > 0:
                return page_size * pages
        except (AttributeError, ValueError, OSError):
            pass

        if sys.platform == "darwin":
            try:
                output = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                ).strip()
                memory_bytes = int(output)
                if memory_bytes > 0:
                    return memory_bytes
            except (subprocess.SubprocessError, ValueError):
                pass

        return None

    @classmethod
    def _auto_formula_batch_size(cls) -> int:
        cpu_count = max(1, os.cpu_count() or 1)
        cpu_limit = min(cpu_count, 32)
        memory_bytes = cls._detect_total_memory_bytes()
        if memory_bytes is None:
            memory_limit = min(cpu_limit, 8)
        else:
            memory_gib = memory_bytes / (1024**3)
            memory_limit = max(1, int(memory_gib // 1.5))
            memory_limit = min(memory_limit, 16)
        return max(1, min(cpu_limit, memory_limit))

    @classmethod
    def _resolve_formula_batch_size(cls) -> int:
        raw_value = os.environ.get("PDF_FORMULA_BATCH_SIZE")
        if raw_value is None or not raw_value.strip():
            return cls._auto_formula_batch_size()

        normalized = raw_value.strip().lower()
        if normalized == "auto":
            return cls._auto_formula_batch_size()

        try:
            batch_size = int(raw_value)
        except ValueError as exc:
            raise AppError(
                ErrorCode.STARTUP_DEPENDENCY_NOT_READY,
                "PDF_FORMULA_BATCH_SIZE must be a positive integer or 'auto'.",
                details={"env": "PDF_FORMULA_BATCH_SIZE", "value": raw_value},
            ) from exc

        if batch_size < 1:
            raise AppError(
                ErrorCode.STARTUP_DEPENDENCY_NOT_READY,
                "PDF_FORMULA_BATCH_SIZE must be >= 1.",
                details={"env": "PDF_FORMULA_BATCH_SIZE", "value": raw_value},
            )
        return batch_size

    @classmethod
    def from_env(cls) -> "AppService":
        bootstrap_data_dir = (Path.cwd() / ".mcp-ebook-read").resolve()
        preflight_errors: list[dict[str, Any]] = []

        def env_bool(name: str, default: bool) -> bool:
            value = os.environ.get(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        qdrant_url = os.environ.get("QDRANT_URL")
        vector_index: QdrantVectorIndex | None = None
        if not qdrant_url:
            preflight_errors.append(
                {
                    "component": "qdrant",
                    "code": ErrorCode.SEARCH_INDEX_NOT_READY,
                    "message": "QDRANT_URL is not configured.",
                    "details": {"required_env": "QDRANT_URL"},
                }
            )
        else:
            vector_index = QdrantVectorIndex.from_env(check_backend_ready=False)

        grobid_client = GrobidClient.from_env()
        if vector_index is not None:
            try:
                vector_index.assert_ready()
            except AppError as exc:
                preflight_errors.append(
                    {
                        "component": "qdrant",
                        "code": exc.code,
                        "message": exc.message,
                        "details": exc.details or None,
                    }
                )

        try:
            grobid_client.assert_available()
        except AppError as exc:
            preflight_errors.append(
                {
                    "component": "grobid",
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details or None,
                }
            )

        try:
            formula_batch_size = cls._resolve_formula_batch_size()
        except AppError as exc:
            preflight_errors.append(
                {
                    "component": "config",
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details or None,
                }
            )
            formula_batch_size = 1

        if preflight_errors:
            raise AppError(
                ErrorCode.STARTUP_DEPENDENCY_NOT_READY,
                "Startup dependency preflight failed. Configure and start required components before running mcp-ebook-read.",
                details={
                    "failed_components": preflight_errors,
                    "required_env": {
                        "QDRANT_URL": "http://127.0.0.1:6333",
                        "GROBID_URL": "http://127.0.0.1:8070",
                        "GROBID_TIMEOUT_SECONDS": "120",
                    },
                    "quick_start": {
                        "qdrant": "docker rm -f qdrant 2>/dev/null || true && docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.16.3",
                        "grobid": "docker rm -f grobid 2>/dev/null || true && docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0",
                    },
                    "setup_reference": "See README.md: One-Command Docker Setup and Run MCP Server",
                },
            )

        assert vector_index is not None
        return cls(
            data_dir=bootstrap_data_dir,
            catalog=CatalogStore(bootstrap_data_dir / "catalog.db"),
            sidecar_dir_name=".mcp-ebook-read",
            vector_index=vector_index,
            pdf_parser=DoclingPdfParser(
                enable_docling_formula_enrichment=env_bool(
                    "DOCLING_FORMULA_ENRICHMENT", True
                ),
                require_formula_engine=env_bool("PDF_FORMULA_REQUIRE_ENGINE", True),
                formula_batch_size=formula_batch_size,
            ),
            pdf_image_extractor=PdfImageExtractor(min_area_ratio=0.01),
            grobid_client=grobid_client,
            epub_parser=EbooklibEpubParser(),
        )

    def _catalog_key(self, catalog: CatalogStore) -> str:
        return str(catalog.db_path.resolve())

    def _get_or_create_catalog(self, sidecar_dir: Path) -> CatalogStore:
        key = str((sidecar_dir / "catalog.db").resolve())
        catalog = self._catalogs.get(key)
        if catalog is not None:
            return catalog
        catalog = CatalogStore(Path(key))
        self._catalogs[key] = catalog
        return catalog

    def _catalog_for_document_path(self, path: str | Path) -> CatalogStore:
        resolved = Path(path).expanduser().resolve()
        return self._get_or_create_catalog(resolved.parent / self.sidecar_dir_name)

    def _catalog_for_doc_id(self, doc_id: str) -> CatalogStore | None:
        cached_key = self._doc_catalog_index.get(doc_id)
        if cached_key:
            cached_catalog = self._catalogs.get(cached_key)
            if cached_catalog is not None and cached_catalog.get_document_by_id(doc_id):
                return cached_catalog

        for key, catalog in self._catalogs.items():
            if catalog.get_document_by_id(doc_id):
                self._doc_catalog_index[doc_id] = key
                return catalog
        return None

    def _bind_doc_catalog(self, doc_id: str, catalog: CatalogStore) -> None:
        self._doc_catalog_index[doc_id] = self._catalog_key(catalog)

    def _doc_workspace_dir(self, doc: DocumentRecord, catalog: CatalogStore) -> Path:
        self._bind_doc_catalog(doc.doc_id, catalog)
        return catalog.db_path.parent / "docs" / doc.doc_id

    def _resolve_doc(
        self, doc_id: str | None, path: str | None
    ) -> tuple[DocumentRecord, CatalogStore]:
        if path:
            catalog = self._catalog_for_document_path(path)
            doc = catalog.get_document_by_path(path)
            if doc:
                self._bind_doc_catalog(doc.doc_id, catalog)
                if doc_id and doc.doc_id != doc_id:
                    raise AppError(
                        ErrorCode.INGEST_DOC_NOT_FOUND,
                        "doc_id does not match the document at path",
                        details={
                            "doc_id": doc_id,
                            "resolved_doc_id": doc.doc_id,
                            "path": str(Path(path).expanduser().resolve()),
                        },
                    )
                return doc, catalog

        if doc_id:
            catalog = self._catalog_for_doc_id(doc_id)
            if catalog is not None:
                doc = catalog.get_document_by_id(doc_id)
                if doc is not None:
                    return doc, catalog

        raise AppError(
            ErrorCode.INGEST_DOC_NOT_FOUND, "Document not found by doc_id or path"
        )

    def _doc_type_from_path(self, path: Path) -> DocumentType:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return DocumentType.PDF
        if suffix == ".epub":
            return DocumentType.EPUB
        raise AppError(
            ErrorCode.INGEST_UNSUPPORTED_TYPE,
            f"Unsupported file extension: {path.suffix}",
        )

    def _compute_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def library_scan(self, root: str, patterns: list[str]) -> dict[str, Any]:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise AppError(
                ErrorCode.SCAN_INVALID_ROOT,
                f"Invalid scan root: {root}",
            )

        found_paths: set[str] = set()
        found_paths_by_catalog: dict[str, set[str]] = defaultdict(set)
        added: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        unchanged_count = 0
        for existing_catalog in self._discover_sidecar_catalogs(str(root_path)):
            found_paths_by_catalog.setdefault(
                self._catalog_key(existing_catalog),
                set(),
            )

        for pattern in patterns:
            for path in root_path.glob(pattern):
                if not path.is_file():
                    continue
                if self.sidecar_dir_name in path.parts:
                    continue
                if path.suffix.lower() not in {".pdf", ".epub"}:
                    continue

                abs_path = str(path.resolve())
                if abs_path in found_paths:
                    continue
                found_paths.add(abs_path)
                catalog = self._catalog_for_document_path(path)
                found_paths_by_catalog[self._catalog_key(catalog)].add(abs_path)

                file_type = self._doc_type_from_path(path)
                sha256 = self._compute_sha256(path)
                doc_id = sha256[:16]
                doc = DocumentRecord(
                    doc_id=doc_id,
                    path=abs_path,
                    type=file_type,
                    sha256=sha256,
                    mtime=path.stat().st_mtime,
                )
                state = catalog.upsert_scanned_document(doc)
                self._bind_doc_catalog(doc_id, catalog)
                payload = {
                    "doc_id": doc_id,
                    "path": abs_path,
                    "sha256": sha256,
                    "mtime": doc.mtime,
                    "type": file_type,
                }
                if state == "added":
                    added.append(payload)
                elif state == "updated":
                    updated.append(payload)
                else:
                    unchanged_count += 1

        removed: list[str] = []
        removed_deleted_count = 0
        maintenance_rows: list[dict[str, Any]] = []
        for catalog_key, paths in found_paths_by_catalog.items():
            catalog = self._catalogs[catalog_key]
            known = set(catalog.list_document_paths_under_root(str(root_path)))
            removed_paths = sorted(list(known - paths))
            if removed_paths:
                for removed_path in removed_paths:
                    removed_doc = catalog.get_document_by_path(removed_path)
                    if removed_doc is not None:
                        self.vector_index.delete_document(removed_doc.doc_id)
                        self._doc_catalog_index.pop(removed_doc.doc_id, None)
                        shutil.rmtree(
                            catalog.db_path.parent / "docs" / removed_doc.doc_id,
                            ignore_errors=True,
                        )
                removed_deleted_count += catalog.delete_documents_by_paths(
                    removed_paths
                )
                removed.extend(removed_paths)
            maintenance_rows.append(
                {
                    "catalog_path": str(catalog.db_path),
                    "db_size_bytes": catalog.db_size_bytes(),
                }
            )

        storage_maintenance = {
            "auto_compaction_enabled": False,
            "catalogs": maintenance_rows,
        }

        return {
            "added": added,
            "updated": updated,
            "unchanged_count": unchanged_count,
            "removed": sorted(set(removed)),
            "removed_deleted_count": removed_deleted_count,
            "storage_maintenance": storage_maintenance,
        }

    def _write_reading_artifact(
        self, workspace_dir: Path, parsed: ParsedDocument
    ) -> None:
        target = workspace_dir / "reading" / "reading.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(parsed.reading_markdown, encoding="utf-8")

    def _persist_extracted_images(
        self,
        *,
        doc_id: str,
        images: list[ExtractedImage],
        target_dir: Path,
    ) -> list[ImageRecord]:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        if not images:
            return []
        target_dir.mkdir(parents=True, exist_ok=True)

        records: list[ImageRecord] = []
        for image in images:
            extension = image.extension if image.extension.startswith(".") else ".bin"
            out_path = target_dir / f"{image.image_id}{extension.lower()}"
            out_path.write_bytes(image.data)
            records.append(
                ImageRecord(
                    image_id=image.image_id,
                    doc_id=doc_id,
                    order_index=image.order_index,
                    section_path=image.section_path,
                    spine_id=image.spine_id,
                    href=image.href,
                    anchor=image.anchor,
                    alt=image.alt,
                    caption=image.caption,
                    media_type=image.media_type,
                    file_path=str(out_path),
                    width=image.width,
                    height=image.height,
                    source=image.source,
                    status=image.status,
                )
            )
        return records

    def _document_ingest(
        self,
        *,
        doc: DocumentRecord,
        catalog: CatalogStore,
        profile: Profile,
        expected_doc_type: DocumentType | None,
        ingest_mode: str,
        force: bool,
    ) -> dict[str, Any]:
        if doc.status == DocumentStatus.READY and not force and doc.profile == profile:
            return {
                "doc_id": doc.doc_id,
                "profile": doc.profile,
                "parser_chain": doc.parser_chain,
                "chunks_count": len(
                    catalog.get_chunks_window(doc.doc_id, 0, 0, 1_000_000)
                ),
                "images_count": len(catalog.list_images(doc.doc_id)),
                "outline_depth": max((node.level for node in doc.outline), default=0),
                "overall_confidence": doc.overall_confidence,
            }

        self._bind_doc_catalog(doc.doc_id, catalog)
        workspace_dir = self._doc_workspace_dir(doc, catalog)
        catalog.set_document_status(
            doc.doc_id, DocumentStatus.INGESTING, profile=profile
        )

        try:
            if expected_doc_type is not None and doc.type != expected_doc_type:
                raise AppError(
                    ErrorCode.INGEST_UNSUPPORTED_TYPE,
                    f"{ingest_mode} only supports {expected_doc_type.value} documents",
                    details={
                        "ingest_mode": ingest_mode,
                        "doc_type": doc.type,
                        "supported_doc_types": [expected_doc_type],
                    },
                )

            if profile == Profile.BOOK:
                if doc.type == DocumentType.PDF:
                    parsed = self.pdf_parser.parse(doc.path, doc.doc_id)
                elif doc.type == DocumentType.EPUB:
                    parsed = self.epub_parser.parse(doc.path, doc.doc_id)
                else:
                    raise AppError(
                        ErrorCode.INGEST_UNSUPPORTED_TYPE,
                        f"book ingest does not support document type: {doc.type}",
                    )
            elif profile == Profile.PAPER:
                if doc.type != DocumentType.PDF:
                    raise AppError(
                        ErrorCode.INGEST_UNSUPPORTED_TYPE,
                        "paper ingest only supports PDF documents",
                        details={
                            "requested_mode": "paper",
                            "doc_type": doc.type,
                            "supported_doc_types": [DocumentType.PDF],
                        },
                    )
                grobid_result = self.grobid_client.parse_fulltext(doc.path)
                parsed = self.pdf_parser.parse(doc.path, doc.doc_id)
                parsed.parser_chain.append("grobid")
                parsed.metadata = {**parsed.metadata, **grobid_result.metadata}
                if grobid_result.outline:
                    parsed.outline = grobid_result.outline
            else:
                raise AppError(
                    ErrorCode.INGEST_UNSUPPORTED_TYPE,
                    f"Unsupported ingest profile: {profile}",
                )

            self._write_reading_artifact(workspace_dir, parsed)
            catalog.replace_chunks(doc.doc_id, parsed.chunks)
            catalog.replace_formulas(doc.doc_id, parsed.formulas)
            image_records: list[ImageRecord] = []
            if doc.type == DocumentType.EPUB:
                image_records = self._persist_extracted_images(
                    doc_id=doc.doc_id,
                    images=parsed.images,
                    target_dir=workspace_dir / "assets" / "epub-images",
                )
            elif doc.type == DocumentType.PDF:
                pdf_image_dir = workspace_dir / "assets" / "pdf-images"
                if pdf_image_dir.exists():
                    shutil.rmtree(pdf_image_dir)
                parsed.metadata = {
                    **parsed.metadata,
                    "pdf_images_extraction_mode": "on_demand",
                }
            catalog.replace_images(doc.doc_id, image_records)
            self.vector_index.rebuild_document(doc.doc_id, parsed.title, parsed.chunks)
            catalog.save_document_parse_output(
                doc_id=doc.doc_id,
                title=parsed.title,
                parser_chain=parsed.parser_chain,
                metadata=parsed.metadata,
                outline=parsed.outline,
                overall_confidence=parsed.overall_confidence,
                status=DocumentStatus.READY,
            )
            return {
                "doc_id": doc.doc_id,
                "profile": profile,
                "parser_chain": parsed.parser_chain,
                "chunks_count": len(parsed.chunks),
                "images_count": len(image_records),
                "outline_depth": max(
                    (node.level for node in parsed.outline), default=0
                ),
                "overall_confidence": parsed.overall_confidence,
            }
        except Exception:
            catalog.set_document_status(
                doc.doc_id, DocumentStatus.FAILED, profile=profile
            )
            raise

    def document_ingest_pdf_book(
        self, *, doc_id: str | None, path: str | None, force: bool
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path)
        return self._document_ingest(
            doc=doc,
            catalog=catalog,
            profile=Profile.BOOK,
            expected_doc_type=DocumentType.PDF,
            ingest_mode="document_ingest_pdf_book",
            force=force,
        )

    def document_ingest_epub_book(
        self, *, doc_id: str | None, path: str | None, force: bool
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path)
        return self._document_ingest(
            doc=doc,
            catalog=catalog,
            profile=Profile.BOOK,
            expected_doc_type=DocumentType.EPUB,
            ingest_mode="document_ingest_epub_book",
            force=force,
        )

    def document_ingest_pdf_paper(
        self, *, doc_id: str | None, path: str | None, force: bool
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path)
        return self._document_ingest(
            doc=doc,
            catalog=catalog,
            profile=Profile.PAPER,
            expected_doc_type=DocumentType.PDF,
            ingest_mode="document_ingest_pdf_paper",
            force=force,
        )

    def _page_ranges_overlap(
        self,
        chunk_range: list[int] | None,
        node_start: int | None,
        node_end: int | None,
    ) -> bool:
        if not chunk_range or len(chunk_range) != 2:
            return False
        if node_start is None or node_end is None:
            return False
        c_start, c_end = chunk_range
        return c_start <= node_end and c_end >= node_start

    def _resolve_outline_node(
        self, doc_id: str, node_id: str
    ) -> tuple[DocumentRecord, OutlineNode, CatalogStore]:
        catalog = self._catalog_for_doc_id(doc_id)
        doc = catalog.get_document_by_id(doc_id) if catalog is not None else None
        if not doc:
            raise AppError(ErrorCode.INGEST_DOC_NOT_FOUND, f"Unknown doc_id: {doc_id}")
        node = next((item for item in doc.outline if item.id == node_id), None)
        if node is None:
            raise AppError(
                ErrorCode.READ_LOCATOR_NOT_FOUND,
                f"Unknown outline node: {node_id}",
                details={
                    "doc_id": doc_id,
                    "node_id": node_id,
                    "hint": "Call get_outline first and use a valid node id.",
                },
            )
        return doc, node, catalog

    def _chunks_for_outline_node(
        self,
        *,
        catalog: CatalogStore,
        doc_id: str,
        node: OutlineNode,
    ) -> list[ChunkRecord]:
        all_chunks = catalog.list_chunks(doc_id)
        if not all_chunks:
            return []

        matched = [
            chunk
            for chunk in all_chunks
            if self._page_ranges_overlap(
                chunk.locator.page_range,
                node.page_start,
                node.page_end,
            )
        ]
        if matched:
            return matched

        node_title = node.title.strip().lower()
        if not node_title:
            return []
        return [
            chunk
            for chunk in all_chunks
            if any(node_title in part.strip().lower() for part in chunk.section_path)
        ]

    def _format_chunk_window(self, chunks: list[ChunkRecord], out_format: str) -> str:
        if out_format not in {"markdown", "text"}:
            raise AppError(
                ErrorCode.READ_LOCATOR_NOT_FOUND,
                f"Unsupported format: {out_format}",
            )
        if out_format == "markdown":
            return "\n\n".join(
                f"## {' / '.join(item.section_path) or 'Section'}\n\n{item.text}"
                for item in chunks
            )
        return "\n\n".join(item.text for item in chunks)

    def search(
        self, query: str, doc_ids: list[str] | None, top_k: int
    ) -> dict[str, Any]:
        hits = self.vector_index.search(query=query, top_k=top_k, doc_ids=doc_ids)
        return {"hits": hits}

    def search_in_outline_node(
        self,
        *,
        doc_id: str,
        node_id: str,
        query: str,
        top_k: int,
    ) -> dict[str, Any]:
        _, node, _ = self._resolve_outline_node(doc_id, node_id)
        expanded_top_k = max(top_k * 5, top_k)
        raw_hits = self.vector_index.search(
            query=query, top_k=expanded_top_k, doc_ids=[doc_id]
        )
        filtered_hits: list[dict[str, Any]] = []
        for hit in raw_hits:
            locator = hit.get("locator") or {}
            page_range = locator.get("page_range")
            if self._page_ranges_overlap(page_range, node.page_start, node.page_end):
                filtered_hits.append(hit)
                if len(filtered_hits) >= top_k:
                    break

        if len(filtered_hits) < top_k:
            node_title = node.title.strip().lower()
            for hit in raw_hits:
                locator = hit.get("locator") or {}
                section_path = locator.get("section_path") or []
                joined = " / ".join(section_path).lower()
                if node_title and node_title in joined and hit not in filtered_hits:
                    filtered_hits.append(hit)
                    if len(filtered_hits) >= top_k:
                        break

        return {
            "node": node.model_dump(),
            "hits": filtered_hits[:top_k],
        }

    def read(
        self, locator: dict[str, Any], before: int, after: int, out_format: str
    ) -> dict[str, Any]:
        parsed_locator = Locator(**locator)
        catalog = self._catalog_for_doc_id(parsed_locator.doc_id)
        if catalog is None:
            raise AppError(
                ErrorCode.READ_LOCATOR_NOT_FOUND,
                "Chunk not found for locator.",
            )
        chunk = catalog.get_chunk(parsed_locator.doc_id, parsed_locator.chunk_id)
        if not chunk:
            raise AppError(
                ErrorCode.READ_LOCATOR_NOT_FOUND,
                "Chunk not found for locator.",
            )

        window = catalog.get_chunks_window(
            doc_id=chunk.doc_id,
            center_order=chunk.order_index,
            before=before,
            after=after,
        )
        content = self._format_chunk_window(window, out_format)

        return {
            "content": content,
            "resolved_locators": [item.locator.model_dump() for item in window],
            "method": chunk.method,
            "confidence": chunk.confidence,
        }

    def read_outline_node(
        self,
        *,
        doc_id: str,
        node_id: str,
        out_format: str,
        max_chunks: int = 120,
    ) -> dict[str, Any]:
        doc, node, catalog = self._resolve_outline_node(doc_id, node_id)
        scoped_chunks = self._chunks_for_outline_node(
            catalog=catalog,
            doc_id=doc_id,
            node=node,
        )
        if not scoped_chunks:
            raise AppError(
                ErrorCode.READ_LOCATOR_NOT_FOUND,
                "No chunks found for outline node.",
                details={
                    "doc_id": doc_id,
                    "node_id": node_id,
                    "node_title": node.title,
                },
            )

        truncated = len(scoped_chunks) > max_chunks
        if truncated:
            scoped_chunks = scoped_chunks[:max_chunks]

        content = self._format_chunk_window(scoped_chunks, out_format)
        return {
            "doc_title": doc.title,
            "node": node.model_dump(),
            "content": content,
            "resolved_locators": [item.locator.model_dump() for item in scoped_chunks],
            "chunks_count": len(scoped_chunks),
            "truncated": truncated,
        }

    def _ensure_epub_doc(self, doc_id: str) -> tuple[DocumentRecord, CatalogStore]:
        catalog = self._catalog_for_doc_id(doc_id)
        doc = catalog.get_document_by_id(doc_id) if catalog is not None else None
        if not doc:
            raise AppError(ErrorCode.INGEST_DOC_NOT_FOUND, f"Unknown doc_id: {doc_id}")
        if doc.type != DocumentType.EPUB:
            raise AppError(
                ErrorCode.INGEST_UNSUPPORTED_TYPE,
                "epub image tools only support EPUB documents",
                details={
                    "doc_id": doc_id,
                    "doc_type": doc.type,
                    "supported_doc_types": [DocumentType.EPUB],
                },
            )
        return doc, catalog

    def _ensure_pdf_doc(self, doc_id: str) -> tuple[DocumentRecord, CatalogStore]:
        catalog = self._catalog_for_doc_id(doc_id)
        doc = catalog.get_document_by_id(doc_id) if catalog is not None else None
        if not doc:
            raise AppError(ErrorCode.INGEST_DOC_NOT_FOUND, f"Unknown doc_id: {doc_id}")
        if doc.type != DocumentType.PDF:
            raise AppError(
                ErrorCode.INGEST_UNSUPPORTED_TYPE,
                "pdf image tools only support PDF documents",
                details={
                    "doc_id": doc_id,
                    "doc_type": doc.type,
                    "supported_doc_types": [DocumentType.PDF],
                },
            )
        return doc, catalog

    def _image_payload(self, image: ImageRecord) -> dict[str, Any]:
        return {
            "image_id": image.image_id,
            "section_path": image.section_path,
            "spine_id": image.spine_id,
            "page": image.page,
            "bbox": image.bbox,
            "href": image.href,
            "anchor": image.anchor,
            "alt": image.alt,
            "caption": image.caption,
            "media_type": image.media_type,
            "image_path": image.file_path,
            "width": image.width,
            "height": image.height,
        }

    def _pdf_images_manifest_path(self, workspace_dir: Path) -> Path:
        return workspace_dir / "assets" / "pdf-images" / ".extracted.json"

    def _ensure_pdf_images_extracted(
        self,
        *,
        doc: DocumentRecord,
        catalog: CatalogStore,
        force: bool = False,
    ) -> list[ImageRecord]:
        workspace_dir = self._doc_workspace_dir(doc, catalog)
        images_dir = workspace_dir / "assets" / "pdf-images"
        manifest_path = self._pdf_images_manifest_path(workspace_dir)

        if force:
            if images_dir.exists():
                shutil.rmtree(images_dir)
            catalog.replace_images(doc.doc_id, [])

        existing_images = catalog.list_images(doc.doc_id)
        if existing_images and not force:
            return existing_images
        if manifest_path.exists() and not force:
            return existing_images
        if self.pdf_image_extractor is None:
            return existing_images

        chunks = catalog.list_chunks(doc.doc_id)
        if not chunks:
            raise AppError(
                ErrorCode.READ_IMAGE_NOT_FOUND,
                "PDF images are unavailable because the document has no parsed chunks.",
                details={
                    "doc_id": doc.doc_id,
                    "hint": "Run document_ingest_pdf_book or document_ingest_pdf_paper first.",
                },
            )

        if images_dir.exists():
            shutil.rmtree(images_dir)

        extracted_images = self.pdf_image_extractor.extract(
            pdf_path=doc.path,
            doc_id=doc.doc_id,
            chunks=chunks,
            out_dir=images_dir,
        )
        catalog.replace_images(doc.doc_id, extracted_images)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "doc_id": doc.doc_id,
                    "images_count": len(extracted_images),
                },
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return extracted_images

    def epub_list_images(
        self,
        *,
        doc_id: str,
        node_id: str | None,
        limit: int,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_epub_doc(doc_id)
        images = catalog.list_images(doc_id)

        node_payload: dict[str, Any] | None = None
        if node_id:
            _, node, _ = self._resolve_outline_node(doc_id, node_id)
            node_payload = node.model_dump()
            node_title = node.title.strip().lower()
            filtered: list[ImageRecord] = []
            for image in images:
                if node.spine_ref and image.spine_id == node.spine_ref:
                    filtered.append(image)
                    continue
                if node_title and any(
                    node_title in part.strip().lower() for part in image.section_path
                ):
                    filtered.append(image)
            images = filtered

        max_items = max(0, min(limit, 500))
        truncated = len(images) > max_items
        if truncated:
            images = images[:max_items]

        return {
            "doc_title": doc.title,
            "node": node_payload,
            "images": [self._image_payload(image) for image in images],
            "images_count": len(images),
            "truncated": truncated,
        }

    def epub_read_image(self, *, doc_id: str, image_id: str) -> dict[str, Any]:
        doc, catalog = self._ensure_epub_doc(doc_id)
        image = catalog.get_image(image_id)
        if image is None or image.doc_id != doc_id:
            raise AppError(
                ErrorCode.READ_IMAGE_NOT_FOUND,
                f"Unknown image_id: {image_id}",
                details={
                    "doc_id": doc_id,
                    "image_id": image_id,
                    "hint": "Call epub_list_images to get valid image ids.",
                },
            )

        image_path = Path(image.file_path)
        if not image_path.exists():
            raise AppError(
                ErrorCode.READ_IMAGE_NOT_FOUND,
                f"Image file does not exist: {image.file_path}",
                details={"doc_id": doc_id, "image_id": image_id},
            )

        chunks = catalog.list_chunks(doc_id)
        matched_chunk: ChunkRecord | None = None
        for chunk in chunks:
            locator = chunk.locator.epub_locator or {}
            if image.spine_id and locator.get("spine_id") != image.spine_id:
                continue
            if image.anchor and locator.get("anchor") == image.anchor:
                matched_chunk = chunk
                break

        if matched_chunk is None:
            for chunk in chunks:
                locator = chunk.locator.epub_locator or {}
                if image.spine_id and locator.get("spine_id") == image.spine_id:
                    matched_chunk = chunk
                    break

        if matched_chunk is None and image.section_path:
            target = image.section_path[-1].strip().lower()
            if target:
                for chunk in chunks:
                    if any(
                        target in part.strip().lower() for part in chunk.section_path
                    ):
                        matched_chunk = chunk
                        break

        return {
            "doc_title": doc.title,
            "image": self._image_payload(image),
            "context": {
                "text": matched_chunk.text,
                "locator": matched_chunk.locator.model_dump(),
            }
            if matched_chunk
            else None,
        }

    def pdf_list_images(
        self,
        *,
        doc_id: str,
        node_id: str | None,
        limit: int,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        self._ensure_pdf_images_extracted(doc=doc, catalog=catalog)
        images = catalog.list_images(doc_id)

        node_payload: dict[str, Any] | None = None
        if node_id:
            _, node, _ = self._resolve_outline_node(doc_id, node_id)
            node_payload = node.model_dump()
            node_title = node.title.strip().lower()
            filtered: list[ImageRecord] = []
            for image in images:
                if image.page is not None and self._page_ranges_overlap(
                    [image.page, image.page], node.page_start, node.page_end
                ):
                    filtered.append(image)
                    continue
                if node_title and any(
                    node_title in part.strip().lower() for part in image.section_path
                ):
                    filtered.append(image)
            images = filtered

        max_items = max(0, min(limit, 500))
        truncated = len(images) > max_items
        if truncated:
            images = images[:max_items]

        return {
            "doc_title": doc.title,
            "node": node_payload,
            "images": [self._image_payload(image) for image in images],
            "images_count": len(images),
            "truncated": truncated,
        }

    def pdf_read_image(self, *, doc_id: str, image_id: str) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        self._ensure_pdf_images_extracted(doc=doc, catalog=catalog)
        image = catalog.get_image(image_id)
        if image is None or image.doc_id != doc_id:
            raise AppError(
                ErrorCode.READ_IMAGE_NOT_FOUND,
                f"Unknown image_id: {image_id}",
                details={
                    "doc_id": doc_id,
                    "image_id": image_id,
                    "hint": "Call pdf_list_images to get valid image ids.",
                },
            )

        image_path = Path(image.file_path)
        if not image_path.exists():
            self._ensure_pdf_images_extracted(doc=doc, catalog=catalog, force=True)
            image = catalog.get_image(image_id)
            if image is None or image.doc_id != doc_id:
                raise AppError(
                    ErrorCode.READ_IMAGE_NOT_FOUND,
                    f"Unknown image_id: {image_id}",
                    details={"doc_id": doc_id, "image_id": image_id},
                )
            image_path = Path(image.file_path)
            if not image_path.exists():
                raise AppError(
                    ErrorCode.READ_IMAGE_NOT_FOUND,
                    f"Image file does not exist: {image.file_path}",
                    details={"doc_id": doc_id, "image_id": image_id},
                )

        chunks = catalog.list_chunks(doc_id)
        matched_chunk: ChunkRecord | None = None
        if image.page is not None:
            for chunk in chunks:
                if self._page_ranges_overlap(
                    chunk.locator.page_range, image.page, image.page
                ):
                    matched_chunk = chunk
                    break

        if matched_chunk is None and image.section_path:
            target = image.section_path[-1].strip().lower()
            if target:
                for chunk in chunks:
                    if any(
                        target in part.strip().lower() for part in chunk.section_path
                    ):
                        matched_chunk = chunk
                        break

        return {
            "doc_title": doc.title,
            "image": self._image_payload(image),
            "context": {
                "text": matched_chunk.text,
                "locator": matched_chunk.locator.model_dump(),
            }
            if matched_chunk
            else None,
        }

    def get_outline(self, doc_id: str) -> dict[str, Any]:
        catalog = self._catalog_for_doc_id(doc_id)
        doc = catalog.get_document_by_id(doc_id) if catalog is not None else None
        if not doc:
            raise AppError(ErrorCode.INGEST_DOC_NOT_FOUND, f"Unknown doc_id: {doc_id}")
        return {
            "title": doc.title,
            "nodes": [node.model_dump() for node in doc.outline],
        }

    def render_pdf_page(self, doc_id: str, page: int, dpi: int) -> dict[str, Any]:
        catalog = self._catalog_for_doc_id(doc_id)
        doc = catalog.get_document_by_id(doc_id) if catalog is not None else None
        if not doc:
            raise AppError(ErrorCode.INGEST_DOC_NOT_FOUND, f"Unknown doc_id: {doc_id}")
        if doc.type != DocumentType.PDF:
            raise AppError(
                ErrorCode.RENDER_PAGE_FAILED,
                "render_pdf_page is only supported for PDF documents",
            )

        assert catalog is not None
        out_path = (
            self._doc_workspace_dir(doc, catalog)
            / "evidence"
            / "pages"
            / f"page_{page:04d}_d{dpi}.png"
        )
        width, height = render_pdf_page(doc.path, out_path, page, dpi)
        return {
            "image_path": str(out_path),
            "width": width,
            "height": height,
            "page": page,
        }

    def _discover_sidecar_catalogs(self, root: str) -> list[CatalogStore]:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise AppError(
                ErrorCode.SCAN_INVALID_ROOT,
                f"Invalid scan root: {root}",
            )

        catalogs: list[CatalogStore] = []
        seen: set[str] = set()
        for sidecar_dir in root_path.rglob(self.sidecar_dir_name):
            if not sidecar_dir.is_dir():
                continue
            db_path = sidecar_dir / "catalog.db"
            if not db_path.exists():
                continue
            key = str(db_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            catalogs.append(self._get_or_create_catalog(sidecar_dir))
        return catalogs

    def storage_list_sidecars(self, *, root: str, limit: int) -> dict[str, Any]:
        catalogs = self._discover_sidecar_catalogs(root)
        max_items = max(1, min(limit, 1000))
        items: list[dict[str, Any]] = []
        total_docs = 0
        for catalog in catalogs:
            docs = catalog.list_documents()
            total_docs += len(docs)
            for doc in docs:
                self._bind_doc_catalog(doc.doc_id, catalog)
            sample_docs = docs[:max_items]
            items.append(
                {
                    "sidecar_path": str(catalog.db_path.parent),
                    "catalog_path": str(catalog.db_path),
                    "db_size_bytes": catalog.db_size_bytes(),
                    "documents_count": len(docs),
                    "documents": [
                        {
                            "doc_id": doc.doc_id,
                            "path": doc.path,
                            "type": doc.type,
                            "status": doc.status,
                            "profile": doc.profile,
                        }
                        for doc in sample_docs
                    ],
                    "documents_truncated": len(docs) > max_items,
                }
            )
        return {
            "root": str(Path(root).expanduser().resolve()),
            "sidecars_count": len(catalogs),
            "documents_count": total_docs,
            "sidecars": sorted(items, key=lambda item: item["sidecar_path"]),
        }

    def storage_delete_document(
        self,
        *,
        doc_id: str | None,
        path: str | None,
        remove_artifacts: bool,
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path)
        deleted_records = catalog.delete_documents_by_paths([doc.path])
        self.vector_index.delete_document(doc.doc_id)
        self._doc_catalog_index.pop(doc.doc_id, None)
        workspace_dir = catalog.db_path.parent / "docs" / doc.doc_id
        artifacts_removed = False
        if remove_artifacts:
            shutil.rmtree(workspace_dir, ignore_errors=True)
            artifacts_removed = not workspace_dir.exists()
        return {
            "doc_id": doc.doc_id,
            "path": doc.path,
            "deleted_records": deleted_records,
            "vector_deleted": True,
            "artifacts_removed": artifacts_removed if remove_artifacts else False,
        }

    def storage_cleanup_sidecars(
        self,
        *,
        root: str,
        remove_missing_documents: bool,
        remove_orphan_artifacts: bool,
        compact_catalog: bool,
    ) -> dict[str, Any]:
        catalogs = self._discover_sidecar_catalogs(root)
        sidecar_rows: list[dict[str, Any]] = []
        removed_paths_total: list[str] = []
        removed_docs_total = 0
        removed_artifacts_total = 0
        reclaimed_bytes_total = 0

        for catalog in catalogs:
            docs = catalog.list_documents()
            for doc in docs:
                self._bind_doc_catalog(doc.doc_id, catalog)
            docs_by_id = {doc.doc_id: doc for doc in docs}
            sidecar_root = catalog.db_path.parent
            removed_paths: list[str] = []
            removed_doc_ids: list[str] = []
            if remove_missing_documents:
                for doc in docs:
                    if not Path(doc.path).exists():
                        removed_paths.append(doc.path)
                        removed_doc_ids.append(doc.doc_id)

            removed_deleted_count = 0
            if removed_paths:
                removed_deleted_count = catalog.delete_documents_by_paths(removed_paths)
                for doc_id_item in removed_doc_ids:
                    self.vector_index.delete_document(doc_id_item)
                    self._doc_catalog_index.pop(doc_id_item, None)
                    shutil.rmtree(
                        sidecar_root / "docs" / doc_id_item,
                        ignore_errors=True,
                    )
                removed_paths_total.extend(removed_paths)
                removed_docs_total += removed_deleted_count

            orphan_artifacts_deleted = 0
            docs_dir = sidecar_root / "docs"
            if remove_orphan_artifacts and docs_dir.exists():
                live_doc_ids = {doc.doc_id for doc in catalog.list_documents()}
                for candidate in docs_dir.iterdir():
                    if not candidate.is_dir():
                        continue
                    if candidate.name in live_doc_ids:
                        continue
                    shutil.rmtree(candidate, ignore_errors=True)
                    orphan_artifacts_deleted += 1
                    removed_artifacts_total += 1

            compact_info: dict[str, Any] = {
                "requested": compact_catalog,
                "performed": False,
                "before_bytes": catalog.db_size_bytes(),
                "after_bytes": catalog.db_size_bytes(),
                "reclaimed_bytes": 0,
            }
            if compact_catalog:
                compact_stats = catalog.compact()
                compact_info = {
                    "requested": True,
                    "performed": True,
                    **compact_stats,
                }
                reclaimed_bytes_total += int(compact_info["reclaimed_bytes"])

            sidecar_rows.append(
                {
                    "sidecar_path": str(sidecar_root),
                    "catalog_path": str(catalog.db_path),
                    "documents_before": len(docs_by_id),
                    "removed_deleted_count": removed_deleted_count,
                    "removed_paths": sorted(removed_paths),
                    "orphan_artifacts_deleted": orphan_artifacts_deleted,
                    "compact": compact_info,
                }
            )

        return {
            "root": str(Path(root).expanduser().resolve()),
            "sidecars_count": len(catalogs),
            "removed_deleted_count": removed_docs_total,
            "removed_paths": sorted(set(removed_paths_total)),
            "orphan_artifacts_deleted": removed_artifacts_total,
            "reclaimed_bytes": reclaimed_bytes_total,
            "sidecars": sorted(sidecar_rows, key=lambda item: item["sidecar_path"]),
        }
