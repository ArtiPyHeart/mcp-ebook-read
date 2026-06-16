"""Core application service for MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import hashlib
import logging
import os
import re
from queue import Empty, Queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp_ebook_read.errors import AppError, ErrorCode, to_error_payload
from mcp_ebook_read.outline import (
    find_outline_node,
    matches_outline_node,
    normalize_section_key,
    normalize_section_path,
    page_ranges_overlap,
    section_path_leaf_matches,
    section_path_prefix_matches,
)
from mcp_ebook_read.parsers.epub_ebooklib import EbooklibEpubParser
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.parsers.pdf_grobid import GrobidClient
from mcp_ebook_read.parsers.pdf_pypdfium2 import Pypdfium2PdfParser
from mcp_ebook_read.render.pdf_images import PdfImageExtractor
from mcp_ebook_read.render.pdf_visuals import (
    DoclingPdfVisualExtractor,
    PdfVisualExtractionResult,
)
from mcp_ebook_read.render.pdf_render import render_pdf_page, render_pdf_region
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    ExtractedImage,
    FormulaRecord,
    ImageRecord,
    IngestJobRecord,
    IngestJobStatus,
    IngestStage,
    Locator,
    OutlineNode,
    ParsedDocument,
    PdfFigureRecord,
    Profile,
    PdfTableRecord,
    PdfParserPerformanceConfig,
)
from mcp_ebook_read.store.catalog import CatalogStore

logger = logging.getLogger(__name__)

_PIPELINE_METADATA_KEY = "mcp_ebook_read_pipeline"
_PIPELINE_SCHEMA_VERSION = 1
_CHUNKER_VERSION = "chunker-v2-outline-aware"
_SEARCH_TEXT_VERSION = "search-text-v2-fts5"
_PDF_FORMULA_PIPELINE_VERSION = "docling-formula-text-provenance-v2"
_PDF_VISUAL_PIPELINE_VERSION = "docling-visuals-eager-v2"
_PDF_DIAGNOSTIC_PIPELINE_VERSION = "pymupdf-diagnostic-lane-v1"
_EPUB_IMAGE_PIPELINE_VERSION = "ebooklib-image-extraction-v1"


@dataclass(frozen=True, slots=True)
class _IngestQueueItem:
    catalog_key: str
    doc_id: str
    job_id: str


class AppService:
    """Orchestrates scan, ingest, indexing and read/render operations."""

    def __init__(
        self,
        *,
        sidecar_dir_name: str,
        default_library_root: str | Path | None = None,
        pdf_parser: DoclingPdfParser,
        pdf_image_extractor: PdfImageExtractor | None = None,
        pdf_visual_extractor: DoclingPdfVisualExtractor | None = None,
        grobid_client: GrobidClient,
        epub_parser: EbooklibEpubParser,
        pdf_parse_timeout_seconds: int = 1800,
        ingest_worker_count: int = 1,
    ) -> None:
        self.sidecar_dir_name = sidecar_dir_name
        self.default_library_root = self._resolve_default_library_root(
            default_library_root
        )
        self.pdf_parser = pdf_parser
        self.pdf_image_extractor = pdf_image_extractor
        self.pdf_visual_extractor = pdf_visual_extractor
        self.grobid_client = grobid_client
        self.epub_parser = epub_parser
        self.pdf_parse_timeout_seconds = pdf_parse_timeout_seconds
        self.ingest_worker_count = max(1, ingest_worker_count)
        self._catalogs: dict[str, CatalogStore] = {}
        self._doc_catalog_index: dict[str, str] = {}
        self._known_roots: set[str] = set()
        self._ingest_queue: Queue[_IngestQueueItem] = Queue()
        self._ingest_lock = threading.Lock()
        self._active_pdf_parse_lock = threading.Lock()
        self._active_pdf_parses = 0
        self._closed = False
        self._ingest_owner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
        self._ingest_workers: list[threading.Thread] = []
        for worker_index in range(self.ingest_worker_count):
            worker = threading.Thread(
                target=self._ingest_worker_loop,
                name=f"mcp-ebook-read-ingest-worker-{worker_index + 1}",
                daemon=True,
            )
            self._ingest_workers.append(worker)
            worker.start()

    _INGEST_STAGE_TOTAL = 6
    _INGEST_STAGE_DONE: dict[IngestStage, int] = {
        IngestStage.QUEUED: 0,
        IngestStage.GROBID: 1,
        IngestStage.PARSE: 2,
        IngestStage.PERSIST: 3,
        IngestStage.INDEX: 4,
        IngestStage.FINALIZE: 5,
        IngestStage.CACHED: 6,
        IngestStage.COMPLETED: 6,
        IngestStage.FAILED: 6,
        IngestStage.CANCELED: 6,
    }

    def close(self) -> None:
        """Release process-scoped resources owned by the service."""

        if self._closed:
            return
        self._closed = True
        self.pdf_parser.close()
        close_visual_extractor = getattr(self.pdf_visual_extractor, "close", None)
        if callable(close_visual_extractor):
            close_visual_extractor()

    @staticmethod
    def _find_project_root(start: Path) -> Path:
        resolved = start.expanduser().resolve()
        if resolved.is_file():
            resolved = resolved.parent
        for candidate in (resolved, *resolved.parents):
            if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
                return candidate
        return resolved

    @classmethod
    def _resolve_default_library_root(cls, root: str | Path | None) -> Path:
        if root is not None:
            return Path(root).expanduser().resolve()
        return cls._find_project_root(Path.cwd())

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
    def _auto_ingest_worker_count(cls) -> int:
        cpu_count = max(1, os.cpu_count() or 1)
        memory_bytes = cls._detect_total_memory_bytes()
        cpu_limit = max(1, min(4, cpu_count // 4))
        if memory_bytes is None:
            memory_limit = 2 if cpu_count >= 8 else 1
        else:
            memory_gib = memory_bytes / (1024**3)
            memory_limit = max(1, min(4, int(memory_gib // 8)))
        return max(1, min(cpu_limit, memory_limit))

    @classmethod
    def _auto_docling_performance_config(cls) -> PdfParserPerformanceConfig:
        cpu_count = max(1, os.cpu_count() or 1)
        memory_bytes = cls._detect_total_memory_bytes()
        num_threads = max(1, min(cpu_count, 16))
        if memory_bytes is None:
            batch_size = max(4, min(num_threads, 8))
        else:
            memory_gib = memory_bytes / (1024**3)
            memory_limit = max(1, int(memory_gib // 2))
            batch_size = max(2, min(num_threads, memory_limit, 16))
        return PdfParserPerformanceConfig(
            num_threads=num_threads,
            device="auto",
            ocr_batch_size=batch_size,
            layout_batch_size=batch_size,
            table_batch_size=batch_size,
        )

    @staticmethod
    def _scale_pdf_performance_for_active_parses(
        config: PdfParserPerformanceConfig,
        *,
        active_pdf_parses: int,
    ) -> PdfParserPerformanceConfig:
        if active_pdf_parses <= 1:
            return config
        return PdfParserPerformanceConfig(
            num_threads=max(1, config.num_threads // active_pdf_parses),
            device=config.device,
            ocr_batch_size=max(1, config.ocr_batch_size // active_pdf_parses),
            layout_batch_size=max(1, config.layout_batch_size // active_pdf_parses),
            table_batch_size=max(1, config.table_batch_size // active_pdf_parses),
        )

    @staticmethod
    def _docling_performance_env_overridden() -> bool:
        return bool(
            os.environ.get("PDF_DOCLING_NUM_THREADS")
            or os.environ.get("PDF_DOCLING_BATCH_SIZE")
        )

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
                ErrorCode.STARTUP_CONFIG_INVALID,
                "PDF_FORMULA_BATCH_SIZE must be a positive integer or 'auto'.",
                details={"env": "PDF_FORMULA_BATCH_SIZE", "value": raw_value},
            ) from exc

        if batch_size < 1:
            raise AppError(
                ErrorCode.STARTUP_CONFIG_INVALID,
                "PDF_FORMULA_BATCH_SIZE must be >= 1.",
                details={"env": "PDF_FORMULA_BATCH_SIZE", "value": raw_value},
            )
        return batch_size

    @classmethod
    def _resolve_positive_int_env(
        cls,
        *,
        env_name: str,
        default: int,
    ) -> int:
        raw_value = os.environ.get(env_name)
        if raw_value is None or not raw_value.strip():
            return default
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise AppError(
                ErrorCode.STARTUP_CONFIG_INVALID,
                f"{env_name} must be a positive integer.",
                details={"env": env_name, "value": raw_value},
            ) from exc
        if value < 1:
            raise AppError(
                ErrorCode.STARTUP_CONFIG_INVALID,
                f"{env_name} must be >= 1.",
                details={"env": env_name, "value": raw_value},
            )
        return value

    @classmethod
    def _resolve_ingest_worker_count(cls) -> int:
        raw_value = os.environ.get("MCP_EBOOK_INGEST_WORKERS")
        if raw_value is None or not raw_value.strip():
            raw_value = "auto"
        if raw_value.strip().lower() == "auto":
            return cls._auto_ingest_worker_count()
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise AppError(
                ErrorCode.STARTUP_CONFIG_INVALID,
                "MCP_EBOOK_INGEST_WORKERS must be a positive integer or 'auto'.",
                details={"env": "MCP_EBOOK_INGEST_WORKERS", "value": raw_value},
            ) from exc
        if value < 1:
            raise AppError(
                ErrorCode.STARTUP_CONFIG_INVALID,
                "MCP_EBOOK_INGEST_WORKERS must be >= 1.",
                details={"env": "MCP_EBOOK_INGEST_WORKERS", "value": raw_value},
            )
        return value

    @classmethod
    def _resolve_docling_performance_config(cls) -> PdfParserPerformanceConfig:
        base = cls._auto_docling_performance_config()
        batch_size = cls._resolve_positive_int_env(
            env_name="PDF_DOCLING_BATCH_SIZE",
            default=base.ocr_batch_size,
        )
        device = (
            os.environ.get("PDF_DOCLING_DEVICE", base.device).strip() or base.device
        )
        return PdfParserPerformanceConfig(
            num_threads=cls._resolve_positive_int_env(
                env_name="PDF_DOCLING_NUM_THREADS",
                default=base.num_threads,
            ),
            device=device,
            ocr_batch_size=batch_size,
            layout_batch_size=batch_size,
            table_batch_size=batch_size,
        )

    @classmethod
    def from_env(cls) -> "AppService":
        preflight_errors: list[dict[str, Any]] = []

        def env_bool(name: str, default: bool) -> bool:
            value = os.environ.get(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        grobid_client = GrobidClient.from_env()

        try:
            ingest_worker_count = cls._resolve_ingest_worker_count()
        except AppError as exc:
            preflight_errors.append(
                {
                    "component": "config",
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details or None,
                }
            )
            ingest_worker_count = 1

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

        try:
            docling_performance_config = cls._resolve_docling_performance_config()
        except AppError as exc:
            preflight_errors.append(
                {
                    "component": "config",
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details or None,
                }
            )
            docling_performance_config = PdfParserPerformanceConfig()

        try:
            pdf_parse_timeout_seconds = cls._resolve_positive_int_env(
                env_name="PDF_PARSE_TIMEOUT_SECONDS",
                default=1800,
            )
        except AppError as exc:
            preflight_errors.append(
                {
                    "component": "config",
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details or None,
                }
            )
            pdf_parse_timeout_seconds = 1800

        if preflight_errors:
            raise AppError(
                ErrorCode.STARTUP_CONFIG_INVALID,
                "Startup configuration is invalid. Fix the listed environment values before running mcp-ebook-read.",
                details={
                    "invalid_config": preflight_errors,
                    "required_env": {},
                    "optional_env": {
                        "GROBID_URL": "http://127.0.0.1:8070",
                        "GROBID_TIMEOUT_SECONDS": "120",
                        "MCP_EBOOK_INGEST_WORKERS": "auto",
                        "PDF_DOCLING_NUM_THREADS": "auto-derived",
                        "PDF_DOCLING_BATCH_SIZE": "auto-derived",
                        "PDF_FORMULA_BATCH_SIZE": "auto",
                    },
                    "setup_reference": "See README.md: Optional GROBID Paper Enrichment",
                },
            )

        return cls(
            sidecar_dir_name=".mcp-ebook-read",
            pdf_parser=DoclingPdfParser(
                enable_docling_formula_enrichment=env_bool(
                    "DOCLING_FORMULA_ENRICHMENT", False
                ),
                require_formula_engine=env_bool("PDF_FORMULA_REQUIRE_ENGINE", False),
                formula_batch_size=formula_batch_size,
                performance_config=docling_performance_config,
            ),
            pdf_image_extractor=PdfImageExtractor(min_area_ratio=0.01),
            pdf_visual_extractor=DoclingPdfVisualExtractor(
                performance_config=docling_performance_config,
            ),
            grobid_client=grobid_client,
            epub_parser=EbooklibEpubParser(),
            pdf_parse_timeout_seconds=pdf_parse_timeout_seconds,
            ingest_worker_count=ingest_worker_count,
        )

    def _catalog_key(self, catalog: CatalogStore) -> str:
        return str(catalog.db_path.resolve())

    def _normalize_root(self, root: str | Path) -> str:
        return str(Path(root).expanduser().resolve())

    def _register_root(self, root: str | Path) -> str:
        normalized = self._normalize_root(root)
        self._known_roots.add(normalized)
        return normalized

    def _resolve_library_root(self, root: str | Path | None = None) -> Path:
        resolved = (
            Path(root).expanduser().resolve()
            if root is not None
            else self.default_library_root
        )
        if not resolved.exists() or not resolved.is_dir():
            raise AppError(
                ErrorCode.SCAN_INVALID_ROOT,
                f"Invalid library root: {resolved}",
                details={"root": str(resolved)},
            )
        self._register_root(resolved)
        return resolved

    def _catalog_for_library_root(self, root: str | Path | None = None) -> CatalogStore:
        root_path = self._resolve_library_root(root)
        return self._get_or_create_catalog(root_path / self.sidecar_dir_name)

    def _library_root_for_path(
        self, path: Path, root: str | Path | None = None
    ) -> Path:
        if root is not None:
            root_path = self._resolve_library_root(root)
            self._ensure_path_under_root(path, root_path)
            return root_path

        matching_roots: list[Path] = []
        for known_root in self._known_roots:
            root_path = Path(known_root)
            if path == root_path or path.is_relative_to(root_path):
                matching_roots.append(root_path)
        if matching_roots:
            return max(matching_roots, key=lambda item: len(item.parts))

        root_path = self.default_library_root
        self._ensure_path_under_root(path, root_path)
        return root_path

    def _ensure_path_under_root(self, path: Path, root: Path) -> None:
        if path == root or path.is_relative_to(root):
            return
        raise AppError(
            ErrorCode.INGEST_DOC_NOT_FOUND,
            "Document path is outside the selected library root.",
            details={"path": str(path), "root": str(root)},
        )

    def _get_or_create_catalog(self, sidecar_dir: Path) -> CatalogStore:
        key = str((sidecar_dir / "catalog.db").resolve())
        catalog = self._catalogs.get(key)
        if catalog is not None:
            return catalog
        catalog = CatalogStore(Path(key))
        self._catalogs[key] = catalog
        return catalog

    def _catalog_for_document_path(
        self, path: str | Path, root: str | Path | None = None
    ) -> CatalogStore:
        resolved = Path(path).expanduser().resolve()
        root_path = self._library_root_for_path(resolved, root)
        return self._catalog_for_library_root(root_path)

    def _lookup_doc_in_loaded_catalogs(self, doc_id: str) -> CatalogStore | None:
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

    def _catalog_for_doc_id(self, doc_id: str) -> CatalogStore | None:
        catalog = self._lookup_doc_in_loaded_catalogs(doc_id)
        if catalog is not None:
            return catalog

        for root in sorted(self._known_roots):
            root_path = Path(root)
            if not root_path.exists() or not root_path.is_dir():
                continue
            self._discover_sidecar_catalogs(root)
            catalog = self._lookup_doc_in_loaded_catalogs(doc_id)
            if catalog is not None:
                return catalog
        return None

    def _bind_doc_catalog(self, doc_id: str, catalog: CatalogStore) -> None:
        self._doc_catalog_index[doc_id] = self._catalog_key(catalog)

    def _doc_workspace_dir(self, doc: DocumentRecord, catalog: CatalogStore) -> Path:
        self._bind_doc_catalog(doc.doc_id, catalog)
        return catalog.db_path.parent / "docs" / doc.doc_id

    def _missing_doc_details(self, doc_id: str) -> dict[str, Any]:
        details: dict[str, Any] = {
            "doc_id": doc_id,
            "known_roots": sorted(self._known_roots),
            "hint": "After a fresh server restart, call library_scan(root=...) or storage_list_sidecars(root=...) before using doc_id-only tools.",
        }
        if self._known_roots:
            details["reason"] = "Document id was not found under discovered roots."
        else:
            details["reason"] = "No roots have been discovered in this process."
        return details

    def _require_doc(
        self,
        doc_id: str,
        *,
        error_code: ErrorCode,
        message: str,
    ) -> tuple[DocumentRecord, CatalogStore]:
        catalog = self._catalog_for_doc_id(doc_id)
        doc = catalog.get_document_by_id(doc_id) if catalog is not None else None
        if doc is None or catalog is None:
            raise AppError(
                error_code,
                message,
                details=self._missing_doc_details(doc_id),
            )
        return doc, catalog

    def _require_doc_from_catalog(
        self,
        *,
        catalog_key: str,
        doc_id: str,
        error_code: ErrorCode,
        message: str,
    ) -> tuple[DocumentRecord, CatalogStore]:
        catalog = self._catalogs.get(catalog_key)
        doc = catalog.get_document_by_id(doc_id) if catalog is not None else None
        if doc is None or catalog is None:
            raise AppError(
                error_code,
                message,
                details={
                    "doc_id": doc_id,
                    "catalog_key": catalog_key,
                    "loaded_catalogs": sorted(self._catalogs.keys()),
                },
            )
        return doc, catalog

    def _resolve_doc(
        self, doc_id: str | None, path: str | None, root: str | None = None
    ) -> tuple[DocumentRecord, CatalogStore]:
        if path:
            resolved_path = Path(path).expanduser().resolve()
            if not resolved_path.exists() or not resolved_path.is_file():
                raise AppError(
                    ErrorCode.INGEST_DOC_NOT_FOUND,
                    f"Document not found: {path}",
                    details={"doc_id": doc_id, "path": str(resolved_path)},
                )
            doc_type = self._doc_type_from_path(resolved_path)
            root_path = self._library_root_for_path(resolved_path, root)
            catalog = self._catalog_for_library_root(root_path)
            doc = catalog.get_document_by_path(str(resolved_path))
            if doc:
                self._bind_doc_catalog(doc.doc_id, catalog)
                if doc_id and doc.doc_id != doc_id:
                    raise AppError(
                        ErrorCode.INGEST_DOC_NOT_FOUND,
                        "doc_id does not match the document at path",
                        details={
                            "doc_id": doc_id,
                            "resolved_doc_id": doc.doc_id,
                            "path": str(resolved_path),
                        },
                    )
                return doc, catalog

            sha256 = self._compute_sha256(resolved_path)
            resolved_doc_id = self._scanned_doc_id(resolved_path, sha256)
            if doc_id and doc_id != resolved_doc_id:
                raise AppError(
                    ErrorCode.INGEST_DOC_NOT_FOUND,
                    "doc_id does not match the document at path",
                    details={
                        "doc_id": doc_id,
                        "resolved_doc_id": resolved_doc_id,
                        "path": str(resolved_path),
                    },
                )
            doc = DocumentRecord(
                doc_id=resolved_doc_id,
                path=str(resolved_path),
                type=doc_type,
                sha256=sha256,
                mtime=resolved_path.stat().st_mtime,
                profile=self._infer_profile_from_path(resolved_path, doc_type),
            )
            catalog.upsert_scanned_document(doc)
            self._bind_doc_catalog(doc.doc_id, catalog)
            return doc, catalog

        if doc_id:
            if root is not None:
                catalog = self._catalog_for_library_root(root)
                doc = catalog.get_document_by_id(doc_id)
                if doc is None:
                    raise AppError(
                        ErrorCode.INGEST_DOC_NOT_FOUND,
                        "Document not found by doc_id under selected library root",
                        details={"doc_id": doc_id, "root": str(Path(root).resolve())},
                    )
                self._bind_doc_catalog(doc.doc_id, catalog)
                return doc, catalog
            return self._require_doc(
                doc_id,
                error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
                message="Document not found by doc_id or path",
            )

        raise AppError(
            ErrorCode.INGEST_DOC_NOT_FOUND, "Document not found by doc_id or path"
        )

    def _resolve_pdf_path_for_operation(
        self,
        doc_id: str | None,
        path: str | None,
        *,
        operation_name: str,
    ) -> tuple[Path, str | None]:
        if path:
            resolved = Path(path).expanduser().resolve()
            if not resolved.exists() or not resolved.is_file():
                raise AppError(
                    ErrorCode.INGEST_DOC_NOT_FOUND,
                    f"Document not found: {path}",
                )
            if resolved.suffix.lower() != ".pdf":
                raise AppError(
                    ErrorCode.INGEST_UNSUPPORTED_TYPE,
                    f"{operation_name} only supports PDF documents",
                    details={"path": str(resolved)},
                )
            if doc_id is not None:
                resolved_doc, _catalog = self._resolve_doc(doc_id, str(resolved))
                if resolved_doc.type != DocumentType.PDF:
                    raise AppError(
                        ErrorCode.INGEST_UNSUPPORTED_TYPE,
                        f"{operation_name} only supports PDF documents",
                        details={
                            "doc_id": resolved_doc.doc_id,
                            "type": resolved_doc.type,
                        },
                    )
                return resolved, resolved_doc.doc_id
            return resolved, None

        if doc_id is None:
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                "Document not found by doc_id or path",
            )

        doc, _catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message="Document not found by doc_id or path",
        )
        if doc.type != DocumentType.PDF:
            raise AppError(
                ErrorCode.INGEST_UNSUPPORTED_TYPE,
                f"{operation_name} only supports PDF documents",
                details={"doc_id": doc.doc_id, "type": doc.type},
            )
        return Path(doc.path).expanduser().resolve(), doc.doc_id

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

    @staticmethod
    def _infer_profile_from_path(path: Path, doc_type: DocumentType) -> Profile:
        if doc_type == DocumentType.PDF:
            parts = {part.lower() for part in path.resolve().parts}
            if "papers" in parts or "paper" in parts:
                return Profile.PAPER
        return Profile.BOOK

    def _compute_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _scan_worker_count(self, document_count: int) -> int:
        if document_count <= 1:
            return 1
        cpu_count = max(1, os.cpu_count() or 1)
        return max(
            1,
            min(
                document_count,
                cpu_count,
                max(2, self.ingest_worker_count * 2),
                8,
            ),
        )

    @staticmethod
    def _scanned_doc_id(path: Path, sha256: str) -> str:
        path_hash = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        return f"{sha256[:16]}-{path_hash[:8]}"

    def _scan_document_file(self, path: Path) -> dict[str, Any]:
        abs_path = str(path.resolve())
        file_type = self._doc_type_from_path(path)
        sha256 = self._compute_sha256(path)
        return {
            "doc_id": self._scanned_doc_id(path, sha256),
            "path": abs_path,
            "path_obj": path,
            "type": file_type,
            "profile": self._infer_profile_from_path(path, file_type),
            "sha256": sha256,
            "mtime": path.stat().st_mtime,
        }

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _ingest_lease_expires_at(self) -> str:
        lease_seconds = max(900, self.pdf_parse_timeout_seconds + 300)
        return (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat()

    def _ingest_result_from_doc(
        self, doc: DocumentRecord, catalog: CatalogStore
    ) -> dict[str, Any]:
        chunks = catalog.list_chunks(doc.doc_id)
        pdf_parser_lanes = doc.metadata.get("pdf_parser_lanes")
        pdf_parse_phase_seconds = doc.metadata.get("pdf_parse_phase_seconds")
        return {
            "doc_id": doc.doc_id,
            "profile": doc.profile,
            "parser_chain": doc.parser_chain,
            "chunks_count": len(chunks),
            "formulas_count": len(catalog.list_formulas(doc.doc_id)),
            "images_count": len(catalog.list_images(doc.doc_id)),
            "pdf_tables_count": len(catalog.list_pdf_tables(doc.doc_id)),
            "pdf_figures_count": len(catalog.list_pdf_figures(doc.doc_id)),
            "outline_depth": max((node.level for node in doc.outline), default=0),
            "overall_confidence": doc.overall_confidence,
            "pipeline_status": self._document_pipeline_status(doc),
            **(
                {"pdf_parser_lanes": pdf_parser_lanes}
                if isinstance(pdf_parser_lanes, dict)
                else {}
            ),
            **(
                {"pdf_parse_phase_seconds": pdf_parse_phase_seconds}
                if isinstance(pdf_parse_phase_seconds, dict)
                else {}
            ),
        }

    def _stable_digest(self, payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _model_payload(self, value: Any) -> Any:
        if value is None:
            return None
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        return value

    def _pipeline_config_payload(self) -> dict[str, Any]:
        return {
            "pdf_parser": {
                "class": type(self.pdf_parser).__name__,
                "enable_docling_formula_enrichment": getattr(
                    self.pdf_parser,
                    "enable_docling_formula_enrichment",
                    None,
                ),
                "require_formula_engine": getattr(
                    self.pdf_parser,
                    "require_formula_engine",
                    None,
                ),
                "performance_config": self._model_payload(
                    getattr(self.pdf_parser, "performance_config", None)
                ),
            },
            "ingest": {
                "mode": "eager_full_parse",
                "worker_count": self.ingest_worker_count,
            },
            "epub_parser": {"class": type(self.epub_parser).__name__},
            "pdf_image_extractor": {
                "class": type(self.pdf_image_extractor).__name__
                if self.pdf_image_extractor is not None
                else None,
                "min_area_ratio": getattr(
                    self.pdf_image_extractor,
                    "min_area_ratio",
                    None,
                ),
            },
            "pdf_visual_extractor": {
                "class": type(self.pdf_visual_extractor).__name__
                if self.pdf_visual_extractor is not None
                else None,
                "performance_config": self._model_payload(
                    getattr(self.pdf_visual_extractor, "performance_config", None)
                ),
            },
        }

    def _pipeline_metadata(
        self, *, doc_type: DocumentType, profile: Profile
    ) -> dict[str, Any]:
        versions = {
            "chunker": _CHUNKER_VERSION,
            "search_text": _SEARCH_TEXT_VERSION,
            "pdf_formula": _PDF_FORMULA_PIPELINE_VERSION
            if doc_type == DocumentType.PDF
            else None,
            "pdf_visuals": _PDF_VISUAL_PIPELINE_VERSION
            if doc_type == DocumentType.PDF
            else None,
            "pdf_diagnostic": _PDF_DIAGNOSTIC_PIPELINE_VERSION
            if doc_type == DocumentType.PDF
            else None,
            "epub_images": _EPUB_IMAGE_PIPELINE_VERSION
            if doc_type == DocumentType.EPUB
            else None,
        }
        config_digest = self._stable_digest(self._pipeline_config_payload())
        payload = {
            "schema_version": _PIPELINE_SCHEMA_VERSION,
            "doc_type": doc_type.value,
            "profile": profile.value,
            "versions": versions,
            "config_digest": config_digest,
        }
        return {
            **payload,
            "digest": self._stable_digest(payload),
        }

    def _document_pipeline_status(self, doc: DocumentRecord) -> dict[str, Any]:
        current = self._pipeline_metadata(doc_type=doc.type, profile=doc.profile)
        stored = doc.metadata.get(_PIPELINE_METADATA_KEY)
        if doc.status != DocumentStatus.READY:
            return {
                "is_stale": False,
                "reason": "document_not_ready",
                "current_digest": current["digest"],
                "stored_digest": stored.get("digest")
                if isinstance(stored, dict)
                else None,
            }
        source_path = Path(doc.path)
        if not source_path.exists():
            return {
                "is_stale": True,
                "reason": "source_path_missing",
                "freshness": "source_missing",
                "current_digest": current["digest"],
                "stored_digest": stored.get("digest")
                if isinstance(stored, dict)
                else None,
                "path": doc.path,
                "hint": "The source file no longer exists. Restore it or run storage_cleanup_sidecars(root=...) to prune stale records.",
            }
        current_mtime = source_path.stat().st_mtime
        if abs(float(current_mtime) - float(doc.mtime)) > 1e-6:
            current_sha256 = self._compute_sha256(source_path)
            if current_sha256 != doc.sha256:
                return {
                    "is_stale": True,
                    "reason": "source_file_changed",
                    "freshness": "needs_reingest",
                    "stored_sha256": doc.sha256,
                    "current_sha256": current_sha256,
                    "stored_mtime": doc.mtime,
                    "current_mtime": current_mtime,
                    "current_digest": current["digest"],
                    "stored_digest": stored.get("digest")
                    if isinstance(stored, dict)
                    else None,
                    "hint": self._reingest_hint(doc),
                }
        if not isinstance(stored, dict):
            return {
                "is_stale": True,
                "reason": "missing_pipeline_metadata",
                "current_digest": current["digest"],
                "stored_digest": None,
                "current_versions": current["versions"],
                "hint": self._reingest_hint(doc),
            }
        if stored.get("digest") != current["digest"]:
            return {
                "is_stale": True,
                "reason": "pipeline_digest_mismatch",
                "current_digest": current["digest"],
                "stored_digest": stored.get("digest"),
                "current_versions": current["versions"],
                "stored_versions": stored.get("versions"),
                "hint": self._reingest_hint(doc),
            }
        return {
            "is_stale": False,
            "reason": "current",
            "current_digest": current["digest"],
            "stored_digest": stored.get("digest"),
            "current_versions": current["versions"],
        }

    def _reingest_hint(self, doc: DocumentRecord) -> str:
        call = self._reingest_call(doc)
        tool_name = call["tool"]
        return f"Run {tool_name}(doc_id='{doc.doc_id}', force=true) to refresh persisted artifacts."

    def _reingest_call(self, doc: DocumentRecord) -> dict[str, Any]:
        if doc.type == DocumentType.EPUB:
            tool_name = "document_ingest"
        elif doc.profile == Profile.PAPER:
            tool_name = "document_ingest"
        else:
            tool_name = "document_ingest"
        return {
            "tool": tool_name,
            "args": {"doc_id": doc.doc_id, "force": True},
        }

    def _serialize_ingest_job(self, job: IngestJobRecord) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "doc_id": job.doc_id,
            "path": job.path,
            "profile": job.profile,
            "status": job.status,
            "stage": job.stage,
            "force": job.force,
            "message": job.message,
            "progress": self._serialize_ingest_progress(job),
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "owner_id": job.owner_id,
            "claimed_at": job.claimed_at,
            "heartbeat_at": job.heartbeat_at,
            "lease_expires_at": job.lease_expires_at,
        }

    def _serialize_ingest_progress(self, job: IngestJobRecord) -> dict[str, Any]:
        progress = dict(job.progress or {})
        started_at = self._parse_iso_datetime(job.started_at)
        finished_at = self._parse_iso_datetime(job.finished_at)
        if started_at is None:
            return progress

        end_at = finished_at or datetime.now(UTC)
        elapsed_ms = max(0, int((end_at - started_at).total_seconds() * 1000))
        progress["elapsed_ms"] = elapsed_ms

        done = progress.get("done")
        total = progress.get("total")
        if (
            job.status == IngestJobStatus.RUNNING
            and isinstance(done, int)
            and isinstance(total, int)
            and 0 < done < total
        ):
            progress["eta_ms"] = int(elapsed_ms / done * (total - done))
        return progress

    def _parse_iso_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _ingest_progress(
        self,
        *,
        stage: IngestStage,
        message: str | None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        total = self._INGEST_STAGE_TOTAL
        done = self._INGEST_STAGE_DONE[stage]
        return {
            "stage": stage.value,
            "done": done,
            "total": total,
            "pct": round(done / total * 100, 1),
            "current_item": message,
            "diagnostics": diagnostics or {},
        }

    def _ingest_job_payload(
        self,
        job: IngestJobRecord,
        *,
        deduplicated: bool = False,
        cached: bool = False,
    ) -> dict[str, Any]:
        payload = self._serialize_ingest_job(job)
        payload["deduplicated"] = deduplicated
        payload["cached"] = cached
        payload["hint"] = (
            "Call document_ingest_status(doc_id=..., job_id=...) to poll status."
        )
        return payload

    def _reading_capture_enabled(self) -> bool:
        return os.environ.get(
            "MCP_EBOOK_CAPTURE_READING_SESSION", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _reading_capture_include_query(self) -> bool:
        return os.environ.get(
            "MCP_EBOOK_CAPTURE_INCLUDE_QUERY", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _reading_session_log_path(self, catalog: CatalogStore) -> Path:
        return catalog.db_path.parent / "eval" / "reading-session.jsonl"

    def _infer_capture_doc_id(
        self,
        *,
        kwargs: dict[str, Any],
        result: dict[str, Any],
    ) -> str | None:
        doc_id = kwargs.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            return doc_id

        locator = kwargs.get("locator")
        if isinstance(locator, dict):
            locator_doc_id = locator.get("doc_id")
            if isinstance(locator_doc_id, str) and locator_doc_id:
                return locator_doc_id

        doc_ids = kwargs.get("doc_ids")
        if isinstance(doc_ids, list) and doc_ids:
            first_doc_id = doc_ids[0]
            if isinstance(first_doc_id, str) and first_doc_id:
                return first_doc_id

        for hit in result.get("hits") or []:
            hit_doc_id = hit.get("doc_id")
            if isinstance(hit_doc_id, str) and hit_doc_id:
                return hit_doc_id
        return None

    def _capture_input_payload(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in (
            "root",
            "doc_id",
            "doc_ids",
            "node_id",
            "top_k",
            "limit",
            "image_id",
            "table_id",
            "figure_id",
            "formula_id",
            "format",
            "max_chunks",
        ):
            if key in kwargs:
                payload[key] = kwargs[key]

        query = kwargs.get("query")
        if isinstance(query, str):
            payload["query_sha256"] = hashlib.sha256(query.encode("utf-8")).hexdigest()
            if self._reading_capture_include_query():
                payload["query"] = query

        locator = kwargs.get("locator")
        if isinstance(locator, dict):
            payload["locator"] = {
                key: locator.get(key)
                for key in (
                    "doc_id",
                    "chunk_id",
                    "section_path",
                    "page_range",
                    "method",
                )
                if key in locator
            }
        return payload

    def _returned_ids(self, result: Any) -> dict[str, list[str]]:
        buckets = {
            "node_ids": set(),
            "chunk_ids": set(),
            "formula_ids": set(),
            "image_ids": set(),
            "table_ids": set(),
            "figure_ids": set(),
        }
        key_to_bucket = {
            "node_id": "node_ids",
            "chunk_id": "chunk_ids",
            "formula_id": "formula_ids",
            "image_id": "image_ids",
            "table_id": "table_ids",
            "figure_id": "figure_ids",
        }

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    bucket = key_to_bucket.get(key)
                    if bucket and isinstance(item, str):
                        buckets[bucket].add(item)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(result)
        return {key: sorted(values) for key, values in buckets.items()}

    def capture_tool_call(
        self,
        *,
        tool_name: str,
        use_case: str,
        kwargs: dict[str, Any],
        result: dict[str, Any],
        latency_ms: int,
    ) -> dict[str, Any]:
        if not self._reading_capture_enabled() or use_case not in {
            "search",
            "read",
            "image",
            "table",
            "figure",
            "formula",
            "outline",
            "render",
        }:
            return result

        output = dict(result)
        try:
            doc_id = self._infer_capture_doc_id(kwargs=kwargs, result=result)
            if doc_id is None:
                output["eval_capture"] = {
                    "enabled": True,
                    "captured": False,
                    "reason": "doc_id_not_found",
                }
                return output

            doc, catalog = self._require_doc(
                doc_id,
                error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
                message="Document not found for reading-session capture.",
            )
            event_id = uuid4().hex
            event = {
                "event_id": event_id,
                "ts": self._now_iso(),
                "tool_name": tool_name,
                "doc_id": doc.doc_id,
                "input": self._capture_input_payload(kwargs),
                "returned_ids": self._returned_ids(result),
                "latency_ms": latency_ms,
                "parser_chain": doc.parser_chain,
                "pipeline_status": self._document_pipeline_status(doc),
            }
            log_path = self._reading_session_log_path(catalog)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            output["eval_capture"] = {
                "enabled": True,
                "captured": True,
                "event_id": event_id,
            }
            return output
        except Exception as exc:  # noqa: BLE001
            logger.exception("reading_session_capture_failed")
            output["eval_capture"] = {
                "enabled": True,
                "captured": False,
                "reason": "capture_failed",
                "error": str(exc),
            }
            return output

    def _read_reading_session_events(
        self, root: str | Path | None, limit: int
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        catalogs = self._discover_sidecar_catalogs(root)
        events: list[dict[str, Any]] = []
        for catalog in catalogs:
            log_path = self._reading_session_log_path(catalog)
            if not log_path.exists():
                continue
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                events.append(json.loads(line))
        events.sort(key=lambda event: event.get("ts") or "")
        return events[-limit:]

    def eval_export_reading_sessions(
        self,
        *,
        root: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        root_path = self._resolve_library_root(root)
        events = self._read_reading_session_events(root_path, limit)
        return {
            "root": str(root_path),
            "capture_enabled": self._reading_capture_enabled(),
            "query_capture_enabled": self._reading_capture_include_query(),
            "events_count": len(events),
            "events": events,
            "privacy": {
                "file_paths": "scrubbed",
                "queries": "captured only when MCP_EBOOK_CAPTURE_INCLUDE_QUERY=1",
            },
        }

    def eval_replay_reading_sessions(
        self,
        *,
        root: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        root_path = self._resolve_library_root(root)
        events = self._read_reading_session_events(root_path, limit)
        replayed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for event in events:
            tool_name = event.get("tool_name")
            event_input = event.get("input") or {}
            query = event_input.get("query")
            if tool_name not in {
                "library_explore",
                "document_explore",
                "search",
                "search_in_outline_node",
            }:
                skipped.append(
                    {
                        "event_id": event.get("event_id"),
                        "tool_name": tool_name,
                        "reason": "tool_not_replayable",
                    }
                )
                continue

            if not query:
                skipped.append(
                    {
                        "event_id": event.get("event_id"),
                        "tool_name": tool_name,
                        "reason": "query_not_captured",
                    }
                )
                continue

            try:
                if tool_name == "library_explore":
                    replay_root = event_input.get("root")
                    if not isinstance(replay_root, str) or not replay_root:
                        raise ValueError(
                            "library_explore replay requires captured root"
                        )
                    current = self.library_explore(
                        root=replay_root,
                        query=query,
                        top_k=int(event_input.get("top_k") or 12),
                    )
                elif tool_name == "document_explore":
                    current = self.document_explore(
                        doc_id=event_input["doc_id"],
                        query=query,
                        top_k=int(event_input.get("top_k") or 8),
                    )
                elif tool_name == "search":
                    current = self.search(
                        query=query,
                        doc_ids=event_input.get("doc_ids"),
                        top_k=int(event_input.get("top_k") or 20),
                    )
                else:
                    current = self.search_in_outline_node(
                        doc_id=event_input["doc_id"],
                        node_id=event_input["node_id"],
                        query=query,
                        top_k=int(event_input.get("top_k") or 20),
                    )
            except Exception as exc:  # noqa: BLE001
                skipped.append(
                    {
                        "event_id": event.get("event_id"),
                        "tool_name": tool_name,
                        "reason": "replay_failed",
                        "error": to_error_payload(exc),
                    }
                )
                continue
            current_ids = self._returned_ids(current)
            original_ids = event.get("returned_ids") or {}
            replayed.append(
                {
                    "event_id": event.get("event_id"),
                    "tool_name": tool_name,
                    "drifted": current_ids != original_ids,
                    "original_ids": original_ids,
                    "current_ids": current_ids,
                }
            )

        return {
            "root": str(root_path),
            "events_count": len(events),
            "replayed_count": len(replayed),
            "skipped_count": len(skipped),
            "drifted_count": sum(1 for item in replayed if item["drifted"]),
            "replayed": replayed,
            "skipped": skipped,
        }

    def _doctor_component(
        self,
        *,
        name: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "status": status,
            "details": details or {},
        }

    def _artifact_findings_for_doc(
        self,
        *,
        catalog: CatalogStore,
        doc: DocumentRecord,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for image in catalog.list_images(doc.doc_id):
            if not Path(image.file_path).exists():
                findings.append(
                    {
                        "severity": "warning",
                        "code": "MISSING_IMAGE_ARTIFACT",
                        "doc_id": doc.doc_id,
                        "image_id": image.image_id,
                        "path": image.file_path,
                        "hint": self._reingest_hint(doc),
                    }
                )
        for table in catalog.list_pdf_tables(doc.doc_id):
            if not Path(table.file_path).exists():
                findings.append(
                    {
                        "severity": "warning",
                        "code": "MISSING_TABLE_ARTIFACT",
                        "doc_id": doc.doc_id,
                        "table_id": table.table_id,
                        "path": table.file_path,
                        "hint": "Call pdf_list_tables or reingest the document to regenerate table evidence.",
                    }
                )
            for segment in table.segments:
                if not Path(segment.file_path).exists():
                    findings.append(
                        {
                            "severity": "warning",
                            "code": "MISSING_TABLE_SEGMENT_ARTIFACT",
                            "doc_id": doc.doc_id,
                            "table_id": table.table_id,
                            "path": segment.file_path,
                            "hint": "Call pdf_list_tables or reingest the document to regenerate table evidence.",
                        }
                    )
        for figure in catalog.list_pdf_figures(doc.doc_id):
            if not Path(figure.file_path).exists():
                findings.append(
                    {
                        "severity": "warning",
                        "code": "MISSING_FIGURE_ARTIFACT",
                        "doc_id": doc.doc_id,
                        "figure_id": figure.figure_id,
                        "path": figure.file_path,
                        "hint": "Call pdf_list_figures or reingest the document to regenerate figure evidence.",
                    }
                )
        for formula in catalog.list_formulas(doc.doc_id):
            if (
                formula.chunk_id
                and catalog.get_chunk(doc.doc_id, formula.chunk_id) is None
            ):
                findings.append(
                    {
                        "severity": "warning",
                        "code": "FORMULA_CHUNK_MISSING",
                        "doc_id": doc.doc_id,
                        "formula_id": formula.formula_id,
                        "chunk_id": formula.chunk_id,
                        "hint": self._reingest_hint(doc),
                    }
                )
        return findings

    def doctor_health_check(
        self,
        *,
        root: str | None = None,
    ) -> dict[str, Any]:
        components: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []

        components.append(
            self._doctor_component(
                name="local_sqlite_index",
                status="ok",
                details={
                    "backend": "sqlite_fts5",
                    "scope": "library-root sidecar",
                },
            )
        )

        if self.grobid_client.base_url:
            try:
                self.grobid_client.assert_available()
                components.append(
                    self._doctor_component(
                        name="grobid_optional",
                        status="ok",
                        details={"base_url": self.grobid_client.base_url},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                components.append(
                    self._doctor_component(
                        name="grobid_optional",
                        status="warning",
                        details=to_error_payload(exc),
                    )
                )
        else:
            components.append(
                self._doctor_component(
                    name="grobid_optional",
                    status="skipped",
                    details={
                        "reason": "GROBID_URL is not configured.",
                        "impact": "PDF paper ingest will skip optional metadata/reference enrichment.",
                    },
                )
            )
        components.append(
            self._doctor_component(
                name="parsers",
                status="ok",
                details={
                    "pdf_parser": type(self.pdf_parser).__name__,
                    "epub_parser": type(self.epub_parser).__name__,
                    "pdf_image_extractor": type(self.pdf_image_extractor).__name__
                    if self.pdf_image_extractor is not None
                    else None,
                    "pdf_visual_extractor": type(self.pdf_visual_extractor).__name__
                    if self.pdf_visual_extractor is not None
                    else None,
                },
            )
        )

        catalog_reports: list[dict[str, Any]] = []
        if root is not None:
            root_path = Path(root).expanduser().resolve()
            catalogs = self._discover_sidecar_catalogs(str(root_path))
            for catalog in catalogs:
                docs = catalog.list_documents()
                known_doc_ids = {doc.doc_id for doc in docs}
                report = {
                    "catalog_path": str(catalog.db_path),
                    "documents_count": len(docs),
                    "db_size_bytes": catalog.db_size_bytes(),
                    "documents": [],
                }
                for doc in docs:
                    chunks_count = len(catalog.list_chunks(doc.doc_id))
                    pipeline_status = self._document_pipeline_status(doc)
                    doc_report = {
                        "doc_id": doc.doc_id,
                        "type": doc.type,
                        "profile": doc.profile,
                        "status": doc.status,
                        "chunks_count": chunks_count,
                        "path_exists": Path(doc.path).exists(),
                        "pipeline_status": pipeline_status,
                        "local_index_backend": "sqlite_fts5",
                    }
                    artifact_findings = self._artifact_findings_for_doc(
                        catalog=catalog,
                        doc=doc,
                    )
                    doc_report["artifact_findings_count"] = len(artifact_findings)
                    findings.extend(artifact_findings)
                    report["documents"].append(doc_report)
                    if doc.status == DocumentStatus.READY and chunks_count == 0:
                        findings.append(
                            {
                                "severity": "warning",
                                "code": "READY_DOCUMENT_HAS_NO_CHUNKS",
                                "doc_id": doc.doc_id,
                                "hint": self._reingest_hint(doc),
                            }
                        )
                    if not doc_report["path_exists"]:
                        findings.append(
                            {
                                "severity": "warning",
                                "code": "SOURCE_PATH_MISSING",
                                "doc_id": doc.doc_id,
                                "hint": "Run storage_cleanup_sidecars(root=...) to prune missing documents.",
                            }
                        )
                    if pipeline_status["is_stale"]:
                        findings.append(
                            {
                                "severity": "warning",
                                "code": "DOCUMENT_PIPELINE_STALE",
                                "doc_id": doc.doc_id,
                                "hint": pipeline_status.get("hint"),
                            }
                        )
                docs_dir = catalog.db_path.parent / "docs"
                if docs_dir.exists():
                    for child in docs_dir.iterdir():
                        if child.is_dir() and child.name not in known_doc_ids:
                            findings.append(
                                {
                                    "severity": "warning",
                                    "code": "ORPHAN_DOC_ARTIFACT_DIR",
                                    "catalog_path": str(catalog.db_path),
                                    "doc_dir": str(child),
                                    "hint": "Run storage_cleanup_sidecars(root=..., remove_orphan_artifacts=true).",
                                }
                            )
                catalog_reports.append(report)
        components.append(
            self._doctor_component(
                name="sidecar_catalogs",
                status="ok",
                details={
                    "root": str(Path(root).expanduser().resolve()) if root else None,
                    "catalogs_count": len(catalog_reports),
                    "catalogs": catalog_reports,
                },
            )
        )

        error_count = sum(1 for item in components if item["status"] == "error")
        warning_count = sum(1 for item in components if item["status"] == "warning")
        warning_count += sum(1 for item in findings if item["severity"] == "warning")
        return {
            "ok": error_count == 0,
            "error_count": error_count,
            "warning_count": warning_count,
            "components": components,
            "findings": findings,
        }

    def _set_job_stage(
        self,
        *,
        catalog: CatalogStore,
        job_id: str,
        status: IngestJobStatus | None = None,
        stage: IngestStage | None = None,
        message: str | None = None,
        progress: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        if progress is None and stage is not None:
            progress = self._ingest_progress(stage=stage, message=message)
        is_terminal = status in {
            IngestJobStatus.SUCCEEDED,
            IngestJobStatus.FAILED,
            IngestJobStatus.CANCELED,
        } or stage in {
            IngestStage.COMPLETED,
            IngestStage.FAILED,
            IngestStage.CANCELED,
        }
        heartbeat_at = self._now_iso()
        with self._ingest_lock:
            catalog.update_ingest_job(
                job_id,
                status=status,
                stage=stage,
                message=message,
                progress=progress,
                result=result,
                error=error,
                started_at=started_at,
                finished_at=finished_at,
                owner_id=self._ingest_owner_id,
                heartbeat_at=heartbeat_at,
                lease_expires_at=None
                if is_terminal
                else self._ingest_lease_expires_at(),
            )

    def _submit_ingest_job(
        self,
        *,
        doc: DocumentRecord,
        catalog: CatalogStore,
        profile: Profile,
        expected_doc_type: DocumentType,
        force: bool,
        ingest_mode: str,
    ) -> dict[str, Any]:
        if doc.type != expected_doc_type:
            raise AppError(
                ErrorCode.INGEST_UNSUPPORTED_TYPE,
                f"{ingest_mode} only supports {expected_doc_type.value} documents",
                details={
                    "ingest_mode": ingest_mode,
                    "doc_type": doc.type,
                    "supported_doc_types": [expected_doc_type],
                },
            )
        with self._ingest_lock:
            catalog.fail_expired_running_ingest_jobs()
            active = catalog.get_active_ingest_job(doc.doc_id)
            if active is not None and active.profile == profile:
                return self._ingest_job_payload(active, deduplicated=True)

            if (
                doc.status == DocumentStatus.READY
                and not force
                and doc.profile == profile
                and not self._document_pipeline_status(doc)["is_stale"]
            ):
                now = self._now_iso()
                message = "Cached READY document reused; no ingest work was scheduled."
                job = IngestJobRecord(
                    job_id=uuid4().hex,
                    doc_id=doc.doc_id,
                    path=doc.path,
                    profile=profile,
                    status=IngestJobStatus.SUCCEEDED,
                    stage=IngestStage.CACHED,
                    force=force,
                    message=message,
                    progress=self._ingest_progress(
                        stage=IngestStage.CACHED,
                        message=message,
                    ),
                    result=self._ingest_result_from_doc(doc, catalog),
                    created_at=now,
                    updated_at=now,
                    started_at=now,
                    finished_at=now,
                )
                catalog.create_ingest_job(job)
                return self._ingest_job_payload(job, cached=True)

            now = self._now_iso()
            message = f"{ingest_mode} queued for background execution."
            job = IngestJobRecord(
                job_id=uuid4().hex,
                doc_id=doc.doc_id,
                path=doc.path,
                profile=profile,
                status=IngestJobStatus.QUEUED,
                stage=IngestStage.QUEUED,
                force=force,
                message=message,
                progress=self._ingest_progress(
                    stage=IngestStage.QUEUED,
                    message=message,
                ),
                created_at=now,
                updated_at=now,
            )
            catalog.create_ingest_job(job)
        catalog_key = self._catalog_key(catalog)
        self._ingest_queue.put(
            _IngestQueueItem(
                catalog_key=catalog_key,
                doc_id=doc.doc_id,
                job_id=job.job_id,
            )
        )
        logger.info(
            "ingest_job_queued",
            extra={
                "doc_id": doc.doc_id,
                "job_id": job.job_id,
                "catalog_key": catalog_key,
                "profile": profile,
                "path": doc.path,
            },
        )
        return self._ingest_job_payload(job)

    def _next_persisted_ingest_item(self) -> _IngestQueueItem | None:
        with self._ingest_lock:
            catalogs = list(self._catalogs.items())
        for catalog_key, catalog in catalogs:
            job = catalog.get_next_queued_ingest_job()
            if job is not None:
                return _IngestQueueItem(
                    catalog_key=catalog_key,
                    doc_id=job.doc_id,
                    job_id=job.job_id,
                )
        return None

    def _ingest_worker_loop(self) -> None:
        while True:
            task_done_required = False
            try:
                item = self._ingest_queue.get(timeout=0.5)
                task_done_required = True
            except Empty:
                if self._closed:
                    return
                item = self._next_persisted_ingest_item()
                if item is None:
                    continue
            try:
                self._run_ingest_job(item=item)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "ingest_worker_unhandled_error",
                    extra={
                        "doc_id": item.doc_id,
                        "job_id": item.job_id,
                        "catalog_key": item.catalog_key,
                    },
                )
            finally:
                if task_done_required:
                    self._ingest_queue.task_done()

    def _run_ingest_job(self, *, item: _IngestQueueItem) -> None:
        doc, catalog = self._require_doc_from_catalog(
            catalog_key=item.catalog_key,
            doc_id=item.doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message="Document not found while executing ingest job",
        )
        job = catalog.get_ingest_job(item.doc_id, item.job_id)
        if job is None:
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                "Ingest job not found",
                details={
                    "doc_id": item.doc_id,
                    "job_id": item.job_id,
                    "catalog_key": item.catalog_key,
                },
            )
        if job.status != IngestJobStatus.QUEUED:
            return

        started_at = self._now_iso()
        claimed = catalog.claim_ingest_job(
            item.job_id,
            owner_id=self._ingest_owner_id,
            lease_expires_at=self._ingest_lease_expires_at(),
            now=started_at,
        )
        if claimed is None:
            return
        job = claimed
        self._set_job_stage(
            catalog=catalog,
            job_id=item.job_id,
            status=IngestJobStatus.RUNNING,
            stage=IngestStage.PARSE,
            message="Worker claimed background ingest job.",
            started_at=started_at,
        )
        logger.info(
            "ingest_job_started",
            extra={
                "doc_id": item.doc_id,
                "job_id": item.job_id,
                "catalog_key": item.catalog_key,
                "profile": job.profile,
            },
        )

        def stage_callback(stage: IngestStage, message: str) -> None:
            self._set_job_stage(
                catalog=catalog,
                job_id=item.job_id,
                status=IngestJobStatus.RUNNING,
                stage=stage,
                message=message,
            )
            logger.info(
                "ingest_job_stage",
                extra={
                    "doc_id": item.doc_id,
                    "job_id": item.job_id,
                    "catalog_key": item.catalog_key,
                    "stage": stage,
                    "stage_message": message,
                },
            )

        try:
            result = self._document_ingest(
                doc=doc,
                catalog=catalog,
                profile=job.profile,
                expected_doc_type=DocumentType.PDF
                if doc.type == DocumentType.PDF
                else DocumentType.EPUB,
                ingest_mode=f"background_{job.profile}_{doc.type}",
                force=job.force,
                stage_callback=stage_callback,
            )
        except Exception as exc:  # noqa: BLE001
            error = to_error_payload(exc)
            error["traceback"] = traceback.format_exc()
            self._set_job_stage(
                catalog=catalog,
                job_id=item.job_id,
                status=IngestJobStatus.FAILED,
                stage=IngestStage.FAILED,
                message=str(exc),
                progress=self._ingest_progress(
                    stage=IngestStage.FAILED,
                    message=str(exc),
                    diagnostics={"error_code": error.get("code")},
                ),
                error=error,
                finished_at=self._now_iso(),
            )
            logger.exception(
                "ingest_job_failed",
                extra={
                    "doc_id": item.doc_id,
                    "job_id": item.job_id,
                    "catalog_key": item.catalog_key,
                },
            )
            return

        self._set_job_stage(
            catalog=catalog,
            job_id=item.job_id,
            status=IngestJobStatus.SUCCEEDED,
            stage=IngestStage.COMPLETED,
            message="Background ingest job completed successfully.",
            progress=self._ingest_progress(
                stage=IngestStage.COMPLETED,
                message="Background ingest job completed successfully.",
                diagnostics={
                    "chunks_count": result.get("chunks_count"),
                    "formulas_count": result.get("formulas_count"),
                    "images_count": result.get("images_count"),
                },
            ),
            result=result,
            finished_at=self._now_iso(),
        )
        logger.info(
            "ingest_job_succeeded",
            extra={
                "doc_id": item.doc_id,
                "job_id": item.job_id,
                "catalog_key": item.catalog_key,
                "chunks": result["chunks_count"],
            },
        )

    def library_scan(self, root: str | None, patterns: list[str]) -> dict[str, Any]:
        scan_started = time.perf_counter()
        root_path = self._resolve_library_root(root)
        root_catalog = self._catalog_for_library_root(root_path)

        found_paths: set[str] = set()
        found_paths_by_catalog: dict[str, set[str]] = defaultdict(set)
        added: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        unchanged_count = 0
        candidate_paths: list[Path] = []
        found_paths_by_catalog.setdefault(self._catalog_key(root_catalog), set())

        for pattern in patterns:
            for path in sorted(root_path.glob(pattern)):
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
                candidate_paths.append(path)

        hash_started = time.perf_counter()
        scan_worker_count = self._scan_worker_count(len(candidate_paths))
        scanned_documents: list[dict[str, Any]] = []
        if scan_worker_count <= 1:
            scanned_documents = [
                self._scan_document_file(path) for path in candidate_paths
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=scan_worker_count,
                thread_name_prefix="mcp-ebook-read-scan",
            ) as executor:
                futures = [
                    executor.submit(self._scan_document_file, path)
                    for path in candidate_paths
                ]
                for future in as_completed(futures):
                    scanned_documents.append(future.result())
            scanned_documents.sort(key=lambda item: str(item["path"]))
        hash_seconds = time.perf_counter() - hash_started

        removed: list[str] = []
        removed_deleted_count = 0
        maintenance_rows: list[dict[str, Any]] = []
        with self._ingest_lock:
            for scanned in scanned_documents:
                path_obj = scanned["path_obj"]
                abs_path = str(scanned["path"])
                self._ensure_path_under_root(path_obj, root_path)
                catalog = root_catalog
                found_paths_by_catalog[self._catalog_key(catalog)].add(abs_path)

                doc = DocumentRecord(
                    doc_id=str(scanned["doc_id"]),
                    path=abs_path,
                    type=scanned["type"],
                    profile=scanned["profile"],
                    sha256=str(scanned["sha256"]),
                    mtime=float(scanned["mtime"]),
                )
                state = catalog.upsert_scanned_document(doc)
                self._bind_doc_catalog(doc.doc_id, catalog)
                payload = {
                    "doc_id": doc.doc_id,
                    "path": abs_path,
                    "sha256": doc.sha256,
                    "mtime": doc.mtime,
                    "type": doc.type,
                    "profile": doc.profile,
                }
                if state == "added":
                    added.append(payload)
                elif state == "updated":
                    updated.append(payload)
                else:
                    unchanged_count += 1

            for catalog_key, paths in found_paths_by_catalog.items():
                catalog = self._catalogs[catalog_key]
                known = set(catalog.list_document_paths_under_root(str(root_path)))
                removed_paths = sorted(list(known - paths))
                if removed_paths:
                    for removed_path in removed_paths:
                        removed_doc = catalog.get_document_by_path(removed_path)
                        if removed_doc is not None:
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

        return {
            "added": added,
            "updated": updated,
            "unchanged_count": unchanged_count,
            "removed": sorted(set(removed)),
            "removed_deleted_count": removed_deleted_count,
            "storage_maintenance": {
                "catalogs": maintenance_rows,
            },
            "scan_performance": {
                "candidate_documents": len(candidate_paths),
                "hash_workers": scan_worker_count,
                "hash_seconds": round(hash_seconds, 6),
                "total_seconds": round(time.perf_counter() - scan_started, 6),
            },
        }

    def _relative_path_payload(self, *, path: str, root_path: Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return str(resolved.relative_to(root_path))
        except ValueError:
            return str(resolved)

    def _ingest_dashboard_job_payload(
        self,
        *,
        job: IngestJobRecord,
        doc: DocumentRecord | None,
        root_path: Path,
        include_error: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "job_id": job.job_id,
            "doc_id": job.doc_id,
            "path": job.path,
            "relative_path": self._relative_path_payload(
                path=job.path,
                root_path=root_path,
            ),
            "document_status": doc.status if doc is not None else None,
            "document_type": doc.type if doc is not None else None,
            "profile": job.profile,
            "status": job.status,
            "stage": job.stage,
            "force": job.force,
            "message": job.message,
            "progress": self._serialize_ingest_progress(job),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "owner_id": job.owner_id,
            "claimed_at": job.claimed_at,
            "heartbeat_at": job.heartbeat_at,
            "lease_expires_at": job.lease_expires_at,
        }
        if isinstance(job.result, dict):
            payload["result_summary"] = {
                key: job.result.get(key)
                for key in (
                    "chunks_count",
                    "formulas_count",
                    "images_count",
                    "pdf_tables_count",
                    "pdf_figures_count",
                    "overall_confidence",
                )
                if key in job.result
            }
        if include_error and isinstance(job.error, dict):
            payload["error"] = {
                key: job.error.get(key)
                for key in ("code", "message", "details")
                if key in job.error
            }
        return payload

    def library_ingest_documents(
        self,
        *,
        root: str | None = None,
        force: bool = False,
        max_documents: int = 0,
    ) -> dict[str, Any]:
        root_path = self._resolve_library_root(root)
        scan = self.library_scan(str(root_path), ["**/*.pdf", "**/*.epub"])
        catalog = self._catalog_for_library_root(root_path)
        catalog.fail_expired_running_ingest_jobs()
        documents = sorted(catalog.list_documents(), key=lambda item: item.path)
        max_to_queue = max(0, max_documents)

        queued: list[dict[str, Any]] = []
        skipped_ready_count = 0
        unsupported_count = 0
        limited_count = 0
        deduplicated_count = 0
        cached_count = 0
        selected_count = 0

        for doc in documents:
            if doc.type == DocumentType.EPUB:
                profile = Profile.BOOK
                expected_doc_type = DocumentType.EPUB
            elif doc.type == DocumentType.PDF:
                profile = doc.profile
                expected_doc_type = DocumentType.PDF
            else:
                unsupported_count += 1
                continue

            pipeline_status = self._document_pipeline_status(doc)
            if (
                doc.status == DocumentStatus.READY
                and not force
                and doc.profile == profile
                and not pipeline_status["is_stale"]
            ):
                skipped_ready_count += 1
                continue

            if max_to_queue and selected_count >= max_to_queue:
                limited_count += 1
                continue

            selected_count += 1
            job = self._submit_ingest_job(
                doc=doc,
                catalog=catalog,
                profile=profile,
                expected_doc_type=expected_doc_type,
                force=force,
                ingest_mode="library_ingest_documents",
            )
            if job.get("deduplicated"):
                deduplicated_count += 1
            elif job.get("cached"):
                cached_count += 1
            else:
                queued.append(
                    {
                        "job_id": job["job_id"],
                        "doc_id": doc.doc_id,
                        "relative_path": self._relative_path_payload(
                            path=doc.path,
                            root_path=root_path,
                        ),
                        "status": job["status"],
                        "stage": job["stage"],
                        "profile": job["profile"],
                    }
                )

        return {
            "root": str(root_path),
            "catalog_path": str(catalog.db_path),
            "documents_total": len(documents),
            "selected_count": selected_count,
            "queued_count": len(queued),
            "deduplicated_count": deduplicated_count,
            "cached_count": cached_count,
            "skipped_ready_count": skipped_ready_count,
            "unsupported_count": unsupported_count,
            "limited_count": limited_count,
            "scan": {
                "added_count": len(scan["added"]),
                "updated_count": len(scan["updated"]),
                "unchanged_count": scan["unchanged_count"],
                "removed_count": len(scan["removed"]),
                "performance": scan["scan_performance"],
            },
            "jobs": queued[:100],
            "jobs_truncated_count": max(0, len(queued) - 100),
            "next_call": "library_ingest_status",
        }

    def library_ingest_status(
        self,
        *,
        root: str | None = None,
        limit_running: int = 20,
        limit_failed: int = 20,
        limit_queued: int = 20,
    ) -> dict[str, Any]:
        root_path = self._resolve_library_root(root)
        catalog = self._catalog_for_library_root(root_path)
        expired_running_recovered_count = catalog.fail_expired_running_ingest_jobs()
        documents = sorted(catalog.list_documents(), key=lambda item: item.path)
        doc_by_id = {doc.doc_id: doc for doc in documents}
        jobs = catalog.list_all_ingest_jobs(limit=10000)

        latest_by_doc: dict[str, IngestJobRecord] = {}
        for job in sorted(jobs, key=lambda item: item.created_at, reverse=True):
            latest_by_doc.setdefault(job.doc_id, job)
        latest_jobs = list(latest_by_doc.values())

        document_status_counts: dict[str, int] = defaultdict(int)
        document_profile_counts: dict[str, int] = defaultdict(int)
        document_type_counts: dict[str, int] = defaultdict(int)
        for doc in documents:
            document_status_counts[doc.status] += 1
            document_profile_counts[doc.profile] += 1
            document_type_counts[doc.type] += 1

        job_status_counts: dict[str, int] = defaultdict(int)
        job_stage_counts: dict[str, int] = defaultdict(int)
        for job in latest_jobs:
            job_status_counts[job.status] += 1
            job_stage_counts[job.stage] += 1

        running_jobs = [
            job for job in latest_jobs if job.status == IngestJobStatus.RUNNING
        ]
        queued_jobs = [
            job for job in latest_jobs if job.status == IngestJobStatus.QUEUED
        ]
        failed_jobs = [
            job for job in latest_jobs if job.status == IngestJobStatus.FAILED
        ]
        running_jobs.sort(key=lambda item: item.updated_at, reverse=True)
        queued_jobs.sort(key=lambda item: item.created_at)
        failed_jobs.sort(key=lambda item: item.updated_at, reverse=True)

        ready_count = document_status_counts[DocumentStatus.READY]
        documents_total = len(documents)
        pct_ready = (
            round(ready_count / documents_total * 100.0, 2)
            if documents_total
            else 100.0
        )

        running_limit = max(0, min(limit_running, 100))
        failed_limit = max(0, min(limit_failed, 100))
        queued_limit = max(0, min(limit_queued, 100))

        return {
            "root": str(root_path),
            "catalog_path": str(catalog.db_path),
            "owner_id": self._ingest_owner_id,
            "local_ingest_workers": self.ingest_worker_count,
            "documents": {
                "total": documents_total,
                "ready": ready_count,
                "pct_ready": pct_ready,
                "by_status": dict(sorted(document_status_counts.items())),
                "by_profile": dict(sorted(document_profile_counts.items())),
                "by_type": dict(sorted(document_type_counts.items())),
            },
            "jobs": {
                "total_records": len(jobs),
                "latest_tracked_documents": len(latest_jobs),
                "documents_without_jobs": documents_total - len(latest_jobs),
                "by_status": dict(sorted(job_status_counts.items())),
                "by_stage": dict(sorted(job_stage_counts.items())),
                "expired_running_recovered_count": expired_running_recovered_count,
            },
            "progress": {
                "completed_documents": ready_count,
                "total_documents": documents_total,
                "pct_ready": pct_ready,
                "running_count": len(running_jobs),
                "queued_count": len(queued_jobs),
                "failed_latest_count": len(failed_jobs),
            },
            "running": [
                self._ingest_dashboard_job_payload(
                    job=job,
                    doc=doc_by_id.get(job.doc_id),
                    root_path=root_path,
                )
                for job in running_jobs[:running_limit]
            ],
            "queued": [
                self._ingest_dashboard_job_payload(
                    job=job,
                    doc=doc_by_id.get(job.doc_id),
                    root_path=root_path,
                )
                for job in queued_jobs[:queued_limit]
            ],
            "recent_failed": [
                self._ingest_dashboard_job_payload(
                    job=job,
                    doc=doc_by_id.get(job.doc_id),
                    root_path=root_path,
                    include_error=True,
                )
                for job in failed_jobs[:failed_limit]
            ],
            "truncated": {
                "running": max(0, len(running_jobs) - running_limit),
                "queued": max(0, len(queued_jobs) - queued_limit),
                "recent_failed": max(0, len(failed_jobs) - failed_limit),
            },
        }

    def _write_reading_artifact(
        self, workspace_dir: Path, parsed: ParsedDocument
    ) -> None:
        target = workspace_dir / "reading" / "reading.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(parsed.reading_markdown, encoding="utf-8")

    def _persist_raw_artifacts(
        self,
        *,
        workspace_dir: Path,
        parsed: ParsedDocument,
    ) -> list[dict[str, Any]]:
        target_dir = workspace_dir / "raw"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        if not parsed.raw_artifacts:
            return []
        target_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for index, (name, content) in enumerate(sorted(parsed.raw_artifacts.items())):
            safe_name = "".join(
                char if char.isalnum() or char in {"-", "_", "."} else "_"
                for char in name.strip()
            ).strip("._")
            if not safe_name:
                safe_name = f"raw-{index}"
            suffix = Path(safe_name).suffix.lower()
            if suffix not in {
                ".html",
                ".htm",
                ".xhtml",
                ".xml",
                ".json",
                ".md",
                ".txt",
            }:
                safe_name = f"{safe_name}.txt"
            target = target_dir / safe_name
            target.write_text(str(content), encoding="utf-8")
            records.append(
                {
                    "name": name,
                    "artifact_id": f"raw-{index}",
                    "file_path": str(target),
                    "media_type": "text/plain",
                    "bytes": target.stat().st_size,
                }
            )
        return records

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

    @staticmethod
    def _remap_staged_path(
        value: str,
        *,
        staging_workspace_dir: Path,
        workspace_dir: Path,
    ) -> str:
        path = Path(value)
        try:
            relative = path.relative_to(staging_workspace_dir)
        except ValueError:
            return value
        return str(workspace_dir / relative)

    def _remap_raw_artifact_records(
        self,
        records: list[dict[str, Any]],
        *,
        staging_workspace_dir: Path,
        workspace_dir: Path,
    ) -> list[dict[str, Any]]:
        remapped: list[dict[str, Any]] = []
        for record in records:
            file_path = record.get("file_path")
            if isinstance(file_path, str):
                record = {
                    **record,
                    "file_path": self._remap_staged_path(
                        file_path,
                        staging_workspace_dir=staging_workspace_dir,
                        workspace_dir=workspace_dir,
                    ),
                }
            remapped.append(record)
        return remapped

    def _remap_image_record_paths(
        self,
        records: list[ImageRecord],
        *,
        staging_workspace_dir: Path,
        workspace_dir: Path,
    ) -> list[ImageRecord]:
        remapped: list[ImageRecord] = []
        for record in records:
            remapped.append(
                record.model_copy(
                    update={
                        "file_path": self._remap_staged_path(
                            record.file_path,
                            staging_workspace_dir=staging_workspace_dir,
                            workspace_dir=workspace_dir,
                        )
                    }
                )
            )
        return remapped

    def _remap_pdf_table_record_paths(
        self,
        records: list[PdfTableRecord],
        *,
        staging_workspace_dir: Path,
        workspace_dir: Path,
    ) -> list[PdfTableRecord]:
        remapped: list[PdfTableRecord] = []
        for record in records:
            remapped_segments = [
                segment.model_copy(
                    update={
                        "file_path": self._remap_staged_path(
                            segment.file_path,
                            staging_workspace_dir=staging_workspace_dir,
                            workspace_dir=workspace_dir,
                        )
                    }
                )
                for segment in record.segments
            ]
            remapped.append(
                record.model_copy(
                    update={
                        "file_path": self._remap_staged_path(
                            record.file_path,
                            staging_workspace_dir=staging_workspace_dir,
                            workspace_dir=workspace_dir,
                        ),
                        "segments": remapped_segments,
                    }
                )
            )
        return remapped

    def _remap_pdf_figure_record_paths(
        self,
        records: list[PdfFigureRecord],
        *,
        staging_workspace_dir: Path,
        workspace_dir: Path,
    ) -> list[PdfFigureRecord]:
        remapped: list[PdfFigureRecord] = []
        for record in records:
            remapped.append(
                record.model_copy(
                    update={
                        "file_path": self._remap_staged_path(
                            record.file_path,
                            staging_workspace_dir=staging_workspace_dir,
                            workspace_dir=workspace_dir,
                        )
                    }
                )
            )
        return remapped

    @staticmethod
    def _replace_workspace_with_staging(
        *,
        staging_workspace_dir: Path,
        workspace_dir: Path,
    ) -> Path | None:
        backup_dir: Path | None = None
        workspace_dir.parent.mkdir(parents=True, exist_ok=True)
        if workspace_dir.exists():
            backup_dir = workspace_dir.with_name(
                f".{workspace_dir.name}.previous-{uuid4().hex}"
            )
            workspace_dir.rename(backup_dir)
        staging_workspace_dir.rename(workspace_dir)
        return backup_dir

    @staticmethod
    def _rollback_workspace_swap(
        *,
        backup_dir: Path | None,
        workspace_dir: Path,
    ) -> None:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists():
            backup_dir.rename(workspace_dir)

    @staticmethod
    def _cleanup_staging_artifacts(
        *,
        staging_workspace_dir: Path,
        backup_dir: Path | None = None,
    ) -> None:
        shutil.rmtree(staging_workspace_dir, ignore_errors=True)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)

    def _persist_parse_output_to_catalog(
        self,
        *,
        catalog: CatalogStore,
        doc: DocumentRecord,
        profile: Profile,
        parsed: ParsedDocument,
        image_records: list[ImageRecord],
        pdf_table_records: list[PdfTableRecord],
        pdf_figure_records: list[PdfFigureRecord],
        current_sha256: str,
        current_mtime: float,
    ) -> None:
        catalog.replace_chunks(doc.doc_id, parsed.chunks, rebuild_graph=False)
        catalog.replace_formulas(doc.doc_id, parsed.formulas, rebuild_graph=False)
        if doc.type == DocumentType.PDF:
            catalog.replace_pdf_tables(
                doc.doc_id,
                pdf_table_records,
                rebuild_graph=False,
            )
            catalog.replace_pdf_figures(
                doc.doc_id,
                pdf_figure_records,
                rebuild_graph=False,
            )
        catalog.replace_images(doc.doc_id, image_records, rebuild_graph=False)
        catalog.update_document_source_metadata(
            doc_id=doc.doc_id,
            sha256=current_sha256,
            mtime=current_mtime,
        )
        catalog.save_document_parse_output(
            doc_id=doc.doc_id,
            title=parsed.title,
            parser_chain=parsed.parser_chain,
            metadata=parsed.metadata,
            outline=parsed.outline,
            overall_confidence=parsed.overall_confidence,
            status=DocumentStatus.READY,
        )
        validation_issues = catalog.validate_document_graph(doc.doc_id)
        validation_errors = [
            issue for issue in validation_issues if issue.get("severity") == "error"
        ]
        if validation_errors:
            raise AppError(
                ErrorCode.INGEST_GRAPH_VALIDATION_FAILED,
                "Document graph validation failed before ingest finalization.",
                details={
                    "doc_id": doc.doc_id,
                    "profile": profile,
                    "errors": validation_errors,
                    "warnings": [
                        issue
                        for issue in validation_issues
                        if issue.get("severity") != "error"
                    ],
                },
            )

    def _pdf_parse_worker_config(
        self,
        *,
        active_pdf_parses: int,
        visual_tables_dir: Path | None = None,
        visual_figures_dir: Path | None = None,
    ) -> dict[str, Any]:
        formula_extractor = getattr(self.pdf_parser, "formula_extractor", None)
        formula_batch_size = getattr(formula_extractor, "batch_size", 1)
        performance_config = getattr(
            self.pdf_parser,
            "performance_config",
            self._auto_docling_performance_config(),
        )
        if not isinstance(performance_config, PdfParserPerformanceConfig):
            performance_config = self._auto_docling_performance_config()
        if not self._docling_performance_env_overridden():
            performance_config = self._scale_pdf_performance_for_active_parses(
                performance_config,
                active_pdf_parses=active_pdf_parses,
            )
        if (
            not os.environ.get("PDF_FORMULA_BATCH_SIZE")
            or os.environ.get("PDF_FORMULA_BATCH_SIZE", "").strip().lower() == "auto"
        ):
            formula_batch_size = max(
                1,
                int(formula_batch_size or 1) // max(1, active_pdf_parses),
            )
        config = {
            "enable_docling_formula_enrichment": bool(
                getattr(self.pdf_parser, "enable_docling_formula_enrichment", False)
            ),
            "require_formula_engine": bool(
                getattr(self.pdf_parser, "require_formula_engine", False)
            ),
            "formula_batch_size": int(formula_batch_size or 1),
            "performance_config": performance_config.model_dump(mode="json"),
            "resource_plan": {
                "active_pdf_parses": active_pdf_parses,
                "ingest_worker_count": self.ingest_worker_count,
                "docling_performance_env_overridden": (
                    self._docling_performance_env_overridden()
                ),
            },
        }
        if (
            isinstance(self.pdf_visual_extractor, DoclingPdfVisualExtractor)
            and visual_tables_dir is not None
            and visual_figures_dir is not None
        ):
            config["visual_extraction"] = {
                "tables_dir": str(visual_tables_dir),
                "figures_dir": str(visual_figures_dir),
                "images_scale": self.pdf_visual_extractor.images_scale,
            }
            config["resource_plan"]["docling_visuals_in_parse_worker"] = True
        else:
            config["resource_plan"]["docling_visuals_in_parse_worker"] = False
        return config

    @staticmethod
    def _parsed_text_summary(parsed: ParsedDocument) -> dict[str, Any]:
        text = "\n".join(chunk.text for chunk in parsed.chunks)
        normalized = re.sub(r"\s+", " ", text).strip()
        return {
            "normalized_text_chars": len(normalized),
            "wordish_tokens": len(re.findall(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", text)),
            "chunks": len(parsed.chunks),
            "outline_nodes": len(parsed.outline),
            "formulas": len(parsed.formulas),
            "images": len(parsed.images),
        }

    @staticmethod
    def _ratio_or_none(
        numerator: int | float, denominator: int | float
    ) -> float | None:
        if denominator <= 0:
            return None
        return numerator / denominator

    def _pdf_fast_preflight(self, path: str, doc_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        parser = Pypdfium2PdfParser()
        try:
            parsed = parser.parse(path, doc_id)
            metadata = dict(parsed.metadata or {})
            return {
                "status": "ok",
                "parser": parser.method,
                "seconds": round(time.perf_counter() - started, 6),
                "pages": metadata.get("pages"),
                "phase_seconds": metadata.get("pdf_parse_phase_seconds"),
                **self._parsed_text_summary(parsed),
                "fidelity_warning": (
                    "Fast preflight is text/outline oriented. Docling remains the "
                    "canonical lane for formulas, tables, figures, and layout-aware evidence."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "parser": parser.method,
                "seconds": round(time.perf_counter() - started, 6),
                "error": to_error_payload(exc),
                "impact": (
                    "High-fidelity Docling ingest will still run; only fast parser "
                    "diagnostics are unavailable for this PDF."
                ),
            }

    def _pdf_diagnostic_preflight(self, path: str, doc_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        parser_name = "pymupdf"
        try:
            import fitz

            images_total = 0
            text_blocks_total = 0
            page_samples: list[dict[str, Any]] = []
            pdf_doc = fitz.open(path)
            try:
                for page_index, page in enumerate(pdf_doc):
                    images_count = len(page.get_images(full=True) or [])
                    try:
                        blocks = page.get_text("blocks") or []
                    except Exception:  # noqa: BLE001
                        blocks = []
                    block_count = len(blocks)
                    images_total += images_count
                    text_blocks_total += block_count
                    if page_index < 5:
                        page_samples.append(
                            {
                                "page": page_index + 1,
                                "images": images_count,
                                "text_blocks": block_count,
                            }
                        )
                metadata = getattr(pdf_doc, "metadata", None) or {}
                outline = pdf_doc.get_toc() or []
                page_count = int(getattr(pdf_doc, "page_count", 0) or 0)
            finally:
                close = getattr(pdf_doc, "close", None)
                if callable(close):
                    close()
            return {
                "status": "ok",
                "parser": parser_name,
                "seconds": round(time.perf_counter() - started, 6),
                "pages": page_count,
                "outline_nodes": len(outline),
                "images": images_total,
                "text_blocks": text_blocks_total,
                "page_samples": page_samples,
                "metadata": {
                    key: str(value)
                    for key, value in dict(metadata).items()
                    if value is not None
                },
                "role": (
                    "Diagnostic lane for image/block inventory and page-level "
                    "rendering evidence. Docling remains canonical for persisted "
                    "text, formula, table, and figure structure."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "parser": parser_name,
                "seconds": round(time.perf_counter() - started, 6),
                "error": to_error_payload(exc),
                "impact": (
                    "High-fidelity Docling ingest will still run; only PyMuPDF "
                    "diagnostic inventory is unavailable for this PDF."
                ),
                "details": {"doc_id": doc_id, "path": path},
            }

    def _run_pdf_preflights(
        self, path: str, doc_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="mcp-ebook-read-pdf-preflight",
        ) as executor:
            fast_future = executor.submit(self._pdf_fast_preflight, path, doc_id)
            diagnostic_future = executor.submit(
                self._pdf_diagnostic_preflight,
                path,
                doc_id,
            )
            fast_preflight = fast_future.result()
            diagnostic_preflight = diagnostic_future.result()

        execution = {
            "mode": "parallel",
            "lanes": ["pypdfium2_fast", "pymupdf_diagnostic"],
            "seconds": round(time.perf_counter() - started, 6),
        }
        fast_preflight["preflight_execution"] = execution
        diagnostic_preflight["preflight_execution"] = execution
        return fast_preflight, diagnostic_preflight

    def _attach_pdf_parser_lane_metadata(
        self,
        parsed: ParsedDocument,
        *,
        fast_preflight: dict[str, Any] | None,
        diagnostic_preflight: dict[str, Any] | None,
    ) -> None:
        if fast_preflight is None and diagnostic_preflight is None:
            return
        canonical_summary = {
            "parser_chain": parsed.parser_chain,
            **self._parsed_text_summary(parsed),
            "pages": parsed.metadata.get("pages"),
            "formula_markers_total": parsed.metadata.get("formula_markers_total"),
            "formula_replaced_by_docling_text": parsed.metadata.get(
                "formula_replaced_by_docling_text"
            ),
            "formula_replaced_by_pix2text": parsed.metadata.get(
                "formula_replaced_by_pix2text"
            ),
            "docling_formula_text_candidates_total": parsed.metadata.get(
                "docling_formula_text_candidates_total"
            ),
            "formula_unresolved": parsed.metadata.get("formula_unresolved"),
            "tables": parsed.metadata.get("pdf_tables_count"),
            "figures": parsed.metadata.get("pdf_figures_count"),
        }
        fast_text_chars = int(fast_preflight.get("normalized_text_chars") or 0)
        canonical_text_chars = int(canonical_summary.get("normalized_text_chars") or 0)
        parsed.metadata = {
            **parsed.metadata,
            "pdf_parser_lanes": {
                "strategy": {
                    "fast_lane": "pypdfium2_pdf",
                    "canonical_fidelity_lane": "docling",
                    "diagnostic_lane": "pymupdf",
                    "canonical_content_source": "docling",
                    "rationale": (
                        "pypdfium2 is used as a fast preflight text/outline lane; "
                        "PyMuPDF is used as a diagnostic image/block/rendering lane; "
                        "Docling output remains canonical for persisted chunks and "
                        "high-fidelity reading evidence."
                    ),
                },
                "fast_preflight": fast_preflight,
                "diagnostic_preflight": diagnostic_preflight,
                "canonical_fidelity": canonical_summary,
                "fast_text_ratio_to_fidelity": self._ratio_or_none(
                    fast_text_chars,
                    canonical_text_chars,
                ),
            },
        }

    def _parse_pdf_document(
        self,
        path: str,
        doc_id: str,
        *,
        visual_tables_dir: Path | None = None,
        visual_figures_dir: Path | None = None,
    ) -> ParsedDocument:
        if not isinstance(self.pdf_parser, DoclingPdfParser):
            return self.pdf_parser.parse(path, doc_id)

        with self._active_pdf_parse_lock:
            self._active_pdf_parses += 1
            active_pdf_parses = self._active_pdf_parses
        worker_config = self._pdf_parse_worker_config(
            active_pdf_parses=active_pdf_parses,
            visual_tables_dir=visual_tables_dir,
            visual_figures_dir=visual_figures_dir,
        )
        command = [
            sys.executable,
            "-m",
            "mcp_ebook_read.workers.pdf_parse",
            "--pdf-path",
            path,
            "--doc-id",
            doc_id,
        ]
        timeout = self.pdf_parse_timeout_seconds
        try:
            try:
                completed = subprocess.run(
                    command,
                    input=json.dumps(worker_config, ensure_ascii=True),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise AppError(
                    ErrorCode.INGEST_PDF_DOCLING_TIMEOUT,
                    "Docling PDF parse timed out in the isolated worker process.",
                    details={
                        "doc_id": doc_id,
                        "path": path,
                        "timeout_seconds": timeout,
                        "resource_plan": worker_config.get("resource_plan"),
                        "hint": (
                            "Increase PDF_PARSE_TIMEOUT_SECONDS for very large PDFs, "
                            "or inspect the source PDF for pathological pages."
                        ),
                    },
                ) from exc

            try:
                payload = json.loads(completed.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise AppError(
                    ErrorCode.INGEST_PDF_DOCLING_FAILED,
                    "Docling PDF worker returned non-JSON output.",
                    details={
                        "doc_id": doc_id,
                        "path": path,
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr[-4000:],
                        "stdout_tail": completed.stdout[-4000:],
                        "resource_plan": worker_config.get("resource_plan"),
                    },
                ) from exc

            if completed.returncode != 0 or not payload.get("ok"):
                error = payload.get("error") if isinstance(payload, dict) else None
                details = error.get("details") if isinstance(error, dict) else None
                if not isinstance(details, dict):
                    details = {}
                error_code = ErrorCode.INGEST_PDF_DOCLING_FAILED
                if isinstance(error, dict) and error.get("code"):
                    try:
                        error_code = ErrorCode(str(error.get("code")))
                    except ValueError:
                        error_code = ErrorCode.INGEST_PDF_DOCLING_FAILED
                raise AppError(
                    error_code,
                    str(error.get("message"))
                    if isinstance(error, dict) and error.get("message")
                    else "Docling PDF worker failed.",
                    details={
                        **details,
                        "doc_id": doc_id,
                        "path": path,
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr[-4000:],
                        "worker_traceback": payload.get("traceback"),
                        "resource_plan": worker_config.get("resource_plan"),
                    },
                )

            parsed = ParsedDocument.model_validate(payload["data"])
            parsed.metadata = {
                **parsed.metadata,
                "pdf_parse_resource_plan": {
                    "performance_config": worker_config["performance_config"],
                    "formula_batch_size": worker_config["formula_batch_size"],
                    **worker_config["resource_plan"],
                },
            }
            return parsed
        finally:
            with self._active_pdf_parse_lock:
                self._active_pdf_parses = max(0, self._active_pdf_parses - 1)

    def _local_paper_metadata_fallback(
        self,
        parsed: ParsedDocument,
    ) -> dict[str, Any]:
        title = parsed.title.strip()
        abstract_text = ""
        abstract_source_chunk_id = None
        reference_evidence: list[dict[str, Any]] = []

        for chunk in parsed.chunks:
            section_label = CatalogStore._normalize_search_label(
                " ".join(chunk.section_path)
            )
            section_tokens = set(section_label.split())
            text = chunk.text.strip()
            text_label = CatalogStore._normalize_search_label(text[:120])
            is_abstract_section = bool(section_tokens & {"abstract", "摘要"})
            starts_with_abstract = text_label.startswith("abstract ")
            if not abstract_text and (is_abstract_section or starts_with_abstract):
                abstract_text = re.sub(
                    r"^\s*abstract\s*[:.\-]?\s*",
                    "",
                    text,
                    flags=re.IGNORECASE,
                ).strip()
                abstract_source_chunk_id = chunk.chunk_id

            is_reference_section = bool(
                section_tokens & {"references", "bibliography", "参考文献"}
            )
            if is_reference_section:
                reference_evidence.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "section_path": chunk.section_path,
                        "excerpt": text[:600],
                    }
                )

        fields: dict[str, Any] = {}
        if title:
            fields["paper_title"] = {
                "value": title,
                "source": "docling_title",
                "confidence": 0.65,
            }
        if abstract_text:
            fields["abstract"] = {
                "value": abstract_text[:4000],
                "source": "docling_section",
                "source_chunk_id": abstract_source_chunk_id,
                "confidence": 0.7,
            }
        if reference_evidence:
            fields["references_text_evidence"] = {
                "value": reference_evidence[:5],
                "source": "docling_reference_section",
                "confidence": 0.55,
                "truncated": len(reference_evidence) > 5,
            }

        fallback = {
            "local_paper_metadata_fallback": {
                "status": "used",
                "source": "docling_local_fallback",
                "fields": sorted(fields),
                "limitations": [
                    "Authors, DOI, citation graph, and bibliography entries are not reconstructed without GROBID.",
                    "Abstract and reference evidence are section heuristics and may need human verification.",
                ],
            }
        }
        for key, payload in fields.items():
            fallback[key] = payload["value"]
        return fallback

    def _pdf_visual_worker_config(self, chunks: list[ChunkRecord]) -> dict[str, Any]:
        if isinstance(self.pdf_visual_extractor, DoclingPdfVisualExtractor):
            performance_config = self.pdf_visual_extractor.performance_config
            images_scale = self.pdf_visual_extractor.images_scale
        else:
            performance_config = PdfParserPerformanceConfig()
            images_scale = 2.0
        return {
            "performance_config": performance_config.model_dump(mode="json"),
            "images_scale": images_scale,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }

    def _pop_embedded_pdf_visuals(
        self,
        parsed: ParsedDocument,
    ) -> tuple[list[PdfTableRecord], list[PdfFigureRecord], dict[str, Any] | None]:
        payload = parsed.metadata.pop("_pdf_visuals_from_docling_document", None)
        if not isinstance(payload, dict):
            return [], [], None
        tables = [
            PdfTableRecord.model_validate(table)
            for table in payload.get("tables", [])
            if isinstance(table, dict)
        ]
        figures = [
            PdfFigureRecord.model_validate(figure)
            for figure in payload.get("figures", [])
            if isinstance(figure, dict)
        ]
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = self._default_pdf_visuals_diagnostics(
                tables=tables,
                figures=figures,
            )
        execution = payload.get("execution")
        if isinstance(execution, dict):
            diagnostics = {
                **diagnostics,
                "execution": execution,
            }
        return tables, figures, diagnostics

    def _extract_pdf_visuals(
        self,
        *,
        pdf_path: str,
        doc_id: str,
        chunks: list[ChunkRecord],
        tables_dir: Path,
        figures_dir: Path,
    ) -> PdfVisualExtractionResult:
        if self.pdf_visual_extractor is None:
            return PdfVisualExtractionResult(tables=[], figures=[], diagnostics={})
        if not isinstance(self.pdf_visual_extractor, DoclingPdfVisualExtractor):
            return self.pdf_visual_extractor.extract(
                pdf_path=pdf_path,
                doc_id=doc_id,
                chunks=chunks,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
            )

        command = [
            sys.executable,
            "-m",
            "mcp_ebook_read.workers.pdf_visuals",
            "--pdf-path",
            pdf_path,
            "--doc-id",
            doc_id,
            "--tables-dir",
            str(tables_dir),
            "--figures-dir",
            str(figures_dir),
        ]
        timeout = self.pdf_parse_timeout_seconds
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(
                    self._pdf_visual_worker_config(chunks),
                    ensure_ascii=True,
                ),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_TIMEOUT,
                "Docling PDF visual extraction timed out in the isolated worker process.",
                details={
                    "doc_id": doc_id,
                    "path": pdf_path,
                    "timeout_seconds": timeout,
                    "component": "pdf_visuals",
                    "hint": (
                        "Increase PDF_PARSE_TIMEOUT_SECONDS for very large PDFs, "
                        "or inspect the source PDF for pathological visual pages."
                    ),
                },
            ) from exc

        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_FAILED,
                "Docling PDF visual worker returned non-JSON output.",
                details={
                    "doc_id": doc_id,
                    "path": pdf_path,
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-4000:],
                    "stdout_tail": completed.stdout[-4000:],
                },
            ) from exc

        if completed.returncode != 0 or not payload.get("ok"):
            error = payload.get("error") if isinstance(payload, dict) else None
            details = error.get("details") if isinstance(error, dict) else None
            if not isinstance(details, dict):
                details = {}
            error_code = ErrorCode.INGEST_PDF_DOCLING_FAILED
            if isinstance(error, dict) and error.get("code"):
                try:
                    error_code = ErrorCode(str(error.get("code")))
                except ValueError:
                    error_code = ErrorCode.INGEST_PDF_DOCLING_FAILED
            raise AppError(
                error_code,
                str(error.get("message"))
                if isinstance(error, dict) and error.get("message")
                else "Docling PDF visual worker failed.",
                details={
                    **details,
                    "doc_id": doc_id,
                    "path": pdf_path,
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-4000:],
                    "worker_traceback": payload.get("traceback"),
                },
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_FAILED,
                "Docling PDF visual worker returned an invalid payload.",
                details={
                    "doc_id": doc_id,
                    "path": pdf_path,
                    "returncode": completed.returncode,
                },
            )
        return PdfVisualExtractionResult(
            tables=[
                PdfTableRecord.model_validate(table)
                for table in data.get("tables", [])
                if isinstance(table, dict)
            ],
            figures=[
                PdfFigureRecord.model_validate(figure)
                for figure in data.get("figures", [])
                if isinstance(figure, dict)
            ],
            diagnostics=data.get("diagnostics") or {},
        )

    def _document_ingest(
        self,
        *,
        doc: DocumentRecord,
        catalog: CatalogStore,
        profile: Profile,
        expected_doc_type: DocumentType | None,
        ingest_mode: str,
        force: bool,
        stage_callback: Callable[[IngestStage, str], None] | None = None,
    ) -> dict[str, Any]:
        if (
            doc.status == DocumentStatus.READY
            and not force
            and doc.profile == profile
            and not self._document_pipeline_status(doc)["is_stale"]
        ):
            return {
                "doc_id": doc.doc_id,
                "profile": doc.profile,
                "parser_chain": doc.parser_chain,
                "chunks_count": len(
                    catalog.get_chunks_window(doc.doc_id, 0, 0, 1_000_000)
                ),
                "formulas_count": len(catalog.list_formulas(doc.doc_id)),
                "images_count": len(catalog.list_images(doc.doc_id)),
                "pdf_tables_count": len(catalog.list_pdf_tables(doc.doc_id)),
                "pdf_figures_count": len(catalog.list_pdf_figures(doc.doc_id)),
                "outline_depth": max((node.level for node in doc.outline), default=0),
                "overall_confidence": doc.overall_confidence,
                "pipeline_status": self._document_pipeline_status(doc),
            }

        self._bind_doc_catalog(doc.doc_id, catalog)
        workspace_dir = self._doc_workspace_dir(doc, catalog)
        staging_workspace_dir = workspace_dir.with_name(
            f".{workspace_dir.name}.staging-{uuid4().hex}"
        )
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
            source_path = Path(doc.path)
            if not source_path.exists() or not source_path.is_file():
                raise AppError(
                    ErrorCode.INGEST_DOC_NOT_FOUND,
                    f"Document not found: {doc.path}",
                    details={"doc_id": doc.doc_id, "path": doc.path},
                )
            current_sha256 = self._compute_sha256(source_path)
            current_mtime = source_path.stat().st_mtime

            if profile == Profile.BOOK:
                if doc.type == DocumentType.PDF:
                    if stage_callback is not None:
                        stage_callback(
                            IngestStage.PARSE,
                            "Running PDF fast/diagnostic preflights.",
                        )
                    (
                        pdf_fast_preflight,
                        pdf_diagnostic_preflight,
                    ) = self._run_pdf_preflights(
                        doc.path,
                        doc.doc_id,
                    )
                    if stage_callback is not None:
                        stage_callback(
                            IngestStage.PARSE,
                            "Parsing PDF book with Docling.",
                        )
                    parsed = self._parse_pdf_document(
                        doc.path,
                        doc.doc_id,
                        visual_tables_dir=self._pdf_tables_dir(staging_workspace_dir),
                        visual_figures_dir=self._pdf_figures_dir(staging_workspace_dir),
                    )
                    self._attach_pdf_parser_lane_metadata(
                        parsed,
                        fast_preflight=pdf_fast_preflight,
                        diagnostic_preflight=pdf_diagnostic_preflight,
                    )
                elif doc.type == DocumentType.EPUB:
                    if stage_callback is not None:
                        stage_callback(
                            IngestStage.PARSE,
                            "Parsing EPUB book with EbookLib.",
                        )
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
                grobid_metadata: dict[str, Any] = {}
                grobid_succeeded = False
                if self.grobid_client.base_url:
                    if stage_callback is not None:
                        stage_callback(
                            IngestStage.GROBID,
                            "Parsing optional PDF paper metadata with GROBID.",
                        )
                    try:
                        grobid_result = self.grobid_client.parse_fulltext(doc.path)
                        grobid_metadata = dict(grobid_result.metadata)
                        grobid_succeeded = True
                    except Exception as exc:  # noqa: BLE001
                        grobid_metadata = {
                            "grobid_enrichment": {
                                "status": "failed",
                                "reason": "GROBID optional paper metadata parsing failed.",
                                "impact": (
                                    "Docling PDF ingest will still run; paper title, "
                                    "abstract, DOI, bibliography count, and reference "
                                    "enrichment fall back to lower-confidence local evidence."
                                ),
                                "base_url": self.grobid_client.base_url,
                                "error": to_error_payload(exc),
                            }
                        }
                else:
                    grobid_metadata = {
                        "grobid_enrichment": {
                            "status": "skipped",
                            "reason": "GROBID_URL is not configured.",
                            "impact": (
                                "Paper title, abstract, DOI, bibliography count, and "
                                "reference enrichment may be unavailable or lower-confidence."
                            ),
                        }
                    }
                if stage_callback is not None:
                    stage_callback(
                        IngestStage.PARSE,
                        "Running PDF fast/diagnostic preflights.",
                    )
                (
                    pdf_fast_preflight,
                    pdf_diagnostic_preflight,
                ) = self._run_pdf_preflights(
                    doc.path,
                    doc.doc_id,
                )
                if stage_callback is not None:
                    stage_callback(
                        IngestStage.PARSE,
                        "Parsing PDF paper structure with Docling.",
                    )
                parsed = self._parse_pdf_document(
                    doc.path,
                    doc.doc_id,
                    visual_tables_dir=self._pdf_tables_dir(staging_workspace_dir),
                    visual_figures_dir=self._pdf_figures_dir(staging_workspace_dir),
                )
                self._attach_pdf_parser_lane_metadata(
                    parsed,
                    fast_preflight=pdf_fast_preflight,
                    diagnostic_preflight=pdf_diagnostic_preflight,
                )
                if grobid_succeeded:
                    parsed.parser_chain.append("grobid")
                    paper_metadata = grobid_metadata
                else:
                    paper_metadata = {
                        **self._local_paper_metadata_fallback(parsed),
                        **grobid_metadata,
                    }
                parsed.metadata = {**parsed.metadata, **paper_metadata}
                paper_title = str(paper_metadata.get("paper_title") or "").strip()
                if paper_title:
                    parsed.title = paper_title
            else:
                raise AppError(
                    ErrorCode.INGEST_UNSUPPORTED_TYPE,
                    f"Unsupported ingest profile: {profile}",
                )

            if stage_callback is not None:
                stage_callback(
                    IngestStage.PERSIST,
                    "Persisting reading artifacts and relational records.",
                )
            self._write_reading_artifact(staging_workspace_dir, parsed)
            raw_artifact_records = self._persist_raw_artifacts(
                workspace_dir=staging_workspace_dir,
                parsed=parsed,
            )
            image_records: list[ImageRecord] = []
            pdf_table_records: list[PdfTableRecord] = []
            pdf_figure_records: list[PdfFigureRecord] = []
            if doc.type == DocumentType.EPUB:
                image_records = self._persist_extracted_images(
                    doc_id=doc.doc_id,
                    images=parsed.images,
                    target_dir=staging_workspace_dir / "assets" / "epub-images",
                )
            elif doc.type == DocumentType.PDF:
                (
                    embedded_pdf_tables,
                    embedded_pdf_figures,
                    embedded_pdf_visual_diagnostics,
                ) = self._pop_embedded_pdf_visuals(parsed)
                if embedded_pdf_visual_diagnostics is not None:
                    pdf_table_records = embedded_pdf_tables
                    pdf_figure_records = embedded_pdf_figures
                image_future = None
                visual_future = None
                if self.pdf_image_extractor is not None:
                    images_dir = self._pdf_images_dir(staging_workspace_dir)
                else:
                    images_dir = None
                if (
                    self.pdf_visual_extractor is not None
                    and embedded_pdf_visual_diagnostics is None
                ):
                    tables_dir = self._pdf_tables_dir(staging_workspace_dir)
                    figures_dir = self._pdf_figures_dir(staging_workspace_dir)
                else:
                    tables_dir = None
                    figures_dir = None

                if self.pdf_image_extractor is not None or (
                    self.pdf_visual_extractor is not None
                    and embedded_pdf_visual_diagnostics is None
                ):
                    with ThreadPoolExecutor(
                        max_workers=2,
                        thread_name_prefix="mcp-ebook-read-pdf-visual",
                    ) as executor:
                        if (
                            self.pdf_image_extractor is not None
                            and images_dir is not None
                        ):
                            image_future = executor.submit(
                                self.pdf_image_extractor.extract,
                                pdf_path=doc.path,
                                doc_id=doc.doc_id,
                                chunks=parsed.chunks,
                                out_dir=images_dir,
                            )
                        if tables_dir is not None and figures_dir is not None:
                            visual_future = executor.submit(
                                self._extract_pdf_visuals,
                                pdf_path=doc.path,
                                doc_id=doc.doc_id,
                                chunks=parsed.chunks,
                                tables_dir=tables_dir,
                                figures_dir=figures_dir,
                            )

                        if image_future is not None:
                            image_records = image_future.result()
                        if visual_future is not None:
                            extracted_visuals = visual_future.result()
                            pdf_table_records = extracted_visuals.tables
                            pdf_figure_records = extracted_visuals.figures

                visual_execution_lanes = [
                    lane
                    for lane, future in (
                        ("pdf_images", image_future),
                        ("docling_tables_figures", visual_future),
                    )
                    if future is not None
                ]
                image_execution = {
                    "mode": "parallel" if len(visual_execution_lanes) > 1 else "single",
                    "lanes": visual_execution_lanes,
                }
                visual_execution = (
                    embedded_pdf_visual_diagnostics.get("execution")
                    if isinstance(embedded_pdf_visual_diagnostics, dict)
                    and isinstance(
                        embedded_pdf_visual_diagnostics.get("execution"), dict
                    )
                    else image_execution
                )

                if self.pdf_image_extractor is not None:
                    images_manifest_path = self._pdf_images_manifest_path(
                        staging_workspace_dir
                    )
                    images_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    images_manifest_path.write_text(
                        json.dumps(
                            {
                                "doc_id": doc.doc_id,
                                "images_count": len(image_records),
                                "mode": "eager",
                                "execution": image_execution,
                            },
                            ensure_ascii=True,
                        ),
                        encoding="utf-8",
                    )
                pdf_visual_diagnostics = self._default_pdf_visuals_diagnostics(
                    tables=[],
                    figures=[],
                )
                if embedded_pdf_visual_diagnostics is not None:
                    pdf_visual_diagnostics = embedded_pdf_visual_diagnostics
                elif (
                    self.pdf_visual_extractor is not None and visual_future is not None
                ):
                    diagnostics = getattr(extracted_visuals, "diagnostics", None)
                    pdf_visual_diagnostics = (
                        diagnostics
                        if isinstance(diagnostics, dict)
                        else self._default_pdf_visuals_diagnostics(
                            tables=pdf_table_records,
                            figures=pdf_figure_records,
                        )
                    )
                visuals_manifest_path = self._pdf_visuals_manifest_path(
                    staging_workspace_dir
                )
                visuals_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                visuals_manifest_path.write_text(
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "tables_count": len(pdf_table_records),
                            "figures_count": len(pdf_figure_records),
                            "mode": "eager",
                            "execution": visual_execution,
                            "diagnostics": pdf_visual_diagnostics,
                        },
                        ensure_ascii=True,
                    ),
                    encoding="utf-8",
                )
                parsed.metadata = {
                    **parsed.metadata,
                    "pdf_images_extraction_mode": "eager",
                    "pdf_tables_extraction_mode": "eager",
                    "pdf_figures_extraction_mode": "eager",
                    "pdf_visuals_diagnostics": {
                        "summary": pdf_visual_diagnostics.get("summary") or {},
                        "issues_count": len(pdf_visual_diagnostics.get("issues") or []),
                        "last_extracted_at": self._now_iso(),
                    },
                }
            raw_artifact_records = self._remap_raw_artifact_records(
                raw_artifact_records,
                staging_workspace_dir=staging_workspace_dir,
                workspace_dir=workspace_dir,
            )
            image_records = self._remap_image_record_paths(
                image_records,
                staging_workspace_dir=staging_workspace_dir,
                workspace_dir=workspace_dir,
            )
            pdf_table_records = self._remap_pdf_table_record_paths(
                pdf_table_records,
                staging_workspace_dir=staging_workspace_dir,
                workspace_dir=workspace_dir,
            )
            pdf_figure_records = self._remap_pdf_figure_record_paths(
                pdf_figure_records,
                staging_workspace_dir=staging_workspace_dir,
                workspace_dir=workspace_dir,
            )
            parsed.metadata = {
                **parsed.metadata,
                "raw_artifacts": raw_artifact_records,
                _PIPELINE_METADATA_KEY: self._pipeline_metadata(
                    doc_type=doc.type,
                    profile=profile,
                ),
            }
            if stage_callback is not None:
                stage_callback(
                    IngestStage.INDEX,
                    "Rebuilding local SQLite FTS index for document graph nodes.",
                )
            if stage_callback is not None:
                stage_callback(
                    IngestStage.FINALIZE,
                    "Saving final document metadata and marking document ready.",
                )
            staged_catalog: CatalogStore | None = None
            backup_workspace_dir: Path | None = None
            with self._ingest_lock:
                staged_catalog = catalog.create_staging_copy(
                    suffix=f"{doc.doc_id}-{uuid4().hex}"
                )
                try:
                    self._persist_parse_output_to_catalog(
                        catalog=staged_catalog,
                        doc=doc,
                        profile=profile,
                        parsed=parsed,
                        image_records=image_records,
                        pdf_table_records=pdf_table_records,
                        pdf_figure_records=pdf_figure_records,
                        current_sha256=current_sha256,
                        current_mtime=current_mtime,
                    )
                    backup_workspace_dir = self._replace_workspace_with_staging(
                        staging_workspace_dir=staging_workspace_dir,
                        workspace_dir=workspace_dir,
                    )
                    catalog.replace_with_staging_copy(staged_catalog)
                    self._cleanup_staging_artifacts(
                        staging_workspace_dir=staging_workspace_dir,
                        backup_dir=backup_workspace_dir,
                    )
                except Exception:
                    if backup_workspace_dir is not None:
                        self._rollback_workspace_swap(
                            backup_dir=backup_workspace_dir,
                            workspace_dir=workspace_dir,
                        )
                    if staged_catalog is not None and staged_catalog.db_path.exists():
                        staged_catalog.db_path.unlink(missing_ok=True)
                    raise
            return {
                "doc_id": doc.doc_id,
                "profile": profile,
                "parser_chain": parsed.parser_chain,
                "chunks_count": len(parsed.chunks),
                "formulas_count": len(parsed.formulas),
                "images_count": len(image_records),
                "pdf_tables_count": len(pdf_table_records),
                "pdf_figures_count": len(pdf_figure_records),
                "outline_depth": max(
                    (node.level for node in parsed.outline), default=0
                ),
                "overall_confidence": parsed.overall_confidence,
                "pipeline_status": self._document_pipeline_status(
                    DocumentRecord(
                        doc_id=doc.doc_id,
                        path=doc.path,
                        type=doc.type,
                        sha256=current_sha256,
                        mtime=current_mtime,
                        title=parsed.title,
                        status=DocumentStatus.READY,
                        profile=profile,
                        parser_chain=parsed.parser_chain,
                        metadata=parsed.metadata,
                        outline=parsed.outline,
                        overall_confidence=parsed.overall_confidence,
                    )
                ),
                **(
                    {"pdf_parser_lanes": parsed.metadata["pdf_parser_lanes"]}
                    if isinstance(parsed.metadata.get("pdf_parser_lanes"), dict)
                    else {}
                ),
                **(
                    {
                        "pdf_parse_phase_seconds": parsed.metadata[
                            "pdf_parse_phase_seconds"
                        ]
                    }
                    if isinstance(
                        parsed.metadata.get("pdf_parse_phase_seconds"),
                        dict,
                    )
                    else {}
                ),
            }
        except Exception:
            shutil.rmtree(staging_workspace_dir, ignore_errors=True)
            catalog.set_document_status(
                doc.doc_id, DocumentStatus.FAILED, profile=profile
            )
            raise

    def document_ingest(
        self,
        *,
        doc_id: str | None = None,
        path: str | None = None,
        root: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path, root)
        if doc.type == DocumentType.EPUB:
            profile = Profile.BOOK
            expected_doc_type = DocumentType.EPUB
        elif doc.type == DocumentType.PDF:
            profile = doc.profile
            expected_doc_type = DocumentType.PDF
        else:
            raise AppError(
                ErrorCode.INGEST_UNSUPPORTED_TYPE,
                f"Unsupported document type: {doc.type}",
                details={"doc_id": doc.doc_id, "path": doc.path, "type": doc.type},
            )
        return self._submit_ingest_job(
            doc=doc,
            catalog=catalog,
            profile=profile,
            expected_doc_type=expected_doc_type,
            force=force,
            ingest_mode="document_ingest",
        )

    def document_ingest_status(
        self, *, doc_id: str, job_id: str | None = None
    ) -> dict[str, Any]:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message="Document not found for ingest status lookup",
        )
        job = catalog.get_ingest_job(doc.doc_id, job_id)
        if job is None:
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                "Ingest job not found for document",
                details={"doc_id": doc.doc_id, "job_id": job_id},
            )
        payload = self._serialize_ingest_job(job)
        payload["document_status"] = doc.status
        payload["document_profile"] = doc.profile
        payload["pipeline_status"] = self._document_pipeline_status(doc)
        if job.status == IngestJobStatus.SUCCEEDED and job.result is None:
            payload["result"] = self._ingest_result_from_doc(doc, catalog)
        return payload

    def document_ingest_list_jobs(
        self, *, doc_id: str, limit: int = 20
    ) -> dict[str, Any]:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message="Document not found for ingest job listing",
        )
        jobs = catalog.list_ingest_jobs(doc.doc_id, limit=limit)
        return {
            "doc_id": doc.doc_id,
            "jobs": [self._serialize_ingest_job(job) for job in jobs],
        }

    def _page_ranges_overlap(
        self,
        chunk_range: list[int] | None,
        node_start: int | None,
        node_end: int | None,
    ) -> bool:
        return page_ranges_overlap(chunk_range, node_start, node_end)

    @staticmethod
    def _normalize_section_key(value: str) -> str:
        return normalize_section_key(value)

    def _normalize_section_path(self, section_path: list[str]) -> list[str]:
        return normalize_section_path(section_path)

    def _section_path_prefix_matches(
        self, section_path: list[str], node_path: list[str]
    ) -> bool:
        return section_path_prefix_matches(section_path, node_path)

    def _section_path_leaf_matches(
        self, section_path: list[str], node_path: list[str]
    ) -> bool:
        return section_path_leaf_matches(section_path, node_path)

    def _find_outline_node(
        self,
        nodes: list[OutlineNode],
        node_id: str,
        ancestry: list[str] | None = None,
    ) -> tuple[OutlineNode, list[str]] | None:
        resolved = find_outline_node(nodes, node_id, ancestry)
        if resolved is None:
            return None
        return resolved.node, resolved.path

    def _matches_outline_node(
        self,
        *,
        page_range: list[int] | None,
        section_path: list[str],
        spine_id: str | None,
        node: OutlineNode,
        node_path: list[str],
    ) -> bool:
        return matches_outline_node(
            page_range=page_range,
            section_path=section_path,
            spine_id=spine_id,
            node=node,
            node_path=node_path,
        )

    def _resolve_outline_node(
        self, doc_id: str, node_id: str
    ) -> tuple[DocumentRecord, OutlineNode, list[str], CatalogStore]:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Unknown doc_id: {doc_id}",
        )
        resolved = self._find_outline_node(doc.outline, node_id)
        if resolved is None:
            raise AppError(
                ErrorCode.READ_LOCATOR_NOT_FOUND,
                f"Unknown outline node: {node_id}",
                details={
                    "doc_id": doc_id,
                    "node_id": node_id,
                    "hint": "Call get_outline first and use a valid node id.",
                },
            )
        node, node_path = resolved
        return doc, node, node_path, catalog

    def _chunks_for_outline_node(
        self,
        *,
        catalog: CatalogStore,
        doc_id: str,
        node: OutlineNode,
        node_path: list[str],
    ) -> list[ChunkRecord]:
        all_chunks = catalog.list_chunks(doc_id)
        if not all_chunks:
            return []

        matched = [
            chunk
            for chunk in all_chunks
            if self._matches_outline_node(
                page_range=chunk.locator.page_range,
                section_path=chunk.section_path,
                spine_id=(chunk.locator.epub_locator or {}).get("spine_id"),
                node=node,
                node_path=node_path,
            )
        ]
        return matched

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

    def _catalogs_for_search(self, doc_ids: list[str] | None) -> list[CatalogStore]:
        if not doc_ids:
            return list(dict.fromkeys(self._catalogs.values()))

        catalogs: dict[str, CatalogStore] = {}
        for doc_id in doc_ids:
            _, catalog = self._require_doc(
                doc_id,
                error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
                message="Document not found for search.",
            )
            catalogs[self._catalog_key(catalog)] = catalog
        return list(catalogs.values())

    def _local_search(
        self, query: str, doc_ids: list[str] | None, top_k: int
    ) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for catalog in self._catalogs_for_search(doc_ids):
            hits.extend(catalog.search_local(query=query, doc_ids=doc_ids, top_k=top_k))
        hits.sort(key=lambda item: item.get("local_score", 0), reverse=True)
        return hits[:top_k]

    def _search_local_index(
        self, query: str, doc_ids: list[str] | None, top_k: int
    ) -> dict[str, Any]:
        local_hits = self._local_search(query=query, doc_ids=doc_ids, top_k=top_k)
        return {
            "hits": local_hits[:top_k],
            "retrieval": {
                "mode": "sqlite_fts5",
                "local_backend": "sqlite_fts5",
                "local_hits_count": len(local_hits),
                "external_vector_backend": None,
            },
        }

    def search(
        self, query: str, doc_ids: list[str] | None, top_k: int
    ) -> dict[str, Any]:
        return self._search_local_index(query=query, doc_ids=doc_ids, top_k=top_k)

    @staticmethod
    def _hit_to_graph_lookup(hit: dict[str, Any]) -> str:
        source_type = str(hit.get("source_type") or "chunk")
        source_id = str(hit.get("source_id") or hit.get("chunk_id") or "")
        if source_type == "pdf_table":
            return source_id
        if source_type == "pdf_figure":
            return source_id
        return source_id

    def _require_document_mode(
        self,
        *,
        doc_id: str,
        expected_type: DocumentType,
        expected_profile: Profile,
        operation_name: str,
    ) -> tuple[DocumentRecord, CatalogStore]:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Document not found for {operation_name}.",
        )
        if doc.type != expected_type or doc.profile != expected_profile:
            raise AppError(
                ErrorCode.INGEST_UNSUPPORTED_TYPE,
                f"{operation_name} requires a {expected_type.value} {expected_profile.value}.",
                details={
                    "doc_id": doc_id,
                    "actual_type": doc.type,
                    "actual_profile": doc.profile,
                    "expected_type": expected_type,
                    "expected_profile": expected_profile,
                },
            )
        return doc, catalog

    def _document_explore(
        self,
        *,
        doc_id: str,
        query: str,
        top_k: int,
        operation_name: str,
    ) -> dict[str, Any]:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Document not found for {operation_name}.",
        )
        selected_limit = max(1, min(top_k, 20))
        search_limit = max(selected_limit * 4, 20)
        search_result = self._search_local_index(
            query=query, doc_ids=[doc_id], top_k=search_limit
        )
        selected_nodes: list[dict[str, Any]] = []
        selected_node_ids: set[str] = set()
        neighbor_rows: list[dict[str, Any]] = []
        ambiguity_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for hit in search_result["hits"]:
            if len(selected_nodes) >= selected_limit:
                break
            lookup_id = self._hit_to_graph_lookup(hit)
            if not lookup_id:
                continue
            node = catalog.get_graph_node(doc_id=doc_id, node_id=lookup_id)
            if node is None or node["node_id"] in selected_node_ids:
                continue
            selected_node_ids.add(node["node_id"])
            ambiguity_key = normalize_section_key(
                str(node.get("title") or node.get("stable_ref") or "")
            )
            if ambiguity_key:
                ambiguity_buckets[ambiguity_key].append(
                    {
                        "node_id": node["node_id"],
                        "kind": node["kind"],
                        "title": node.get("title"),
                        "stable_ref": node.get("stable_ref"),
                    }
                )
            retrieval_sources = hit.get("retrieval_sources") or []
            primary_source = (
                retrieval_sources[0].get("source")
                if retrieval_sources and isinstance(retrieval_sources[0], dict)
                else hit.get("retrieval_source") or "sqlite_fts5"
            )
            why_included = {
                "source": primary_source,
                "retrieval_sources": retrieval_sources,
                "snippet": hit.get("snippet"),
                "local_rank": hit.get("local_rank"),
                "source_type": hit.get("source_type"),
            }
            if hit.get("why_included"):
                why_included["structured_reason"] = hit["why_included"]
            if hit.get("additional_relevance"):
                why_included["additional_relevance"] = hit["additional_relevance"]
            selected_nodes.append(
                {
                    **node,
                    "why_included": why_included,
                }
            )
            for neighbor in catalog.list_graph_neighbors(
                doc_id=doc_id,
                node_id=node["node_id"],
                limit=20,
            ):
                neighbor_node_id = neighbor["node"]["node_id"]
                if neighbor_node_id in selected_node_ids:
                    continue
                neighbor_rows.append(neighbor)

        graph_stats = catalog.graph_stats(doc_id)
        pipeline_status = self._document_pipeline_status(doc)
        persisted_diagnostics = catalog.list_diagnostics(doc_id=doc_id, limit=50)
        diagnostics = (
            [
                {
                    "severity": "warning",
                    "code": "DOCUMENT_PIPELINE_STALE",
                    "message": pipeline_status.get("reason"),
                    "hint": pipeline_status.get("hint"),
                }
            ]
            if pipeline_status.get("is_stale")
            else []
        )
        diagnostics.extend(persisted_diagnostics)
        ambiguity_candidates = [
            {"match_key": key, "candidates": values}
            for key, values in sorted(ambiguity_buckets.items())
            if len(values) > 1
        ]
        return {
            "document": {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "type": doc.type,
                "profile": doc.profile,
                "status": doc.status,
                "path": doc.path,
            },
            "query": query,
            "hits": search_result["hits"][:search_limit],
            "selected_nodes": selected_nodes,
            "neighbor_nodes": neighbor_rows[: max(selected_limit * 3, 12)],
            "graph": graph_stats,
            "retrieval": {
                **search_result["retrieval"],
                "mode": "sqlite_fts5_plus_document_graph",
                "search_limit": search_limit,
                "selected_limit": selected_limit,
            },
            "pipeline_status": pipeline_status,
            "diagnostics": diagnostics,
            "ambiguity_candidates": ambiguity_candidates,
            "truncation": {
                "selected_nodes_truncated": len(search_result["hits"]) > selected_limit,
                "neighbor_nodes_truncated": len(neighbor_rows)
                > max(selected_limit * 3, 12),
                "next_call": {
                    "tool": operation_name,
                    "args": {
                        "doc_id": doc_id,
                        "query": query,
                        "top_k": min(selected_limit * 2, 20),
                    },
                }
                if len(search_result["hits"]) > selected_limit
                else None,
            },
            "suggested_next_calls": [
                {
                    "tool": "document_node",
                    "reason": "Read one selected node precisely with graph neighbors.",
                    "args": {"doc_id": doc_id, "node_id": "<selected_nodes[].node_id>"},
                },
                {
                    "tool": "read_outline_node",
                    "reason": "Read a full chapter/section when an outline node is selected.",
                    "args": {"doc_id": doc_id, "node_id": "<outline node id>"},
                },
            ],
        }

    def document_explore(
        self, doc_id: str, query: str, top_k: int = 8
    ) -> dict[str, Any]:
        return self._document_explore(
            doc_id=doc_id,
            query=query,
            top_k=top_k,
            operation_name="document_explore",
        )

    def _library_document_ranking(
        self,
        *,
        query: str,
        bucket: dict[str, Any],
    ) -> dict[str, Any]:
        query_label = CatalogStore._normalize_search_label(query)
        query_tokens = set(query_label.split())
        title_label = CatalogStore._normalize_search_label(
            str(bucket.get("title") or "")
        )
        path_label = CatalogStore._normalize_search_label(str(bucket.get("path") or ""))
        matched_title_tokens = sorted(query_tokens & set(title_label.split()))
        matched_path_tokens = sorted(query_tokens & set(path_label.split()))
        hits_count = int(bucket.get("hits_count") or 0)
        node_kinds = bucket.get("_node_kinds") or set()
        section_keys = {
            key for key in (bucket.get("_section_keys") or set()) if str(key).strip()
        }
        retrieval_sources = bucket.get("_retrieval_sources") or set()
        best_local_score = float(bucket.get("best_local_score") or 0)
        score = best_local_score
        score += min(hits_count, 12) * 3.0
        score += min(len(node_kinds), 6) * 4.0
        score += min(len(section_keys), 8) * 2.0
        score += min(len(retrieval_sources), 4) * 3.0
        if matched_title_tokens:
            score += 24.0 * (len(matched_title_tokens) / max(len(query_tokens), 1))
        if matched_path_tokens:
            score += 8.0 * (len(matched_path_tokens) / max(len(query_tokens), 1))
        if bucket.get("status") == DocumentStatus.READY:
            score += 5.0

        return {
            "score": round(score, 4),
            "signals": {
                "best_local_score": best_local_score,
                "hits_count": hits_count,
                "node_kinds": sorted(str(item) for item in node_kinds),
                "section_paths_count": len(section_keys),
                "retrieval_sources": sorted(str(item) for item in retrieval_sources),
                "matched_title_tokens": matched_title_tokens,
                "matched_path_tokens": matched_path_tokens,
                "ready_boost_applied": bucket.get("status") == DocumentStatus.READY,
            },
        }

    def _library_document_bucket_payload(
        self, bucket: dict[str, Any]
    ) -> dict[str, Any]:
        return {key: value for key, value in bucket.items() if not key.startswith("_")}

    def library_explore(
        self, *, root: str | None = None, query: str, top_k: int = 12
    ) -> dict[str, Any]:
        root_path = self._resolve_library_root(root)
        catalogs = self._discover_sidecar_catalogs(root_path)
        selected_limit = max(1, min(top_k, 50))
        search_limit = max(selected_limit * 4, 40)
        doc_catalogs: dict[str, tuple[DocumentRecord, CatalogStore]] = {}
        unready_documents: list[dict[str, Any]] = []
        searched_documents_count = 0
        all_hits: list[dict[str, Any]] = []

        for catalog in catalogs:
            docs = catalog.list_documents()
            searched_documents_count += len(docs)
            for doc in docs:
                doc_catalogs[doc.doc_id] = (doc, catalog)
                if doc.status != DocumentStatus.READY:
                    suggested_next_call = self._reingest_call(doc)
                    unready_documents.append(
                        {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "type": doc.type,
                            "profile": doc.profile,
                            "status": doc.status,
                            "path": doc.path,
                            "suggested_next_call": suggested_next_call,
                        }
                    )
            all_hits.extend(
                catalog.search_local(query=query, doc_ids=None, top_k=search_limit)
            )

        document_buckets: dict[str, dict[str, Any]] = {}
        for hit in all_hits:
            doc_id = str(hit.get("doc_id") or "")
            doc_catalog = doc_catalogs.get(doc_id)
            if doc_catalog is None:
                continue
            doc, catalog = doc_catalog
            bucket = document_buckets.setdefault(
                doc_id,
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "type": doc.type,
                    "profile": doc.profile,
                    "status": doc.status,
                    "path": doc.path,
                    "hits_count": 0,
                    "best_local_score": hit.get("local_score"),
                    "sidecar_path": str(catalog.db_path.parent),
                    "_node_kinds": set(),
                    "_section_keys": set(),
                    "_retrieval_sources": set(),
                },
            )
            bucket["hits_count"] += 1
            bucket["best_local_score"] = max(
                bucket.get("best_local_score") or 0,
                hit.get("local_score") or 0,
            )
            source_type = str(hit.get("source_type") or "")
            if source_type:
                bucket["_node_kinds"].add(source_type)
            section_path = hit.get("section_path") or []
            if isinstance(section_path, list):
                bucket["_section_keys"].add(
                    " / ".join(str(item) for item in section_path)
                )
            for source in hit.get("retrieval_sources") or []:
                if isinstance(source, dict) and source.get("source"):
                    bucket["_retrieval_sources"].add(str(source["source"]))

        for bucket in document_buckets.values():
            ranking = self._library_document_ranking(query=query, bucket=bucket)
            bucket["document_score"] = ranking["score"]
            bucket["ranking_signals"] = ranking["signals"]

        for hit in all_hits:
            bucket = document_buckets.get(str(hit.get("doc_id") or ""))
            if bucket is None:
                continue
            local_score = float(hit.get("local_score") or 0)
            document_score = float(bucket.get("document_score") or 0)
            hit["library_document_score"] = document_score
            hit["library_score"] = local_score + min(document_score, 200.0) * 0.08

        all_hits.sort(
            key=lambda item: (
                float(item.get("library_score") or item.get("local_score") or 0),
                float(item.get("local_score") or 0),
            ),
            reverse=True,
        )
        hits = [
            hit for hit in all_hits if str(hit.get("doc_id") or "") in doc_catalogs
        ][:search_limit]
        selected_results: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, str]] = set()
        diagnostics: list[dict[str, Any]] = []

        for hit in hits:
            doc_id = str(hit.get("doc_id") or "")
            doc_catalog = doc_catalogs.get(doc_id)
            if doc_catalog is None:
                continue
            doc, catalog = doc_catalog
            if len(selected_results) >= selected_limit:
                continue
            lookup_id = self._hit_to_graph_lookup(hit)
            if not lookup_id:
                continue
            node = catalog.get_graph_node(doc_id=doc_id, node_id=lookup_id)
            if node is None:
                continue
            selected_key = (doc_id, node["node_id"])
            if selected_key in selected_keys:
                continue
            selected_keys.add(selected_key)
            pipeline_status = self._document_pipeline_status(doc)
            if pipeline_status.get("is_stale"):
                diagnostics.append(
                    {
                        "severity": "warning",
                        "code": "DOCUMENT_PIPELINE_STALE",
                        "doc_id": doc_id,
                        "message": pipeline_status.get("reason"),
                        "hint": pipeline_status.get("hint"),
                    }
                )
            retrieval_sources = hit.get("retrieval_sources") or []
            selected_results.append(
                {
                    "document": {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "type": doc.type,
                        "profile": doc.profile,
                        "status": doc.status,
                        "path": doc.path,
                    },
                    "node": node,
                    "hit": hit,
                    "why_included": {
                        "source": hit.get("retrieval_source") or "sqlite_fts5",
                        "retrieval_sources": retrieval_sources,
                        "snippet": hit.get("snippet"),
                        "local_rank": hit.get("local_rank"),
                        "library_score": hit.get("library_score"),
                        "library_document_score": hit.get("library_document_score"),
                        "source_type": hit.get("source_type"),
                    },
                    "pipeline_status": pipeline_status,
                    "suggested_next_calls": [
                        {
                            "tool": "document_explore",
                            "reason": "Narrow follow-up exploration inside this document.",
                            "args": {
                                "doc_id": doc_id,
                                "query": query,
                                "top_k": min(top_k, 12),
                            },
                        },
                        {
                            "tool": "document_node",
                            "reason": "Read this selected graph node precisely.",
                            "args": {
                                "doc_id": doc_id,
                                "node_id": node["node_id"],
                            },
                        },
                    ],
                }
            )

        if unready_documents:
            diagnostics.append(
                {
                    "severity": "info",
                    "code": "DOCUMENTS_NOT_READY",
                    "message": "Some scanned documents under this root are not ready for library_explore search.",
                    "documents": unready_documents[:20],
                }
            )

        return {
            "root": str(root_path),
            "query": query,
            "documents": sorted(
                [
                    self._library_document_bucket_payload(bucket)
                    for bucket in document_buckets.values()
                ],
                key=lambda item: item.get("document_score") or 0,
                reverse=True,
            ),
            "selected_results": selected_results,
            "hits": hits,
            "retrieval": {
                "mode": "root_sidecar_sqlite_fts5_plus_document_graph",
                "sidecars_count": len(catalogs),
                "searched_documents_count": searched_documents_count,
                "hits_count": len(all_hits),
                "search_limit": search_limit,
                "selected_limit": selected_limit,
                "external_vector_backend": None,
            },
            "diagnostics": diagnostics,
            "truncation": {
                "selected_results_truncated": len(hits) > selected_limit,
                "next_call": {
                    "tool": "library_explore",
                    "args": {
                        "root": str(root_path),
                        "query": query,
                        "top_k": min(selected_limit * 2, 50),
                    },
                }
                if len(hits) > selected_limit
                else None,
            },
            "suggested_next_calls": [
                {
                    "tool": "document_explore",
                    "reason": "Explore one selected document in detail.",
                    "args": {
                        "doc_id": "<selected_results[].document.doc_id>",
                        "query": query,
                        "top_k": 8,
                    },
                },
                {
                    "tool": "document_node",
                    "reason": "Read a precise selected graph node.",
                    "args": {
                        "doc_id": "<selected_results[].document.doc_id>",
                        "node_id": "<selected_results[].node.node_id>",
                    },
                },
            ],
        }

    def _document_node(
        self,
        *,
        doc_id: str,
        node_id: str,
        operation_name: str,
    ) -> dict[str, Any]:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Document not found for {operation_name}.",
        )
        node = catalog.get_graph_node(doc_id=doc_id, node_id=node_id)
        if node is None:
            raise AppError(
                ErrorCode.READ_LOCATOR_NOT_FOUND,
                f"Document graph node not found: {node_id}",
                details={"doc_id": doc_id, "node_id": node_id},
            )

        read_result: dict[str, Any] | None = None
        locator = node["locator"]
        if node["kind"] == "chunk":
            read_result = self.read(
                locator=locator, before=1, after=1, out_format="markdown"
            )
        elif node["kind"] == "outline_node":
            outline_node_id = locator.get("node_id")
            if outline_node_id:
                read_result = self.read_outline_node(
                    doc_id=doc_id,
                    node_id=outline_node_id,
                    out_format="markdown",
                    max_chunks=80,
                )
        elif node["kind"] == "formula":
            formula_id = locator.get("formula_id")
            if formula_id:
                if doc.profile == Profile.PAPER:
                    read_result = self.pdf_read_formula(
                        doc_id=doc_id,
                        formula_id=formula_id,
                    )
                else:
                    read_result = self.pdf_read_formula(
                        doc_id=doc_id,
                        formula_id=formula_id,
                    )
        elif node["kind"] == "image":
            image_id = locator.get("image_id")
            if image_id:
                if doc.type == DocumentType.EPUB:
                    read_result = self.epub_read_image(doc_id=doc_id, image_id=image_id)
                else:
                    read_result = self.pdf_read_image(doc_id=doc_id, image_id=image_id)
        elif node["kind"] == "table":
            table_id = locator.get("table_id")
            if table_id:
                read_result = self.pdf_read_table(doc_id=doc_id, table_id=table_id)
        elif node["kind"] == "figure":
            figure_id = locator.get("figure_id")
            if figure_id:
                read_result = self.pdf_read_figure(doc_id=doc_id, figure_id=figure_id)
        elif node["kind"] == "page":
            page = locator.get("page")
            if isinstance(page, int) and doc.type == DocumentType.PDF:
                read_result = {
                    "page": page,
                    "rendered_page": self.render_pdf_page(
                        doc_id=doc_id,
                        page=page,
                        dpi=160,
                    ),
                    "hint": "Use neighboring chunk/formula/figure/table nodes for semantic reading context.",
                }
        elif node["kind"] == "artifact":
            metadata = node.get("metadata") or {}
            read_result = {
                "artifact_id": locator.get("artifact_id"),
                "file_path": metadata.get("file_path"),
                "media_type": metadata.get("media_type"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "source_ref": metadata.get("source_ref"),
                "hint": "Pass file_path to a multimodal LLM when visual inspection is needed.",
            }
        elif node["kind"] in {"reference", "citation"}:
            read_result = {
                "kind": node["kind"],
                "title": node.get("title"),
                "text": node.get("text"),
                "locator": locator,
                "metadata": node.get("metadata") or {},
            }

        return {
            "document": {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "type": doc.type,
                "profile": doc.profile,
                "status": doc.status,
            },
            "node": node,
            "neighbors": catalog.list_graph_neighbors(
                doc_id=doc_id,
                node_id=node["node_id"],
                limit=80,
            ),
            "read_result": read_result,
            "pipeline_status": self._document_pipeline_status(doc),
        }

    def document_node(self, doc_id: str, node_id: str) -> dict[str, Any]:
        return self._document_node(
            doc_id=doc_id,
            node_id=node_id,
            operation_name="document_node",
        )

    def search_in_outline_node(
        self,
        *,
        doc_id: str,
        node_id: str,
        query: str,
        top_k: int,
    ) -> dict[str, Any]:
        _, node, node_path, _ = self._resolve_outline_node(doc_id, node_id)
        expanded_top_k = max(top_k * 5, top_k)
        search_result = self._search_local_index(
            query=query,
            doc_ids=[doc_id],
            top_k=expanded_top_k,
        )
        filtered_hits: list[dict[str, Any]] = []
        for hit in search_result["hits"]:
            locator = hit.get("locator") or {}
            if self._matches_outline_node(
                page_range=locator.get("page_range"),
                section_path=locator.get("section_path") or [],
                spine_id=(locator.get("epub_locator") or {}).get("spine_id"),
                node=node,
                node_path=node_path,
            ):
                filtered_hits.append(hit)
                if len(filtered_hits) >= top_k:
                    break

        return {
            "node": node.model_dump(),
            "hits": filtered_hits[:top_k],
            "retrieval": search_result["retrieval"],
        }

    def read(
        self, locator: dict[str, Any], before: int, after: int, out_format: str
    ) -> dict[str, Any]:
        parsed_locator = Locator(**locator)
        _, catalog = self._require_doc(
            parsed_locator.doc_id,
            error_code=ErrorCode.READ_LOCATOR_NOT_FOUND,
            message="Chunk not found for locator.",
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
        doc, node, node_path, catalog = self._resolve_outline_node(doc_id, node_id)
        scoped_chunks = self._chunks_for_outline_node(
            catalog=catalog,
            doc_id=doc_id,
            node=node,
            node_path=node_path,
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
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Unknown doc_id: {doc_id}",
        )
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
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Unknown doc_id: {doc_id}",
        )
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

    def _formula_payload(self, formula: FormulaRecord) -> dict[str, Any]:
        return {
            "formula_id": formula.formula_id,
            "chunk_id": formula.chunk_id,
            "section_path": formula.section_path,
            "page": formula.page,
            "bbox": formula.bbox,
            "latex": formula.latex,
            "source": formula.source,
            "confidence": formula.confidence,
            "status": formula.status,
        }

    def _image_semantic_enrichment(
        self,
        image: ImageRecord,
        *,
        context_text: str | None = None,
    ) -> dict[str, Any]:
        signals = {
            "section_path": image.section_path,
            "alt": image.alt,
            "caption": image.caption,
            "anchor": image.anchor,
            "href": image.href,
            "context_excerpt": context_text[:500] if context_text else None,
        }
        summary_parts = [
            part
            for part in (
                image.caption,
                image.alt,
                " / ".join(image.section_path),
                context_text[:160] if context_text else None,
            )
            if part
        ]
        return {
            "summary": " | ".join(summary_parts) if summary_parts else None,
            "signals": signals,
            "diagnostics": {
                "mode": "metadata_and_nearby_text_only",
                "ocr_performed": False,
                "visual_embedding_performed": False,
                "original_image_untouched": True,
            },
        }

    def _caption_evidence_payload(
        self,
        *,
        caption: str | None,
        default_source: str,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        caption_observation = (
            observation.get("caption")
            if isinstance(observation, dict)
            and isinstance(observation.get("caption"), dict)
            else None
        )
        if caption_observation is not None:
            needs_review = bool(caption_observation.get("needs_review"))
            return {
                "status": "needs_review"
                if needs_review
                else "resolved"
                if caption_observation.get("text")
                else "missing",
                "text": caption_observation.get("text"),
                "source": caption_observation.get("source") or default_source,
                "confidence": caption_observation.get("confidence"),
                "needs_review": needs_review,
                "candidates": caption_observation.get("candidates") or [],
            }
        return {
            "status": "resolved" if caption else "missing",
            "text": caption,
            "source": default_source if caption else None,
            "confidence": None,
            "needs_review": not bool(caption),
            "candidates": [],
        }

    def _image_payload(
        self,
        image: ImageRecord,
        *,
        context_text: str | None = None,
    ) -> dict[str, Any]:
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
            "caption_evidence": self._caption_evidence_payload(
                caption=image.caption,
                default_source="epub_figcaption"
                if image.source in {"ebooklib", "epub"}
                else "pdf_image_caption_nearby_text",
            ),
            "media_type": image.media_type,
            "image_path": image.file_path,
            "width": image.width,
            "height": image.height,
            "semantic": self._image_semantic_enrichment(
                image,
                context_text=context_text,
            ),
        }

    def _table_payload(
        self,
        table: PdfTableRecord,
        *,
        include_rows: bool = False,
        include_segments: bool = False,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "table_id": table.table_id,
            "section_path": table.section_path,
            "page_range": table.page_range,
            "bbox": table.bbox,
            "caption": table.caption,
            "caption_evidence": self._caption_evidence_payload(
                caption=table.caption,
                default_source="pdf_table_extractor",
                observation=observation,
            ),
            "headers": table.headers,
            "row_count": len(table.rows),
            "column_count": len(table.headers),
            "image_path": table.file_path,
            "width": table.width,
            "height": table.height,
            "merged": table.merged,
            "merge_confidence": table.merge_confidence,
            "source": table.source,
            "status": table.status,
        }
        if include_rows:
            payload["rows"] = table.rows
            payload["markdown"] = table.markdown
            payload["html"] = table.html
        if include_segments:
            payload["segments"] = [
                segment.model_dump(mode="json") for segment in table.segments
            ]
        if observation is not None:
            payload["issues"] = observation.get("issues") or []
            if observation.get("merge") is not None:
                payload["merge_diagnostics"] = observation["merge"]
        return payload

    def _figure_payload(
        self,
        figure: PdfFigureRecord,
        *,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "figure_id": figure.figure_id,
            "section_path": figure.section_path,
            "page": figure.page,
            "bbox": figure.bbox,
            "caption": figure.caption,
            "caption_evidence": self._caption_evidence_payload(
                caption=figure.caption,
                default_source="pdf_figure_extractor",
                observation=observation,
            ),
            "kind": figure.kind,
            "image_path": figure.file_path,
            "width": figure.width,
            "height": figure.height,
            "source": figure.source,
            "status": figure.status,
        }
        if observation is not None:
            payload["issues"] = observation.get("issues") or []
        return payload

    def _pdf_images_dir(self, workspace_dir: Path) -> Path:
        return workspace_dir / "assets" / "pdf-images"

    def _pdf_tables_dir(self, workspace_dir: Path) -> Path:
        return workspace_dir / "assets" / "pdf-tables"

    def _pdf_figures_dir(self, workspace_dir: Path) -> Path:
        return workspace_dir / "assets" / "pdf-figures"

    def _pdf_images_manifest_path(self, workspace_dir: Path) -> Path:
        return self._pdf_images_dir(workspace_dir) / ".extracted.json"

    def _pdf_visuals_manifest_path(self, workspace_dir: Path) -> Path:
        return workspace_dir / "assets" / "pdf-visuals" / ".extracted.json"

    def _default_pdf_visuals_diagnostics(
        self,
        *,
        tables: list[PdfTableRecord],
        figures: list[PdfFigureRecord],
    ) -> dict[str, Any]:
        return {
            "extractor": "docling-visuals",
            "summary": {
                "tables_detected_raw": len(tables),
                "tables_returned": len(tables),
                "figures_returned": len(figures),
                "merged_tables_count": sum(1 for table in tables if table.merged),
                "issues_count": 0,
                "warning_count": 0,
                "error_count": 0,
                "info_count": 0,
            },
            "issues": [],
            "merge_decisions": [],
            "tables": {},
            "figures": {},
        }

    def _load_pdf_visuals_manifest(
        self,
        manifest_path: Path,
        *,
        tables: list[PdfTableRecord],
        figures: list[PdfFigureRecord],
    ) -> dict[str, Any]:
        if not manifest_path.exists():
            return self._default_pdf_visuals_diagnostics(tables=tables, figures=figures)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return self._default_pdf_visuals_diagnostics(tables=tables, figures=figures)
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return self._default_pdf_visuals_diagnostics(tables=tables, figures=figures)
        return diagnostics

    def _persist_pdf_visuals_diagnostics_summary(
        self,
        *,
        doc: DocumentRecord,
        catalog: CatalogStore,
        diagnostics: dict[str, Any],
    ) -> None:
        metadata = {
            **doc.metadata,
            "pdf_visuals_diagnostics": {
                "summary": diagnostics.get("summary") or {},
                "issues_count": len(diagnostics.get("issues") or []),
                "last_extracted_at": self._now_iso(),
            },
        }
        catalog.save_document_parse_output(
            doc_id=doc.doc_id,
            title=doc.title or Path(doc.path).stem,
            parser_chain=doc.parser_chain,
            metadata=metadata,
            outline=doc.outline,
            overall_confidence=doc.overall_confidence,
            status=doc.status,
        )

    def _pdf_visuals_payload(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": diagnostics.get("summary") or {},
            "issues": diagnostics.get("issues") or [],
            "merge_decisions": diagnostics.get("merge_decisions") or [],
        }

    def _table_observation(
        self,
        diagnostics: dict[str, Any],
        table_id: str,
    ) -> dict[str, Any]:
        tables = diagnostics.get("tables") or {}
        observation = tables.get(table_id)
        return observation if isinstance(observation, dict) else {"issues": []}

    def _figure_observation(
        self,
        diagnostics: dict[str, Any],
        figure_id: str,
    ) -> dict[str, Any]:
        figures = diagnostics.get("figures") or {}
        observation = figures.get(figure_id)
        return observation if isinstance(observation, dict) else {"issues": []}

    def _load_pdf_images_evidence(
        self,
        *,
        doc: DocumentRecord,
        catalog: CatalogStore,
    ) -> list[ImageRecord]:
        return catalog.list_images(doc.doc_id)

    def _load_pdf_visuals_evidence(
        self,
        *,
        doc: DocumentRecord,
        catalog: CatalogStore,
    ) -> tuple[list[PdfTableRecord], list[PdfFigureRecord], dict[str, Any]]:
        workspace_dir = self._doc_workspace_dir(doc, catalog)
        manifest_path = self._pdf_visuals_manifest_path(workspace_dir)

        existing_tables = catalog.list_pdf_tables(doc.doc_id)
        existing_figures = catalog.list_pdf_figures(doc.doc_id)
        diagnostics = self._load_pdf_visuals_manifest(
            manifest_path,
            tables=existing_tables,
            figures=existing_figures,
        )
        return existing_tables, existing_figures, diagnostics

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
            _, node, node_path, _ = self._resolve_outline_node(doc_id, node_id)
            node_payload = node.model_dump()
            filtered: list[ImageRecord] = []
            for image in images:
                if self._matches_outline_node(
                    page_range=[image.page, image.page]
                    if image.page is not None
                    else None,
                    section_path=image.section_path,
                    spine_id=image.spine_id,
                    node=node,
                    node_path=node_path,
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
            "image": self._image_payload(
                image,
                context_text=matched_chunk.text if matched_chunk else None,
            ),
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
        images = self._load_pdf_images_evidence(doc=doc, catalog=catalog)

        node_payload: dict[str, Any] | None = None
        if node_id:
            _, node, node_path, _ = self._resolve_outline_node(doc_id, node_id)
            node_payload = node.model_dump()
            filtered: list[ImageRecord] = []
            for image in images:
                if self._matches_outline_node(
                    page_range=[image.page, image.page]
                    if image.page is not None
                    else None,
                    section_path=image.section_path,
                    spine_id=image.spine_id,
                    node=node,
                    node_path=node_path,
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
            raise AppError(
                ErrorCode.READ_IMAGE_NOT_FOUND,
                f"Image evidence file does not exist: {image.file_path}",
                details={
                    "doc_id": doc_id,
                    "image_id": image_id,
                    "path": image.file_path,
                    "hint": (
                        "PDF image extraction is eager and read-only at read time. "
                        "Re-run document_ingest "
                        "with force=true to regenerate sidecar evidence."
                    ),
                },
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
            "image": self._image_payload(
                image,
                context_text=matched_chunk.text if matched_chunk else None,
            ),
            "context": {
                "text": matched_chunk.text,
                "locator": matched_chunk.locator.model_dump(),
            }
            if matched_chunk
            else None,
        }

    def _table_matches_outline_node(
        self,
        *,
        table: PdfTableRecord,
        node: OutlineNode,
        node_path: list[str],
    ) -> bool:
        return self._matches_outline_node(
            page_range=table.page_range,
            section_path=table.section_path,
            spine_id=None,
            node=node,
            node_path=node_path,
        )

    def _find_table_context_chunk(
        self,
        *,
        catalog: CatalogStore,
        doc_id: str,
        table: PdfTableRecord,
    ) -> ChunkRecord | None:
        chunks = catalog.list_chunks(doc_id)
        if table.page_range is not None:
            for chunk in chunks:
                if self._page_ranges_overlap(
                    chunk.locator.page_range,
                    table.page_range[0],
                    table.page_range[1],
                ):
                    return chunk

        if table.section_path:
            target = table.section_path[-1].strip().lower()
            if target:
                for chunk in chunks:
                    if any(
                        target in part.strip().lower() for part in chunk.section_path
                    ):
                        return chunk

        return None

    def pdf_list_tables(
        self,
        *,
        doc_id: str,
        node_id: str | None,
        limit: int,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        tables, _, diagnostics = self._load_pdf_visuals_evidence(
            doc=doc, catalog=catalog
        )

        node_payload: dict[str, Any] | None = None
        if node_id:
            _, node, node_path, _ = self._resolve_outline_node(doc_id, node_id)
            node_payload = node.model_dump()
            tables = [
                table
                for table in tables
                if self._table_matches_outline_node(
                    table=table,
                    node=node,
                    node_path=node_path,
                )
            ]

        max_items = max(0, min(limit, 500))
        truncated = len(tables) > max_items
        if truncated:
            tables = tables[:max_items]

        return {
            "doc_title": doc.title,
            "node": node_payload,
            "tables": [
                self._table_payload(
                    table,
                    observation=self._table_observation(diagnostics, table.table_id),
                )
                for table in tables
            ],
            "tables_count": len(tables),
            "truncated": truncated,
            "diagnostics": self._pdf_visuals_payload(diagnostics),
        }

    def pdf_read_table(self, *, doc_id: str, table_id: str) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        _, _, diagnostics = self._load_pdf_visuals_evidence(doc=doc, catalog=catalog)
        table = catalog.get_pdf_table(table_id)
        if table is None or table.doc_id != doc_id:
            raise AppError(
                ErrorCode.READ_TABLE_NOT_FOUND,
                f"Unknown table_id: {table_id}",
                details={
                    "doc_id": doc_id,
                    "table_id": table_id,
                    "hint": "Call pdf_list_tables to get valid table ids.",
                },
            )

        image_path = Path(table.file_path)
        if not image_path.exists():
            raise AppError(
                ErrorCode.READ_TABLE_NOT_FOUND,
                f"Table evidence file does not exist: {table.file_path}",
                details={
                    "doc_id": doc_id,
                    "table_id": table_id,
                    "path": table.file_path,
                    "hint": (
                        "PDF table extraction is eager and read-only at read time. "
                        "Re-run document_ingest "
                        "with force=true to regenerate sidecar evidence."
                    ),
                },
            )

        matched_chunk = self._find_table_context_chunk(
            catalog=catalog,
            doc_id=doc_id,
            table=table,
        )
        return {
            "doc_title": doc.title,
            "table": self._table_payload(
                table,
                include_rows=True,
                include_segments=True,
                observation=self._table_observation(diagnostics, table.table_id),
            ),
            "context": {
                "text": matched_chunk.text,
                "locator": matched_chunk.locator.model_dump(),
            }
            if matched_chunk
            else None,
            "diagnostics": {
                "document": self._pdf_visuals_payload(diagnostics),
                "table": self._table_observation(diagnostics, table.table_id),
            },
        }

    def _find_figure_context_chunk(
        self,
        *,
        catalog: CatalogStore,
        doc_id: str,
        figure: PdfFigureRecord,
    ) -> ChunkRecord | None:
        chunks = catalog.list_chunks(doc_id)
        if figure.page is not None:
            for chunk in chunks:
                if self._page_ranges_overlap(
                    chunk.locator.page_range,
                    figure.page,
                    figure.page,
                ):
                    return chunk

        if figure.section_path:
            target = figure.section_path[-1].strip().lower()
            if target:
                for chunk in chunks:
                    if any(
                        target in part.strip().lower() for part in chunk.section_path
                    ):
                        return chunk

        return None

    def pdf_list_figures(
        self,
        *,
        doc_id: str,
        node_id: str | None,
        limit: int,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        _, figures, diagnostics = self._load_pdf_visuals_evidence(
            doc=doc, catalog=catalog
        )

        node_payload: dict[str, Any] | None = None
        if node_id:
            _, node, node_path, _ = self._resolve_outline_node(doc_id, node_id)
            node_payload = node.model_dump()
            figures = [
                figure
                for figure in figures
                if self._matches_outline_node(
                    page_range=[figure.page, figure.page]
                    if figure.page is not None
                    else None,
                    section_path=figure.section_path,
                    spine_id=None,
                    node=node,
                    node_path=node_path,
                )
            ]

        max_items = max(0, min(limit, 500))
        truncated = len(figures) > max_items
        if truncated:
            figures = figures[:max_items]

        return {
            "doc_title": doc.title,
            "node": node_payload,
            "figures": [
                self._figure_payload(
                    figure,
                    observation=self._figure_observation(diagnostics, figure.figure_id),
                )
                for figure in figures
            ],
            "figures_count": len(figures),
            "truncated": truncated,
            "diagnostics": self._pdf_visuals_payload(diagnostics),
        }

    def pdf_read_figure(self, *, doc_id: str, figure_id: str) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        _, _, diagnostics = self._load_pdf_visuals_evidence(doc=doc, catalog=catalog)
        figure = catalog.get_pdf_figure(figure_id)
        if figure is None or figure.doc_id != doc_id:
            raise AppError(
                ErrorCode.READ_FIGURE_NOT_FOUND,
                f"Unknown figure_id: {figure_id}",
                details={
                    "doc_id": doc_id,
                    "figure_id": figure_id,
                    "hint": "Call pdf_list_figures to get valid figure ids.",
                },
            )

        image_path = Path(figure.file_path)
        if not image_path.exists():
            raise AppError(
                ErrorCode.READ_FIGURE_NOT_FOUND,
                f"Figure evidence file does not exist: {figure.file_path}",
                details={
                    "doc_id": doc_id,
                    "figure_id": figure_id,
                    "path": figure.file_path,
                    "hint": (
                        "PDF figure extraction is eager and read-only at read time. "
                        "Re-run document_ingest "
                        "with force=true to regenerate sidecar evidence."
                    ),
                },
            )

        matched_chunk = self._find_figure_context_chunk(
            catalog=catalog,
            doc_id=doc_id,
            figure=figure,
        )
        return {
            "doc_title": doc.title,
            "figure": self._figure_payload(
                figure,
                observation=self._figure_observation(diagnostics, figure.figure_id),
            ),
            "context": {
                "text": matched_chunk.text,
                "locator": matched_chunk.locator.model_dump(),
            }
            if matched_chunk
            else None,
            "diagnostics": {
                "document": self._pdf_visuals_payload(diagnostics),
                "figure": self._figure_observation(diagnostics, figure.figure_id),
            },
        }

    def _formula_matches_outline_node(
        self,
        *,
        formula: FormulaRecord,
        node: OutlineNode,
        node_path: list[str],
    ) -> bool:
        return self._matches_outline_node(
            page_range=[formula.page, formula.page]
            if formula.page is not None
            else None,
            section_path=formula.section_path,
            spine_id=None,
            node=node,
            node_path=node_path,
        )

    def _find_formula_context_chunk(
        self,
        *,
        catalog: CatalogStore,
        doc_id: str,
        formula: FormulaRecord,
    ) -> ChunkRecord | None:
        if formula.chunk_id:
            chunk = catalog.get_chunk(doc_id, formula.chunk_id)
            if chunk is not None:
                return chunk

        chunks = catalog.list_chunks(doc_id)
        if formula.page is not None:
            for chunk in chunks:
                if self._page_ranges_overlap(
                    chunk.locator.page_range, formula.page, formula.page
                ):
                    return chunk

        if formula.section_path:
            target = formula.section_path[-1].strip().lower()
            if target:
                for chunk in chunks:
                    if any(
                        target in part.strip().lower() for part in chunk.section_path
                    ):
                        return chunk

        return None

    def _list_pdf_formulas(
        self,
        *,
        doc_id: str,
        node_id: str | None,
        limit: int,
        status: str | None,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        formulas = catalog.list_formulas(doc_id)

        normalized_status = (status or "").strip().lower()
        if normalized_status:
            allowed_statuses = {"resolved", "fallback_text", "unresolved"}
            if normalized_status not in allowed_statuses:
                raise AppError(
                    ErrorCode.READ_FORMULA_NOT_FOUND,
                    f"Unsupported formula status filter: {status}",
                    details={
                        "allowed_statuses": sorted(allowed_statuses),
                        "received_status": status,
                    },
                )
            formulas = [
                item
                for item in formulas
                if item.status.strip().lower() == normalized_status
            ]

        node_payload: dict[str, Any] | None = None
        if node_id:
            _, node, node_path, _ = self._resolve_outline_node(doc_id, node_id)
            node_payload = node.model_dump()
            formulas = [
                item
                for item in formulas
                if self._formula_matches_outline_node(
                    formula=item, node=node, node_path=node_path
                )
            ]

        max_items = max(0, min(limit, 500))
        truncated = len(formulas) > max_items
        if truncated:
            formulas = formulas[:max_items]

        return {
            "doc_title": doc.title,
            "profile": doc.profile,
            "node": node_payload,
            "formulas": [self._formula_payload(item) for item in formulas],
            "formulas_count": len(formulas),
            "truncated": truncated,
        }

    def _read_pdf_formula(
        self,
        *,
        doc_id: str,
        formula_id: str,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        formula = catalog.get_formula(formula_id)
        if formula is None or formula.doc_id != doc_id:
            raise AppError(
                ErrorCode.READ_FORMULA_NOT_FOUND,
                f"Unknown formula_id: {formula_id}",
                details={
                    "doc_id": doc_id,
                    "formula_id": formula_id,
                    "hint": "Call pdf_list_formulas first.",
                },
            )

        context_chunk = self._find_formula_context_chunk(
            catalog=catalog, doc_id=doc_id, formula=formula
        )

        workspace_dir = self._doc_workspace_dir(doc, catalog)
        evidence_path = (
            workspace_dir / "evidence" / "formulas" / f"{formula.formula_id}.png"
        )
        evidence: dict[str, Any] | None = None

        render_page = formula.page
        if render_page is None and context_chunk and context_chunk.locator.page_range:
            render_page = context_chunk.locator.page_range[0]

        if render_page is not None:
            if formula.bbox is not None:
                try:
                    width, height = render_pdf_region(
                        doc.path,
                        evidence_path,
                        page=render_page,
                        bbox=formula.bbox,
                    )
                    evidence = {
                        "type": "formula_region",
                        "image_path": str(evidence_path),
                        "width": width,
                        "height": height,
                        "page": render_page,
                        "bbox": formula.bbox,
                    }
                except AppError as exc:
                    width, height = render_pdf_page(
                        doc.path,
                        evidence_path,
                        page=render_page,
                    )
                    evidence = {
                        "type": "page_fallback",
                        "image_path": str(evidence_path),
                        "width": width,
                        "height": height,
                        "page": render_page,
                        "bbox": None,
                        "fallback_reason": exc.message,
                    }
            else:
                width, height = render_pdf_page(
                    doc.path,
                    evidence_path,
                    page=render_page,
                )
                evidence = {
                    "type": "page",
                    "image_path": str(evidence_path),
                    "width": width,
                    "height": height,
                    "page": render_page,
                    "bbox": None,
                }

        if evidence is not None and evidence_path.exists():
            artifact_node_id = catalog.upsert_formula_evidence_artifact(
                doc_id=doc_id,
                formula=formula,
                file_path=str(evidence_path),
                evidence_type=str(evidence["type"]),
                width=evidence.get("width"),
                height=evidence.get("height"),
                page=evidence.get("page"),
                bbox=evidence.get("bbox"),
            )
            if artifact_node_id is not None:
                evidence["artifact_node_id"] = artifact_node_id

        return {
            "doc_title": doc.title,
            "profile": doc.profile,
            "formula": self._formula_payload(formula),
            "context": {
                "text": context_chunk.text,
                "locator": context_chunk.locator.model_dump(),
            }
            if context_chunk
            else None,
            "evidence": evidence,
        }

    def pdf_list_formulas(
        self,
        *,
        doc_id: str,
        node_id: str | None,
        limit: int,
        status: str | None,
    ) -> dict[str, Any]:
        return self._list_pdf_formulas(
            doc_id=doc_id,
            node_id=node_id,
            limit=limit,
            status=status,
        )

    def pdf_read_formula(self, *, doc_id: str, formula_id: str) -> dict[str, Any]:
        return self._read_pdf_formula(
            doc_id=doc_id,
            formula_id=formula_id,
        )

    def get_outline(self, doc_id: str) -> dict[str, Any]:
        doc, _ = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Unknown doc_id: {doc_id}",
        )
        return {
            "title": doc.title,
            "nodes": [node.model_dump() for node in doc.outline],
            "pipeline_status": self._document_pipeline_status(doc),
        }

    def render_pdf_page(self, doc_id: str, page: int, dpi: int) -> dict[str, Any]:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message=f"Unknown doc_id: {doc_id}",
        )
        if doc.type != DocumentType.PDF:
            raise AppError(
                ErrorCode.RENDER_PAGE_FAILED,
                "render_pdf_page is only supported for PDF documents",
            )

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

    def _discover_sidecar_catalogs(
        self, root: str | Path | None = None
    ) -> list[CatalogStore]:
        root_path = self._resolve_library_root(root)
        sidecar_dir = root_path / self.sidecar_dir_name
        db_path = sidecar_dir / "catalog.db"
        if not db_path.exists():
            return []
        return [self._get_or_create_catalog(sidecar_dir)]

    def storage_list_sidecars(
        self, *, root: str | None = None, limit: int
    ) -> dict[str, Any]:
        root_path = self._resolve_library_root(root)
        catalogs = self._discover_sidecar_catalogs(root_path)
        max_items = max(1, min(limit, 1000))
        items: list[dict[str, Any]] = []
        total_docs = 0
        for catalog in catalogs:
            docs = catalog.list_documents()
            sidecar_summary = catalog.sidecar_summary()
            total_docs += len(docs)
            for doc in docs:
                self._bind_doc_catalog(doc.doc_id, catalog)
            sample_docs = docs[:max_items]
            items.append(
                {
                    "sidecar_path": str(catalog.db_path.parent),
                    "catalog_path": str(catalog.db_path),
                    "db_size_bytes": sidecar_summary["db_size_bytes"],
                    "total_bytes": sidecar_summary["total_bytes"],
                    "documents_count": len(docs),
                    "artifact_count": sidecar_summary["artifacts_count"],
                    "node_count": sidecar_summary["nodes_count"],
                    "edge_count": sidecar_summary["edges_count"],
                    "diagnostics_count": sidecar_summary["diagnostics_count"],
                    "documents": [
                        {
                            "doc_id": doc.doc_id,
                            "path": doc.path,
                            "type": doc.type,
                            "status": doc.status,
                            "profile": doc.profile,
                            "graph": catalog.graph_stats(doc.doc_id),
                            "pipeline_status": self._document_pipeline_status(doc),
                            "latest_ingest_job": (
                                self._serialize_ingest_job(latest_job)
                                if (latest_job := catalog.get_ingest_job(doc.doc_id))
                                is not None
                                else None
                            ),
                        }
                        for doc in sample_docs
                    ],
                    "documents_truncated": len(docs) > max_items,
                }
            )
        return {
            "root": str(root_path),
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
        with self._ingest_lock:
            doc, catalog = self._resolve_doc(doc_id, path)
            deleted_records = catalog.delete_documents_by_paths([doc.path])
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
            "artifacts_removed": artifacts_removed if remove_artifacts else False,
        }

    def storage_cleanup_sidecars(
        self,
        *,
        root: str | None = None,
        remove_missing_documents: bool = True,
        remove_orphan_artifacts: bool = True,
        compact_catalog: bool = True,
    ) -> dict[str, Any]:
        root_path = self._resolve_library_root(root)
        sidecar_rows: list[dict[str, Any]] = []
        removed_paths_total: list[str] = []
        removed_docs_total = 0
        removed_artifacts_total = 0
        reclaimed_bytes_total = 0

        with self._ingest_lock:
            catalogs = self._discover_sidecar_catalogs(root_path)
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
                    removed_deleted_count = catalog.delete_documents_by_paths(
                        removed_paths
                    )
                    for doc_id_item in removed_doc_ids:
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
            "root": str(root_path),
            "sidecars_count": len(catalogs),
            "removed_deleted_count": removed_docs_total,
            "removed_paths": sorted(set(removed_paths_total)),
            "orphan_artifacts_deleted": removed_artifacts_total,
            "reclaimed_bytes": reclaimed_bytes_total,
            "sidecars": sorted(sidecar_rows, key=lambda item: item["sidecar_path"]),
        }
