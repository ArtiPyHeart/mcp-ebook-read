"""Core application service for MCP tools."""

from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict
from datetime import UTC, datetime
import json
import hashlib
import logging
import os
from queue import Empty, Queue
import shutil
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp_ebook_read.errors import AppError, ErrorCode, to_error_payload
from mcp_ebook_read.index.vector import QdrantVectorIndex
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
from mcp_ebook_read.render.pdf_images import PdfImageExtractor
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
    Profile,
    PdfParserPerformanceConfig,
    PdfParserTuningProfile,
)
from mcp_ebook_read.store.catalog import CatalogStore

logger = logging.getLogger(__name__)


class AppService:
    """Orchestrates scan, ingest, indexing and read/render operations."""

    def __init__(
        self,
        *,
        sidecar_dir_name: str,
        vector_index: QdrantVectorIndex,
        pdf_parser: DoclingPdfParser,
        pdf_image_extractor: PdfImageExtractor | None = None,
        grobid_client: GrobidClient,
        epub_parser: EbooklibEpubParser,
    ) -> None:
        self.sidecar_dir_name = sidecar_dir_name
        self.vector_index = vector_index
        self.pdf_parser = pdf_parser
        self.pdf_image_extractor = pdf_image_extractor
        self.grobid_client = grobid_client
        self.epub_parser = epub_parser
        self._catalogs: dict[str, CatalogStore] = {}
        self._doc_catalog_index: dict[str, str] = {}
        self._known_roots: set[str] = set()
        self._ingest_queue: Queue[tuple[str, str]] = Queue()
        self._ingest_lock = threading.Lock()
        self._closed = False
        self._ingest_worker = threading.Thread(
            target=self._ingest_worker_loop,
            name="mcp-ebook-read-ingest-worker",
            daemon=True,
        )
        self._ingest_worker.start()

    def close(self) -> None:
        """Release process-scoped resources owned by the service."""

        if self._closed:
            return
        self._closed = True
        self.pdf_parser.close()

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

    @staticmethod
    def _pdf_tuning_profile_path() -> Path:
        override = os.environ.get("PDF_DOCLING_TUNING_PROFILE_PATH")
        if override and override.strip():
            return Path(override).expanduser()
        if sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Caches"
                / "mcp-ebook-read"
                / "docling_pdf_tuning.json"
            )
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home and xdg_cache_home.strip():
            cache_root = Path(xdg_cache_home).expanduser()
        else:
            cache_root = Path.home() / ".cache"
        return cache_root / "mcp-ebook-read" / "docling_pdf_tuning.json"

    @classmethod
    def _load_pdf_tuning_profile(cls) -> PdfParserTuningProfile | None:
        profile_path = cls._pdf_tuning_profile_path()
        if not profile_path.exists():
            return None
        try:
            return PdfParserTuningProfile.model_validate_json(
                profile_path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"pdf_docling_tuning_profile_invalid path={profile_path} error={exc}"
            )
            return None

    @classmethod
    def _write_pdf_tuning_profile(cls, profile: PdfParserTuningProfile) -> Path:
        profile_path = cls._pdf_tuning_profile_path()
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return profile_path

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
                ErrorCode.STARTUP_DEPENDENCY_NOT_READY,
                f"{env_name} must be a positive integer.",
                details={"env": env_name, "value": raw_value},
            ) from exc
        if value < 1:
            raise AppError(
                ErrorCode.STARTUP_DEPENDENCY_NOT_READY,
                f"{env_name} must be >= 1.",
                details={"env": env_name, "value": raw_value},
            )
        return value

    @classmethod
    def _resolve_docling_performance_config(cls) -> PdfParserPerformanceConfig:
        tuned_profile = cls._load_pdf_tuning_profile()
        base = (
            tuned_profile.selected_config
            if tuned_profile is not None
            else PdfParserPerformanceConfig()
        )
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
            sidecar_dir_name=".mcp-ebook-read",
            vector_index=vector_index,
            pdf_parser=DoclingPdfParser(
                enable_docling_formula_enrichment=env_bool(
                    "DOCLING_FORMULA_ENRICHMENT", True
                ),
                require_formula_engine=env_bool("PDF_FORMULA_REQUIRE_ENGINE", True),
                formula_batch_size=formula_batch_size,
                performance_config=docling_performance_config,
            ),
            pdf_image_extractor=PdfImageExtractor(min_area_ratio=0.01),
            grobid_client=grobid_client,
            epub_parser=EbooklibEpubParser(),
        )

    def _catalog_key(self, catalog: CatalogStore) -> str:
        return str(catalog.db_path.resolve())

    def _normalize_root(self, root: str | Path) -> str:
        return str(Path(root).expanduser().resolve())

    def _register_root(self, root: str | Path) -> str:
        normalized = self._normalize_root(root)
        self._known_roots.add(normalized)
        return normalized

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
        self._register_root(resolved.parent)
        return self._get_or_create_catalog(resolved.parent / self.sidecar_dir_name)

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
            return self._require_doc(
                doc_id,
                error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
                message="Document not found by doc_id or path",
            )

        raise AppError(
            ErrorCode.INGEST_DOC_NOT_FOUND, "Document not found by doc_id or path"
        )

    def _resolve_pdf_path_for_autotune(
        self, doc_id: str | None, path: str | None
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
                    "document_autotune_pdf_parser only supports PDF documents",
                    details={"path": str(resolved)},
                )
            if doc_id is not None:
                resolved_doc, _catalog = self._resolve_doc(doc_id, str(resolved))
                if resolved_doc.type != DocumentType.PDF:
                    raise AppError(
                        ErrorCode.INGEST_UNSUPPORTED_TYPE,
                        "document_autotune_pdf_parser only supports PDF documents",
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
                "document_autotune_pdf_parser only supports PDF documents",
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

    def _compute_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _ingest_result_from_doc(
        self, doc: DocumentRecord, catalog: CatalogStore
    ) -> dict[str, Any]:
        chunks = catalog.list_chunks(doc.doc_id)
        return {
            "doc_id": doc.doc_id,
            "profile": doc.profile,
            "parser_chain": doc.parser_chain,
            "chunks_count": len(chunks),
            "formulas_count": len(catalog.list_formulas(doc.doc_id)),
            "images_count": len(catalog.list_images(doc.doc_id)),
            "outline_depth": max((node.level for node in doc.outline), default=0),
            "overall_confidence": doc.overall_confidence,
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
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
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

    def _set_job_stage(
        self,
        *,
        catalog: CatalogStore,
        job_id: str,
        status: IngestJobStatus | None = None,
        stage: IngestStage | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        catalog.update_ingest_job(
            job_id,
            status=status,
            stage=stage,
            message=message,
            result=result,
            error=error,
            started_at=started_at,
            finished_at=finished_at,
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
        active = catalog.get_active_ingest_job(doc.doc_id)
        if active is not None and active.profile == profile:
            return self._ingest_job_payload(active, deduplicated=True)

        if doc.status == DocumentStatus.READY and not force and doc.profile == profile:
            now = self._now_iso()
            job = IngestJobRecord(
                job_id=uuid4().hex,
                doc_id=doc.doc_id,
                path=doc.path,
                profile=profile,
                status=IngestJobStatus.SUCCEEDED,
                stage=IngestStage.CACHED,
                force=force,
                message="Cached READY document reused; no ingest work was scheduled.",
                result=self._ingest_result_from_doc(doc, catalog),
                created_at=now,
                updated_at=now,
                started_at=now,
                finished_at=now,
            )
            catalog.create_ingest_job(job)
            return self._ingest_job_payload(job, cached=True)

        now = self._now_iso()
        job = IngestJobRecord(
            job_id=uuid4().hex,
            doc_id=doc.doc_id,
            path=doc.path,
            profile=profile,
            status=IngestJobStatus.QUEUED,
            stage=IngestStage.QUEUED,
            force=force,
            message=f"{ingest_mode} queued for background execution.",
            created_at=now,
            updated_at=now,
        )
        catalog.create_ingest_job(job)
        self._ingest_queue.put((doc.doc_id, job.job_id))
        logger.info(
            "ingest_job_queued",
            extra={
                "doc_id": doc.doc_id,
                "job_id": job.job_id,
                "profile": profile,
                "path": doc.path,
            },
        )
        return self._ingest_job_payload(job)

    def _ingest_worker_loop(self) -> None:
        while True:
            try:
                doc_id, job_id = self._ingest_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._run_ingest_job(doc_id=doc_id, job_id=job_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "ingest_worker_unhandled_error",
                    extra={"doc_id": doc_id, "job_id": job_id},
                )
            finally:
                self._ingest_queue.task_done()

    def _run_ingest_job(self, *, doc_id: str, job_id: str) -> None:
        doc, catalog = self._require_doc(
            doc_id,
            error_code=ErrorCode.INGEST_DOC_NOT_FOUND,
            message="Document not found while executing ingest job",
        )
        job = catalog.get_ingest_job(doc_id, job_id)
        if job is None:
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                "Ingest job not found",
                details={"doc_id": doc_id, "job_id": job_id},
            )
        if job.status != IngestJobStatus.QUEUED:
            return

        started_at = self._now_iso()
        self._set_job_stage(
            catalog=catalog,
            job_id=job_id,
            status=IngestJobStatus.RUNNING,
            stage=IngestStage.PARSE,
            message="Worker started background ingest job.",
            started_at=started_at,
        )
        logger.info(
            "ingest_job_started",
            extra={"doc_id": doc_id, "job_id": job_id, "profile": job.profile},
        )

        def stage_callback(stage: IngestStage, message: str) -> None:
            self._set_job_stage(
                catalog=catalog,
                job_id=job_id,
                status=IngestJobStatus.RUNNING,
                stage=stage,
                message=message,
            )
            logger.info(
                "ingest_job_stage",
                extra={
                    "doc_id": doc_id,
                    "job_id": job_id,
                    "stage": stage,
                    "message": message,
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
                job_id=job_id,
                status=IngestJobStatus.FAILED,
                stage=IngestStage.FAILED,
                message=str(exc),
                error=error,
                finished_at=self._now_iso(),
            )
            logger.exception(
                "ingest_job_failed",
                extra={"doc_id": doc_id, "job_id": job_id},
            )
            return

        self._set_job_stage(
            catalog=catalog,
            job_id=job_id,
            status=IngestJobStatus.SUCCEEDED,
            stage=IngestStage.COMPLETED,
            message="Background ingest job completed successfully.",
            result=result,
            finished_at=self._now_iso(),
        )
        logger.info(
            "ingest_job_succeeded",
            extra={
                "doc_id": doc_id,
                "job_id": job_id,
                "chunks": result["chunks_count"],
            },
        )

    def library_scan(self, root: str, patterns: list[str]) -> dict[str, Any]:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise AppError(
                ErrorCode.SCAN_INVALID_ROOT,
                f"Invalid scan root: {root}",
            )
        self._register_root(root_path)

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
        stage_callback: Callable[[IngestStage, str], None] | None = None,
    ) -> dict[str, Any]:
        if doc.status == DocumentStatus.READY and not force and doc.profile == profile:
            return {
                "doc_id": doc.doc_id,
                "profile": doc.profile,
                "parser_chain": doc.parser_chain,
                "chunks_count": len(
                    catalog.get_chunks_window(doc.doc_id, 0, 0, 1_000_000)
                ),
                "formulas_count": len(catalog.list_formulas(doc.doc_id)),
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
                    if stage_callback is not None:
                        stage_callback(
                            IngestStage.PARSE,
                            "Parsing PDF book with Docling.",
                        )
                    parsed = self.pdf_parser.parse(doc.path, doc.doc_id)
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
                if stage_callback is not None:
                    stage_callback(
                        IngestStage.GROBID,
                        "Parsing PDF paper metadata with GROBID.",
                    )
                grobid_result = self.grobid_client.parse_fulltext(doc.path)
                if stage_callback is not None:
                    stage_callback(
                        IngestStage.PARSE,
                        "Parsing PDF paper structure with Docling.",
                    )
                parsed = self.pdf_parser.parse(doc.path, doc.doc_id)
                parsed.parser_chain.append("grobid")
                parsed.metadata = {**parsed.metadata, **grobid_result.metadata}
                paper_title = str(
                    grobid_result.metadata.get("paper_title") or ""
                ).strip()
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
            if stage_callback is not None:
                stage_callback(
                    IngestStage.INDEX,
                    "Rebuilding vector index for document chunks.",
                )
            self.vector_index.rebuild_document(doc.doc_id, parsed.title, parsed.chunks)
            if stage_callback is not None:
                stage_callback(
                    IngestStage.FINALIZE,
                    "Saving final document metadata and marking document ready.",
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
            return {
                "doc_id": doc.doc_id,
                "profile": profile,
                "parser_chain": parsed.parser_chain,
                "chunks_count": len(parsed.chunks),
                "formulas_count": len(parsed.formulas),
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

    def document_autotune_pdf_parser(
        self,
        *,
        doc_id: str | None,
        path: str | None,
        sample_pages: int = 20,
    ) -> dict[str, Any]:
        resolved_path, resolved_doc_id = self._resolve_pdf_path_for_autotune(
            doc_id, path
        )
        profile = self.pdf_parser.autotune(
            pdf_path=str(resolved_path),
            sample_pages=sample_pages,
            total_memory_bytes=self._detect_total_memory_bytes(),
        )
        profile_path = self._write_pdf_tuning_profile(profile)
        self.pdf_parser.set_performance_config(profile.selected_config)
        logger.info(
            "pdf_docling_autotune_applied "
            f"path={resolved_path} profile_path={profile_path} "
            f"threads={profile.selected_config.num_threads} "
            f"batch={profile.selected_config.ocr_batch_size}"
        )
        return {
            "doc_id": resolved_doc_id,
            "path": str(resolved_path),
            "profile_path": str(profile_path),
            "selected_config": profile.selected_config.model_dump(mode="json"),
            "benchmarks": [row.model_dump(mode="json") for row in profile.benchmarks],
            "sample_pages": profile.sample_pages,
            "cpu_count": profile.cpu_count,
            "total_memory_bytes": profile.total_memory_bytes,
        }

    def document_ingest_pdf_book(
        self, *, doc_id: str | None, path: str | None, force: bool
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path)
        return self._submit_ingest_job(
            doc=doc,
            catalog=catalog,
            profile=Profile.BOOK,
            expected_doc_type=DocumentType.PDF,
            force=force,
            ingest_mode="document_ingest_pdf_book",
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

    def document_ingest_epub_book(
        self, *, doc_id: str | None, path: str | None, force: bool
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path)
        return self._submit_ingest_job(
            doc=doc,
            catalog=catalog,
            profile=Profile.BOOK,
            expected_doc_type=DocumentType.EPUB,
            force=force,
            ingest_mode="document_ingest_epub_book",
        )

    def document_ingest_pdf_paper(
        self, *, doc_id: str | None, path: str | None, force: bool
    ) -> dict[str, Any]:
        doc, catalog = self._resolve_doc(doc_id, path)
        return self._submit_ingest_job(
            doc=doc,
            catalog=catalog,
            profile=Profile.PAPER,
            expected_doc_type=DocumentType.PDF,
            force=force,
            ingest_mode="document_ingest_pdf_paper",
        )

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
        _, node, node_path, _ = self._resolve_outline_node(doc_id, node_id)
        expanded_top_k = max(top_k * 5, top_k)
        raw_hits = self.vector_index.search(
            query=query, top_k=expanded_top_k, doc_ids=[doc_id]
        )
        filtered_hits: list[dict[str, Any]] = []
        for hit in raw_hits:
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

    def _ensure_pdf_profile_doc(
        self, doc_id: str, *, expected_profile: Profile
    ) -> tuple[DocumentRecord, CatalogStore]:
        doc, catalog = self._ensure_pdf_doc(doc_id)
        if doc.profile != expected_profile:
            raise AppError(
                ErrorCode.INGEST_UNSUPPORTED_TYPE,
                "The requested formula tool does not match the ingested PDF profile.",
                details={
                    "doc_id": doc_id,
                    "doc_profile": doc.profile,
                    "required_profile": expected_profile,
                    "hint": "Use the matching document_ingest_pdf_book or document_ingest_pdf_paper before formula tools.",
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
        expected_profile: Profile,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_profile_doc(
            doc_id, expected_profile=expected_profile
        )
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
            "profile": expected_profile,
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
        expected_profile: Profile,
    ) -> dict[str, Any]:
        doc, catalog = self._ensure_pdf_profile_doc(
            doc_id, expected_profile=expected_profile
        )
        formula = catalog.get_formula(formula_id)
        if formula is None or formula.doc_id != doc_id:
            raise AppError(
                ErrorCode.READ_FORMULA_NOT_FOUND,
                f"Unknown formula_id: {formula_id}",
                details={
                    "doc_id": doc_id,
                    "formula_id": formula_id,
                    "hint": "Call the corresponding pdf_*_list_formulas tool first.",
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

        return {
            "doc_title": doc.title,
            "profile": expected_profile,
            "formula": self._formula_payload(formula),
            "context": {
                "text": context_chunk.text,
                "locator": context_chunk.locator.model_dump(),
            }
            if context_chunk
            else None,
            "evidence": evidence,
        }

    def pdf_book_list_formulas(
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
            expected_profile=Profile.BOOK,
        )

    def pdf_book_read_formula(self, *, doc_id: str, formula_id: str) -> dict[str, Any]:
        return self._read_pdf_formula(
            doc_id=doc_id,
            formula_id=formula_id,
            expected_profile=Profile.BOOK,
        )

    def pdf_paper_list_formulas(
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
            expected_profile=Profile.PAPER,
        )

    def pdf_paper_read_formula(self, *, doc_id: str, formula_id: str) -> dict[str, Any]:
        return self._read_pdf_formula(
            doc_id=doc_id,
            formula_id=formula_id,
            expected_profile=Profile.PAPER,
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

    def _discover_sidecar_catalogs(self, root: str) -> list[CatalogStore]:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise AppError(
                ErrorCode.SCAN_INVALID_ROOT,
                f"Invalid scan root: {root}",
            )
        self._register_root(root_path)

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
