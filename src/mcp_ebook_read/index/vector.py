"""Qdrant + FastEmbed vector index."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.network import should_trust_env_proxy
from mcp_ebook_read.schema.models import ChunkRecord

logger = logging.getLogger(__name__)

_DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_FASTEMBED_INIT_RETRIES = 3
_FASTEMBED_RETRY_BASE_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class _FastembedModelCacheLayout:
    model_name: str
    hf_repo: str
    model_file: str
    model_cache_dir: Path
    lock_dir: Path


def _default_fastembed_cache_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mcp-ebook-read" / "fastembed"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home and xdg_cache_home.strip():
        cache_root = Path(xdg_cache_home).expanduser()
    else:
        cache_root = Path.home() / ".cache"
    return cache_root / "mcp-ebook-read" / "fastembed"


def _resolve_fastembed_cache_path() -> Path:
    override = os.environ.get("FASTEMBED_CACHE_PATH")
    cache_path = (
        Path(override).expanduser()
        if override and override.strip()
        else _default_fastembed_cache_path()
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _extract_broken_model_cache_dir(exc: Exception, cache_dir: Path) -> Path | None:
    pattern = re.compile(rf"({re.escape(str(cache_dir))}[^\s\"']+)")
    match = pattern.search(str(exc))
    if match is None:
        return None

    failed_path = Path(match.group(1))
    for parent in failed_path.parents:
        if parent.parent == cache_dir and parent.name.startswith("models--"):
            return parent
    return None


def _resolve_fastembed_model_cache_layout(
    model_name: str | None, cache_dir: Path
) -> _FastembedModelCacheLayout | None:
    resolved_model_name = model_name or _DEFAULT_FASTEMBED_MODEL
    list_supported_models = getattr(TextEmbedding, "list_supported_models", None)
    if not callable(list_supported_models):
        return None
    for model_info in list_supported_models():
        current_model_name = model_info.get("model")
        if not isinstance(current_model_name, str):
            continue
        if current_model_name.lower() != resolved_model_name.lower():
            continue
        sources = model_info.get("sources")
        model_file = model_info.get("model_file")
        hf_repo = sources.get("hf") if isinstance(sources, dict) else None
        if not isinstance(hf_repo, str) or not hf_repo.strip():
            return None
        if not isinstance(model_file, str) or not model_file.strip():
            return None
        model_cache_dir = cache_dir / f"models--{hf_repo.replace('/', '--')}"
        return _FastembedModelCacheLayout(
            model_name=resolved_model_name,
            hf_repo=hf_repo,
            model_file=model_file,
            model_cache_dir=model_cache_dir,
            lock_dir=cache_dir / ".locks" / model_cache_dir.name,
        )
    return None


def _model_snapshot_has_required_artifact(layout: _FastembedModelCacheLayout) -> bool:
    refs_main = layout.model_cache_dir / "refs" / "main"
    checked_snapshot_dirs: set[Path] = set()

    def iter_snapshot_dirs() -> list[Path]:
        candidates: list[Path] = []
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(layout.model_cache_dir / "snapshots" / revision)
        snapshots_dir = layout.model_cache_dir / "snapshots"
        if snapshots_dir.exists():
            candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())
        unique_candidates: list[Path] = []
        for candidate in candidates:
            if candidate in checked_snapshot_dirs:
                continue
            checked_snapshot_dirs.add(candidate)
            unique_candidates.append(candidate)
        return unique_candidates

    return any(
        (snapshot_dir / layout.model_file).exists()
        for snapshot_dir in iter_snapshot_dirs()
    )


def _clear_fastembed_model_cache_artifacts(
    cache_dir: Path, model_cache_dir: Path
) -> tuple[str | None, str | None]:
    cleared_model_cache_dir = None
    if model_cache_dir.exists():
        shutil.rmtree(model_cache_dir, ignore_errors=True)
        cleared_model_cache_dir = str(model_cache_dir)

    lock_dir = cache_dir / ".locks" / model_cache_dir.name
    cleared_lock_dir = None
    if lock_dir.exists():
        shutil.rmtree(lock_dir, ignore_errors=True)
        cleared_lock_dir = str(lock_dir)
    return cleared_model_cache_dir, cleared_lock_dir


def _prepare_fastembed_cache(
    model_name: str | None, cache_dir: Path
) -> dict[str, Any] | None:
    layout = _resolve_fastembed_model_cache_layout(model_name, cache_dir)
    if layout is None or not layout.model_cache_dir.exists():
        return None
    if _model_snapshot_has_required_artifact(layout):
        return None

    lock_files = (
        sorted(layout.lock_dir.glob("*.lock")) if layout.lock_dir.exists() else []
    )
    active_lock_files: list[str] = []
    stale_lock_files: list[str] = []
    if lock_files:
        from filelock import FileLock, Timeout

        for lock_file in lock_files:
            file_lock = FileLock(str(lock_file))
            try:
                file_lock.acquire(timeout=0)
            except Timeout:
                active_lock_files.append(str(lock_file))
                continue
            try:
                lock_file.unlink(missing_ok=True)
            finally:
                file_lock.release()
            stale_lock_files.append(str(lock_file))

    if active_lock_files:
        raise AppError(
            ErrorCode.SEARCH_INDEX_NOT_READY,
            "FastEmbed cache is incomplete and currently locked by another process.",
            details={
                "cache_dir": str(cache_dir),
                "model_name": layout.model_name,
                "model_cache_dir": str(layout.model_cache_dir),
                "lock_files": active_lock_files,
                "hint": (
                    "Stop stale mcp-ebook-read processes holding FastEmbed cache locks "
                    "or set FASTEMBED_CACHE_PATH to a fresh directory."
                ),
            },
        )

    cleared_model_cache_dir, cleared_lock_dir = _clear_fastembed_model_cache_artifacts(
        cache_dir, layout.model_cache_dir
    )
    return {
        "cache_dir": str(cache_dir),
        "model_name": layout.model_name,
        "model_cache_dir": str(layout.model_cache_dir),
        "cleared_model_cache_dir": cleared_model_cache_dir,
        "cleared_lock_dir": cleared_lock_dir,
        "stale_lock_files": stale_lock_files,
    }


def _should_retry_fastembed_init(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_tokens = (
        "no_suchfile",
        "no such file",
        "doesn't exist",
        "ssl",
        "maxretryerror",
        "unexpected eof",
        "connection reset",
        "temporarily unavailable",
        "timed out",
    )
    return any(token in message for token in retry_tokens)


def _build_text_embedder(model_name: str | None, cache_dir: Path) -> TextEmbedding:
    prepared_cache = _prepare_fastembed_cache(model_name, cache_dir)
    if prepared_cache is not None:
        logger.warning("fastembed_cache_repaired", extra=prepared_cache)

    last_exc: Exception | None = None
    for attempt in range(1, _FASTEMBED_INIT_RETRIES + 1):
        try:
            if model_name:
                return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
            return TextEmbedding(cache_dir=str(cache_dir))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            should_retry = (
                attempt < _FASTEMBED_INIT_RETRIES and _should_retry_fastembed_init(exc)
            )
            cleared_cache_dir = None
            cleared_lock_dir = None
            if should_retry:
                broken_model_cache_dir = _extract_broken_model_cache_dir(exc, cache_dir)
                if broken_model_cache_dir and broken_model_cache_dir.exists():
                    cleared_cache_dir, cleared_lock_dir = (
                        _clear_fastembed_model_cache_artifacts(
                            cache_dir, broken_model_cache_dir
                        )
                    )
            logger.warning(
                "fastembed_init_failed",
                extra={
                    "attempt": attempt,
                    "retries": _FASTEMBED_INIT_RETRIES,
                    "cache_dir": str(cache_dir),
                    "model_name": model_name or _DEFAULT_FASTEMBED_MODEL,
                    "retrying": should_retry,
                    "cleared_cache_dir": cleared_cache_dir,
                    "cleared_lock_dir": cleared_lock_dir,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            if not should_retry:
                break
            time.sleep(_FASTEMBED_RETRY_BASE_DELAY_SECONDS * attempt)

    assert last_exc is not None
    raise AppError(
        ErrorCode.SEARCH_INDEX_NOT_READY,
        "FastEmbed model initialization failed.",
        details={
            "cache_dir": str(cache_dir),
            "model_name": model_name or _DEFAULT_FASTEMBED_MODEL,
            "attempts": _FASTEMBED_INIT_RETRIES,
        },
    ) from last_exc


class QdrantVectorIndex:
    """Single-path vector search index powered by Qdrant and FastEmbed."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        model_name: str | None = None,
        timeout: float = 10.0,
        check_backend_ready: bool = True,
    ) -> None:
        self.url = url
        self.collection = collection
        self.timeout = timeout
        trust_env_proxy = should_trust_env_proxy(url)
        self.client = QdrantClient(
            url=url,
            timeout=timeout,
            trust_env=trust_env_proxy,
            check_compatibility=trust_env_proxy,
        )
        self.fastembed_cache_dir = _resolve_fastembed_cache_path()
        self.embedder = _build_text_embedder(model_name, self.fastembed_cache_dir)
        self._vector_size: int | None = None
        if check_backend_ready:
            self._assert_backend_ready()

    @classmethod
    def from_env(cls, *, check_backend_ready: bool = True) -> "QdrantVectorIndex":
        return cls(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            collection=os.environ.get("QDRANT_COLLECTION", "mcp_ebook_read_chunks"),
            model_name=os.environ.get("FASTEMBED_MODEL"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT_SECONDS", "10")),
            check_backend_ready=check_backend_ready,
        )

    def assert_ready(self) -> None:
        """Explicit startup preflight check for Qdrant availability."""
        self._assert_backend_ready()

    def _assert_backend_ready(self) -> None:
        try:
            self.client.get_collections()
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.SEARCH_INDEX_NOT_READY,
                f"Qdrant backend unavailable at {self.url}",
                details={"url": self.url},
            ) from exc

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = [vector.tolist() for vector in self.embedder.embed(texts)]
        if not vectors:
            return []
        if self._vector_size is None:
            self._vector_size = len(vectors[0])
        return vectors

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

    def _point_id(self, chunk_id: str) -> str:
        """Convert arbitrary chunk id into a stable UUID acceptable by Qdrant."""
        return str(uuid5(NAMESPACE_URL, chunk_id))

    def rebuild_document(
        self, doc_id: str, title: str, chunks: list[ChunkRecord]
    ) -> None:
        if not chunks:
            return

        vectors = self._embed([chunk.search_text for chunk in chunks])
        if not vectors:
            return

        self._ensure_collection(len(vectors[0]))

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": title,
                "section": " / ".join(chunk.section_path),
                "section_path": chunk.section_path,
                "page_range": chunk.locator.page_range,
                "text": chunk.text,
                "locator": chunk.locator.model_dump(),
            }
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_document(self, doc_id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "vector_delete_document_failed",
                extra={"doc_id": doc_id, "collection": self.collection},
            )

    def search(
        self, query: str, top_k: int = 20, doc_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        vector = self._embed([query])
        if not vector:
            return []

        query_filter = None
        if doc_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchAny(any=doc_ids),
                    )
                ]
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector[0],
                query_filter=query_filter,
                with_payload=True,
                limit=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.SEARCH_INDEX_NOT_READY,
                "Vector search failed.",
                details={"collection": self.collection},
            ) from exc

        hits = response.points
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            locator = payload.get("locator")
            if isinstance(locator, str):
                locator = json.loads(locator)

            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "title": payload.get("title"),
                    "section": payload.get("section"),
                    "score": float(hit.score),
                    "snippet": (payload.get("text") or "")[:320],
                    "locator": locator,
                }
            )
        logger.info("vector_search_completed", extra={"hits": len(results)})
        return results
