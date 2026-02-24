"""Qdrant + FastEmbed vector index."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import ChunkRecord

logger = logging.getLogger(__name__)


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
        self.client = QdrantClient(url=url, timeout=timeout)
        self.embedder = (
            TextEmbedding(model_name=model_name) if model_name else TextEmbedding()
        )
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
