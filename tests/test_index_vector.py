from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.index.vector import QdrantVectorIndex
from mcp_ebook_read.schema.models import ChunkRecord, Locator


class FakeEmbeddingVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class FakeEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    def embed(self, texts: list[str]):
        return [
            FakeEmbeddingVector([float(i + 1), float(i + 2)])
            for i, _ in enumerate(texts)
        ]


class FakeQdrantClient:
    def __init__(self, *, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.exists = False
        self.raise_get = False
        self.raise_query = False
        self.created_collections: list[str] = []
        self.deleted_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self.query_points_result: list[SimpleNamespace] = []

    def get_collections(self) -> dict[str, object]:
        if self.raise_get:
            raise RuntimeError("qdrant down")
        return {"status": "ok"}

    def collection_exists(self, _collection_name: str) -> bool:
        return self.exists

    def create_collection(self, collection_name: str, vectors_config: object) -> None:  # noqa: ARG002
        self.created_collections.append(collection_name)
        self.exists = True

    def delete(self, **kwargs: object) -> None:
        self.deleted_calls.append(kwargs)

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs: object):
        self.query_calls.append(kwargs)
        if self.raise_query:
            raise RuntimeError("query failed")
        return SimpleNamespace(points=self.query_points_result)


def _make_chunk(doc_id: str, chunk_id: str, text: str) -> ChunkRecord:
    locator = Locator(
        doc_id=doc_id,
        chunk_id=chunk_id,
        section_path=["Sec"],
        method="docling",
    )
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        order_index=0,
        section_path=["Sec"],
        text=text,
        search_text=text,
        locator=locator,
        method="docling",
    )


def _build_index(monkeypatch: pytest.MonkeyPatch) -> QdrantVectorIndex:
    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", FakeQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", FakeEmbedder)
    return QdrantVectorIndex(url="http://localhost:6333", collection="test_collection")


def test_vector_index_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "my_collection")
    monkeypatch.setenv("FASTEMBED_MODEL", "model-a")
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", FakeQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", FakeEmbedder)

    index = QdrantVectorIndex.from_env()

    assert index.url == "http://qdrant:6333"
    assert index.collection == "my_collection"
    assert index.timeout == 12.0


def test_vector_index_backend_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class DownQdrantClient(FakeQdrantClient):
        def get_collections(self) -> dict[str, object]:
            raise RuntimeError("down")

    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", DownQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", FakeEmbedder)

    with pytest.raises(AppError) as exc:
        QdrantVectorIndex(url="http://localhost:6333", collection="x")

    assert exc.value.code == ErrorCode.SEARCH_INDEX_NOT_READY


def test_rebuild_document_creates_collection_and_upserts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _build_index(monkeypatch)
    chunks = [
        _make_chunk("doc1", "c1", "alpha"),
        _make_chunk("doc1", "c2", "beta"),
    ]

    index.rebuild_document("doc1", "Doc", chunks)

    assert index.client.created_collections == ["test_collection"]
    assert len(index.client.deleted_calls) == 1
    assert len(index.client.upsert_calls) == 1
    upsert_points = index.client.upsert_calls[0]["points"]
    assert len(upsert_points) == 2


def test_search_empty_query_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _build_index(monkeypatch)

    assert index.search("   ") == []


def test_search_success_with_doc_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _build_index(monkeypatch)
    index.client.query_points_result = [
        SimpleNamespace(
            score=0.8,
            payload={
                "chunk_id": "c1",
                "doc_id": "doc1",
                "title": "Doc",
                "section": "Sec",
                "text": "alpha beta gamma",
                "locator": json.dumps(
                    {
                        "doc_id": "doc1",
                        "chunk_id": "c1",
                        "section_path": ["Sec"],
                        "method": "docling",
                    }
                ),
            },
        )
    ]

    hits = index.search("alpha", top_k=3, doc_ids=["doc1"])

    assert len(hits) == 1
    assert hits[0]["doc_id"] == "doc1"
    assert hits[0]["locator"]["chunk_id"] == "c1"
    assert len(index.client.query_calls) == 1
    assert index.client.query_calls[0]["limit"] == 3


def test_search_query_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _build_index(monkeypatch)
    index.client.raise_query = True

    with pytest.raises(AppError) as exc:
        index.search("alpha")

    assert exc.value.code == ErrorCode.SEARCH_INDEX_NOT_READY
