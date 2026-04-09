from __future__ import annotations

import json
from pathlib import Path
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
    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

    def embed(self, texts: list[str]):
        return [
            FakeEmbeddingVector([float(i + 1), float(i + 2)])
            for i, _ in enumerate(texts)
        ]


class FakeQdrantClient:
    def __init__(self, *, url: str, timeout: float, **kwargs: object) -> None:
        self.url = url
        self.timeout = timeout
        self.trust_env = kwargs.get("trust_env")
        self.check_compatibility = kwargs.get("check_compatibility")
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
    assert index.client.trust_env is True
    assert index.client.check_compatibility is True


def test_vector_index_ignores_env_proxy_for_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", FakeQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", FakeEmbedder)

    index = QdrantVectorIndex(url="http://127.0.0.1:6333", collection="my_collection")

    assert index.client.trust_env is False
    assert index.client.check_compatibility is False


def test_vector_index_backend_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class DownQdrantClient(FakeQdrantClient):
        def get_collections(self) -> dict[str, object]:
            raise RuntimeError("down")

    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", DownQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", FakeEmbedder)

    with pytest.raises(AppError) as exc:
        QdrantVectorIndex(url="http://localhost:6333", collection="x")

    assert exc.value.code == ErrorCode.SEARCH_INDEX_NOT_READY


def test_vector_index_uses_stable_fastembed_cache_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", FakeQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", FakeEmbedder)
    monkeypatch.setattr(
        "mcp_ebook_read.index.vector._resolve_fastembed_cache_path",
        lambda: tmp_path / "fastembed",
    )

    index = QdrantVectorIndex(
        url="http://localhost:6333", collection="cache_collection"
    )

    assert index.fastembed_cache_dir == tmp_path / "fastembed"
    assert index.embedder.cache_dir == str(tmp_path / "fastembed")


def test_vector_index_retries_after_broken_fastembed_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts: list[int] = []
    cache_root = tmp_path / "fastembed"
    broken_model_dir = cache_root / "models--broken-model"

    class FlakyEmbedder(FakeEmbedder):
        def __init__(
            self,
            model_name: str | None = None,
            cache_dir: str | None = None,
        ) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                broken_snapshot = (
                    Path(cache_dir or "")
                    / "models--broken-model"
                    / "snapshots"
                    / "abc123"
                )
                broken_snapshot.mkdir(parents=True, exist_ok=True)
                raise RuntimeError(
                    f"Load model from {broken_snapshot / 'model_optimized.onnx'} failed: "
                    "File doesn't exist"
                )
            super().__init__(model_name=model_name, cache_dir=cache_dir)

    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", FakeQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", FlakyEmbedder)
    monkeypatch.setattr(
        "mcp_ebook_read.index.vector._resolve_fastembed_cache_path",
        lambda: cache_root,
    )
    monkeypatch.setattr("mcp_ebook_read.index.vector.time.sleep", lambda *_args: None)

    index = QdrantVectorIndex(
        url="http://localhost:6333", collection="retry_collection"
    )

    assert len(attempts) == 2
    assert not broken_model_dir.exists()
    assert index.embedder.cache_dir == str(cache_root)


def test_vector_index_reports_fastembed_init_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def __init__(
            self,
            model_name: str | None = None,
            cache_dir: str | None = None,
        ) -> None:
            raise RuntimeError("TLS handshake failed with unexpected EOF")

    monkeypatch.setattr("mcp_ebook_read.index.vector.QdrantClient", FakeQdrantClient)
    monkeypatch.setattr("mcp_ebook_read.index.vector.TextEmbedding", BrokenEmbedder)
    monkeypatch.setattr(
        "mcp_ebook_read.index.vector._resolve_fastembed_cache_path",
        lambda: tmp_path / "fastembed",
    )
    monkeypatch.setattr("mcp_ebook_read.index.vector.time.sleep", lambda *_args: None)

    with pytest.raises(AppError) as exc:
        QdrantVectorIndex(url="http://localhost:6333", collection="broken_collection")

    assert exc.value.code == ErrorCode.SEARCH_INDEX_NOT_READY
    assert exc.value.details["cache_dir"] == str(tmp_path / "fastembed")


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
