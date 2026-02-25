from __future__ import annotations

from pathlib import Path

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.service import AppService


class ReadyVectorIndex:
    def assert_ready(self) -> None:
        return None


class FailingVectorIndex:
    def __init__(self, message: str = "qdrant down") -> None:
        self.message = message

    def assert_ready(self) -> None:
        raise AppError(
            ErrorCode.SEARCH_INDEX_NOT_READY,
            self.message,
            details={"url": "http://localhost:6333"},
        )


class ReadyGrobid:
    def assert_available(self) -> None:
        return None


class FailingGrobid:
    def __init__(self, message: str = "grobid missing") -> None:
        self.message = message

    def assert_available(self) -> None:
        raise AppError(
            ErrorCode.INGEST_PAPER_GROBID_UNAVAILABLE,
            self.message,
            details={"base_url": ""},
        )


def test_from_env_startup_preflight_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = ReadyVectorIndex()
    grobid = ReadyGrobid()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("GROBID_URL", "http://127.0.0.1:8070")
    monkeypatch.delenv("PDF_FORMULA_BATCH_SIZE", raising=False)
    monkeypatch.setattr(
        AppService,
        "_auto_formula_batch_size",
        classmethod(lambda cls: 7),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.QdrantVectorIndex.from_env",
        lambda **_kwargs: vector,
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: grobid,
    )

    service = AppService.from_env()

    assert service.vector_index is vector
    assert service.grobid_client is grobid
    assert service.pdf_parser.formula_extractor.batch_size == 7
    assert not (tmp_path / ".mcp-ebook-read").exists()


def test_from_env_formula_batch_size_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = ReadyVectorIndex()
    grobid = ReadyGrobid()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("GROBID_URL", "http://127.0.0.1:8070")
    monkeypatch.setenv("PDF_FORMULA_BATCH_SIZE", "3")
    monkeypatch.setattr(
        AppService,
        "_auto_formula_batch_size",
        classmethod(lambda cls: 9),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.QdrantVectorIndex.from_env",
        lambda **_kwargs: vector,
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: grobid,
    )

    service = AppService.from_env()
    assert service.pdf_parser.formula_extractor.batch_size == 3


def test_from_env_startup_preflight_aggregates_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("GROBID_URL", raising=False)
    monkeypatch.setattr(
        "mcp_ebook_read.service.QdrantVectorIndex.from_env",
        lambda **_kwargs: FailingVectorIndex("qdrant unavailable"),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: FailingGrobid("GROBID_URL is not configured."),
    )

    with pytest.raises(AppError) as exc:
        AppService.from_env()

    err = exc.value
    assert err.code == ErrorCode.STARTUP_DEPENDENCY_NOT_READY

    details = err.details
    assert isinstance(details, dict)
    failed_components = details.get("failed_components")
    assert isinstance(failed_components, list)
    assert [item["component"] for item in failed_components] == [
        "qdrant",
        "grobid",
    ]

    required_env = details.get("required_env")
    assert isinstance(required_env, dict)
    assert required_env["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert required_env["GROBID_URL"] == "http://127.0.0.1:8070"
    assert "quick_start" in details


def test_from_env_invalid_formula_batch_size_fails_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = ReadyVectorIndex()
    grobid = ReadyGrobid()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("GROBID_URL", "http://127.0.0.1:8070")
    monkeypatch.setenv("PDF_FORMULA_BATCH_SIZE", "0")
    monkeypatch.setattr(
        "mcp_ebook_read.service.QdrantVectorIndex.from_env",
        lambda **_kwargs: vector,
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: grobid,
    )

    with pytest.raises(AppError) as exc:
        AppService.from_env()

    details = exc.value.details
    failed_components = details.get("failed_components")
    assert isinstance(failed_components, list)
    assert any(item["component"] == "config" for item in failed_components)
