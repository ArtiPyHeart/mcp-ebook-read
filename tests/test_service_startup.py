from __future__ import annotations

from pathlib import Path

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import (
    PdfParserPerformanceConfig,
    PdfParserTuningProfile,
)
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


def test_from_env_loads_docling_tuning_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = ReadyVectorIndex()
    grobid = ReadyGrobid()
    profile_path = tmp_path / "docling_tuning.json"
    profile = PdfParserTuningProfile(
        created_at="2026-03-15T00:00:00Z",
        source_path="/tmp/sample.pdf",
        sample_pages=12,
        cpu_count=8,
        total_memory_bytes=32 * 1024**3,
        selected_config=PdfParserPerformanceConfig(
            num_threads=6,
            device="auto",
            ocr_batch_size=5,
            layout_batch_size=5,
            table_batch_size=5,
        ),
        benchmarks=[],
    )
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("GROBID_URL", "http://127.0.0.1:8070")
    monkeypatch.setenv("PDF_DOCLING_TUNING_PROFILE_PATH", str(profile_path))
    monkeypatch.delenv("PDF_DOCLING_NUM_THREADS", raising=False)
    monkeypatch.delenv("PDF_DOCLING_BATCH_SIZE", raising=False)
    monkeypatch.setattr(
        "mcp_ebook_read.service.QdrantVectorIndex.from_env",
        lambda **_kwargs: vector,
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: grobid,
    )

    service = AppService.from_env()

    assert service.pdf_parser.performance_config.num_threads == 6
    assert service.pdf_parser.performance_config.ocr_batch_size == 5


def test_from_env_explicit_docling_env_overrides_tuned_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vector = ReadyVectorIndex()
    grobid = ReadyGrobid()
    profile_path = tmp_path / "docling_tuning.json"
    profile = PdfParserTuningProfile(
        created_at="2026-03-15T00:00:00Z",
        source_path="/tmp/sample.pdf",
        sample_pages=12,
        cpu_count=8,
        total_memory_bytes=32 * 1024**3,
        selected_config=PdfParserPerformanceConfig(
            num_threads=6,
            device="auto",
            ocr_batch_size=5,
            layout_batch_size=5,
            table_batch_size=5,
        ),
        benchmarks=[],
    )
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("GROBID_URL", "http://127.0.0.1:8070")
    monkeypatch.setenv("PDF_DOCLING_TUNING_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("PDF_DOCLING_NUM_THREADS", "9")
    monkeypatch.setenv("PDF_DOCLING_BATCH_SIZE", "7")
    monkeypatch.setattr(
        "mcp_ebook_read.service.QdrantVectorIndex.from_env",
        lambda **_kwargs: vector,
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: grobid,
    )

    service = AppService.from_env()

    assert service.pdf_parser.performance_config.num_threads == 9
    assert service.pdf_parser.performance_config.ocr_batch_size == 7
    assert service.pdf_parser.performance_config.layout_batch_size == 7
    assert service.pdf_parser.performance_config.table_batch_size == 7


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


def test_from_env_vector_init_failure_is_aggregated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    grobid = ReadyGrobid()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("GROBID_URL", "http://127.0.0.1:8070")
    monkeypatch.setattr(
        "mcp_ebook_read.service.QdrantVectorIndex.from_env",
        lambda **_kwargs: (_ for _ in ()).throw(
            AppError(
                ErrorCode.SEARCH_INDEX_NOT_READY,
                "FastEmbed model initialization failed.",
                details={"cache_dir": "/tmp/fastembed"},
            )
        ),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: grobid,
    )

    with pytest.raises(AppError) as exc:
        AppService.from_env()

    failed_components = exc.value.details.get("failed_components")
    assert isinstance(failed_components, list)
    assert failed_components[0]["component"] == "qdrant"
    assert failed_components[0]["message"] == "FastEmbed model initialization failed."
