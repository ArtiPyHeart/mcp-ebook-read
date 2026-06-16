from __future__ import annotations

from pathlib import Path

import pytest

from mcp_ebook_read.errors import AppError
from mcp_ebook_read.schema.models import (
    PdfParserPerformanceConfig,
    PdfParserTuningProfile,
)
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.service import AppService


class OptionalGrobid:
    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url
        self.assert_calls = 0

    def assert_available(self) -> None:
        self.assert_calls += 1


def test_from_env_starts_without_qdrant_or_grobid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    grobid = OptionalGrobid()

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("GROBID_URL", raising=False)
    monkeypatch.delenv("PDF_FORMULA_BATCH_SIZE", raising=False)
    monkeypatch.setattr(
        AppService,
        "_auto_formula_batch_size",
        classmethod(lambda cls: 7),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: grobid,
    )

    service = AppService.from_env()

    assert service.grobid_client is grobid
    assert grobid.assert_calls == 0
    assert service.pdf_parser.formula_extractor.batch_size == 7
    assert not (tmp_path / ".mcp-ebook-read").exists()
    service.close()


def test_from_env_auto_sizes_eager_ingest_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MCP_EBOOK_INGEST_WORKERS", raising=False)
    monkeypatch.delenv("PDF_DOCLING_NUM_THREADS", raising=False)
    monkeypatch.delenv("PDF_DOCLING_BATCH_SIZE", raising=False)
    monkeypatch.setenv(
        "PDF_DOCLING_TUNING_PROFILE_PATH",
        str(tmp_path / "missing-docling-tuning.json"),
    )
    monkeypatch.setattr("mcp_ebook_read.service.os.cpu_count", lambda: 12)
    monkeypatch.setattr(
        AppService,
        "_detect_total_memory_bytes",
        staticmethod(lambda: 32 * 1024**3),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: OptionalGrobid(),
    )

    service = AppService.from_env()

    assert service.ingest_worker_count == 3
    assert service.pdf_parser.performance_config.num_threads == 12
    assert service.pdf_parser.performance_config.ocr_batch_size == 12
    service.close()


def test_from_env_formula_batch_size_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDF_FORMULA_BATCH_SIZE", "3")
    monkeypatch.setenv("PDF_PARSE_TIMEOUT_SECONDS", "2400")
    monkeypatch.setattr(
        AppService,
        "_auto_formula_batch_size",
        classmethod(lambda cls: 9),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: OptionalGrobid(),
    )

    service = AppService.from_env()
    assert service.pdf_parser.formula_extractor.batch_size == 3
    assert service.pdf_parse_timeout_seconds == 2400
    service.close()


def test_from_env_loads_docling_tuning_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setenv("PDF_DOCLING_TUNING_PROFILE_PATH", str(profile_path))
    monkeypatch.delenv("PDF_DOCLING_NUM_THREADS", raising=False)
    monkeypatch.delenv("PDF_DOCLING_BATCH_SIZE", raising=False)
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: OptionalGrobid(),
    )

    service = AppService.from_env()

    assert service.pdf_parser.performance_config.num_threads == 6
    assert service.pdf_parser.performance_config.ocr_batch_size == 5
    service.close()


def test_from_env_explicit_docling_env_overrides_tuned_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setenv("PDF_DOCLING_TUNING_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("PDF_DOCLING_NUM_THREADS", "9")
    monkeypatch.setenv("PDF_DOCLING_BATCH_SIZE", "7")
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: OptionalGrobid(),
    )

    service = AppService.from_env()

    assert service.pdf_parser.performance_config.num_threads == 9
    assert service.pdf_parser.performance_config.ocr_batch_size == 7
    assert service.pdf_parser.performance_config.layout_batch_size == 7
    assert service.pdf_parser.performance_config.table_batch_size == 7
    service.close()


def test_pdf_worker_config_scales_with_active_eager_pdf_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PDF_DOCLING_NUM_THREADS", raising=False)
    monkeypatch.delenv("PDF_DOCLING_BATCH_SIZE", raising=False)
    monkeypatch.delenv("PDF_FORMULA_BATCH_SIZE", raising=False)
    service = AppService(
        sidecar_dir_name=".mcp-ebook-read",
        pdf_parser=DoclingPdfParser(
            formula_batch_size=9,
            performance_config=PdfParserPerformanceConfig(
                num_threads=12,
                device="auto",
                ocr_batch_size=9,
                layout_batch_size=9,
                table_batch_size=9,
            ),
        ),
        grobid_client=OptionalGrobid(),
        epub_parser=object(),  # type: ignore[arg-type]
        ingest_worker_count=3,
    )

    try:
        config = service._pdf_parse_worker_config(active_pdf_parses=3)
    finally:
        service.close()

    assert config["formula_batch_size"] == 3
    assert config["performance_config"]["num_threads"] == 4
    assert config["performance_config"]["ocr_batch_size"] == 3
    assert config["resource_plan"]["active_pdf_parses"] == 3
    assert config["resource_plan"]["ingest_worker_count"] == 3


def test_pdf_worker_config_respects_explicit_docling_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PDF_DOCLING_NUM_THREADS", "12")
    service = AppService(
        sidecar_dir_name=".mcp-ebook-read",
        pdf_parser=DoclingPdfParser(
            formula_batch_size=9,
            performance_config=PdfParserPerformanceConfig(
                num_threads=12,
                device="auto",
                ocr_batch_size=9,
                layout_batch_size=9,
                table_batch_size=9,
            ),
        ),
        grobid_client=OptionalGrobid(),
        epub_parser=object(),  # type: ignore[arg-type]
        ingest_worker_count=3,
    )

    try:
        config = service._pdf_parse_worker_config(active_pdf_parses=3)
    finally:
        service.close()

    assert config["performance_config"]["num_threads"] == 12
    assert config["performance_config"]["ocr_batch_size"] == 9
    assert config["resource_plan"]["docling_performance_env_overridden"] is True


def test_from_env_invalid_formula_batch_size_fails_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDF_FORMULA_BATCH_SIZE", "0")
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: OptionalGrobid(),
    )

    with pytest.raises(AppError) as exc:
        AppService.from_env()

    details = exc.value.details
    assert isinstance(details, dict)
    invalid_config = details.get("invalid_config")
    assert isinstance(invalid_config, list)
    assert [item["component"] for item in invalid_config] == ["config"]
    assert details["required_env"] == {}
    assert "GROBID_URL" in details["optional_env"]


def test_from_env_invalid_ingest_worker_count_fails_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MCP_EBOOK_INGEST_WORKERS", "0")
    monkeypatch.setattr(
        "mcp_ebook_read.service.GrobidClient.from_env",
        lambda: OptionalGrobid(),
    )

    with pytest.raises(AppError) as exc:
        AppService.from_env()

    details = exc.value.details
    assert isinstance(details, dict)
    invalid_config = details.get("invalid_config")
    assert isinstance(invalid_config, list)
    assert invalid_config[0]["details"]["env"] == "MCP_EBOOK_INGEST_WORKERS"
