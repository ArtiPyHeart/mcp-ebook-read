from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from mcp_ebook_read.benchmark.concurrency import (
    _collect_documents,
    _parse_backends,
    run_concurrency_benchmark,
)


def _write_sample(path: Path) -> Path:
    path.write_bytes(b"sample")
    return path


def _fake_runner(path: str, workload: str, index: int) -> dict[str, Any]:
    return {
        "path": path,
        "workload": workload,
        "status": "ok",
        "elapsed_seconds": 0.001,
        "parser_chain": ["fake"],
        "chunks_count": index + 1,
        "outline_nodes_count": 1,
        "formulas_count": 0,
        "images_count": 0,
        "tables_count": 0,
        "figures_count": 0,
        "text_chars": 10 * (index + 1),
    }


def test_collect_documents_filters_epub_and_pdf(tmp_path: Path) -> None:
    epub = _write_sample(tmp_path / "a.epub")
    pdf = _write_sample(tmp_path / "b.pdf")
    _write_sample(tmp_path / "c.txt")

    assert _collect_documents(tmp_path) == [epub, pdf]
    assert _collect_documents(tmp_path, max_documents=1) == [epub]


def test_parse_backends_trims_empty_items() -> None:
    assert _parse_backends(" sequential, thread,,bocpy ") == [
        "sequential",
        "thread",
        "bocpy",
    ]


def test_run_concurrency_benchmark_compares_sequential_and_thread(
    tmp_path: Path,
) -> None:
    paths = [
        _write_sample(tmp_path / "a.epub"),
        _write_sample(tmp_path / "b.pdf"),
    ]

    result = run_concurrency_benchmark(
        paths,
        workload="auto",
        backends=["sequential", "thread"],
        max_workers=2,
        task_runner=_fake_runner,
    )

    assert result["summary"]["documents_total"] == 2
    assert result["summary"]["max_workers"] == 2
    assert [item["backend"] for item in result["results"]] == [
        "sequential",
        "thread",
    ]
    assert result["results"][0]["documents_ok"] == 2
    assert result["results"][1]["documents_ok"] == 2
    assert result["results"][0]["totals"]["text_chars"] == 30
    assert result["recommendation"]["backend"] in {"sequential", "thread"}


def test_run_concurrency_benchmark_skips_process_with_injected_runner(
    tmp_path: Path,
) -> None:
    path = _write_sample(tmp_path / "a.epub")

    result = run_concurrency_benchmark(
        [path],
        backends=["process"],
        task_runner=_fake_runner,
    )

    assert result["results"][0]["backend"] == "process"
    assert result["results"][0]["status"] == "skipped"
    assert "default pickle-safe task runner" in result["results"][0]["reason"]


def test_run_concurrency_benchmark_skips_bocpy_with_injected_runner(
    tmp_path: Path, monkeypatch
) -> None:
    path = _write_sample(tmp_path / "a.epub")
    monkeypatch.setitem(sys.modules, "bocpy", None)

    result = run_concurrency_benchmark(
        [path],
        backends=["bocpy"],
        task_runner=_fake_runner,
    )

    assert result["results"][0]["backend"] == "bocpy"
    assert result["results"][0]["status"] == "skipped"
    assert "default importable parser task runner" in result["results"][0]["reason"]


def test_run_concurrency_benchmark_reports_missing_bocpy(
    tmp_path: Path, monkeypatch
) -> None:
    path = _write_sample(tmp_path / "a.epub")
    monkeypatch.setitem(sys.modules, "bocpy", None)

    result = run_concurrency_benchmark(
        [path],
        backends=["bocpy"],
    )

    assert result["results"][0]["backend"] == "bocpy"
    assert result["results"][0]["status"] == "skipped"
    assert "uvx --with bocpy" in result["results"][0]["hint"]


def test_run_concurrency_benchmark_rejects_unknown_inputs(tmp_path: Path) -> None:
    path = _write_sample(tmp_path / "a.epub")

    try:
        run_concurrency_benchmark([path], workload="unknown")
    except ValueError as exc:
        assert "Unsupported workload" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unsupported workload error")

    try:
        run_concurrency_benchmark([path], backends=["unknown"])
    except ValueError as exc:
        assert "Unsupported concurrency backends" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unsupported backend error")
