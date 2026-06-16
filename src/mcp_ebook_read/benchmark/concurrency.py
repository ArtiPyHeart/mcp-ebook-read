"""Concurrency backend benchmark for EPUB/PDF parser workloads."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable

from mcp_ebook_read.benchmark.paths import DOCUMENT_SUFFIXES, collect_documents

_BACKENDS = {"sequential", "thread", "process", "bocpy"}
_WORKLOADS = {"auto", "epub_full", "pdf_fast", "pdf_fidelity"}
_SUFFIXES = {".epub", ".pdf"}
TaskRunner = Callable[[str, str, int], dict[str, Any]]


@dataclass(frozen=True)
class _Task:
    index: int
    path: str
    workload: str


def _doc_id(path: str, index: int, workload: str) -> str:
    digest = hashlib.sha1(
        f"{path}:{index}:{workload}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return f"bench-{digest[:16]}"


def _error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _error_payload_with_traceback(exc: Exception) -> dict[str, Any]:
    payload = _error_payload(exc)
    payload["traceback"] = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return payload


def _parsed_metrics(parsed: Any, *, elapsed_seconds: float) -> dict[str, Any]:
    text_chars = sum(len(chunk.text or "") for chunk in parsed.chunks)
    return {
        "status": "ok",
        "elapsed_seconds": round(elapsed_seconds, 6),
        "title": parsed.title,
        "parser_chain": parsed.parser_chain,
        "chunks_count": len(parsed.chunks),
        "outline_nodes_count": len(parsed.outline),
        "formulas_count": len(parsed.formulas),
        "images_count": len(parsed.images),
        "tables_count": int(parsed.metadata.get("pdf_tables_count") or 0),
        "figures_count": int(parsed.metadata.get("pdf_figures_count") or 0),
        "text_chars": text_chars,
        "metadata": {
            key: parsed.metadata.get(key)
            for key in (
                "pages",
                "formula_markers_total",
                "formula_replaced_by_docling_text",
                "formula_unresolved",
                "pdf_tables_count",
                "pdf_figures_count",
            )
            if key in parsed.metadata
        },
    }


def _parse_document_task(path: str, workload: str, index: int) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in _SUFFIXES:
        return {
            "path": path,
            "status": "skipped",
            "reason": f"Unsupported suffix for parser benchmark: {suffix}",
        }

    parser: Any | None = None
    try:
        if suffix == ".epub":
            if workload not in {"auto", "epub_full"}:
                return {
                    "path": path,
                    "status": "skipped",
                    "reason": f"Workload {workload} does not parse EPUB files.",
                }
            from mcp_ebook_read.parsers.epub_ebooklib import EbooklibEpubParser

            parser = EbooklibEpubParser()
        elif workload == "pdf_fast":
            from mcp_ebook_read.parsers.pdf_pypdfium2 import Pypdfium2PdfParser

            parser = Pypdfium2PdfParser()
        elif workload in {"auto", "pdf_fidelity"}:
            from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser

            parser = DoclingPdfParser()
        else:
            return {
                "path": path,
                "status": "skipped",
                "reason": f"Workload {workload} does not parse PDF files.",
            }

        parsed = parser.parse(path, _doc_id(path, index, workload))
        payload = _parsed_metrics(parsed, elapsed_seconds=time.perf_counter() - started)
        payload["path"] = path
        payload["workload"] = workload
        return payload
    except Exception as exc:  # noqa: BLE001
        return {
            "path": path,
            "status": "error",
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "error": _error_payload(exc),
        }
    finally:
        close = getattr(parser, "close", None)
        if callable(close):
            close()


def _run_sequential(tasks: list[_Task], runner: TaskRunner) -> list[dict[str, Any]]:
    return [runner(task.path, task.workload, task.index) for task in tasks]


def _run_thread(
    tasks: list[_Task], runner: TaskRunner, *, max_workers: int
) -> list[dict[str, Any]]:
    results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_index = {
            executor.submit(runner, task.path, task.workload, task.index): task.index
            for task in tasks
        }
        for future in as_completed(future_by_index):
            results.append((future_by_index[future], future.result()))
    return [item for _, item in sorted(results, key=lambda pair: pair[0])]


def _run_process(tasks: list[_Task], *, max_workers: int) -> list[dict[str, Any]]:
    results: list[tuple[int, dict[str, Any]]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_by_index = {
            executor.submit(
                _parse_document_task, task.path, task.workload, task.index
            ): task.index
            for task in tasks
        }
        for future in as_completed(future_by_index):
            index = future_by_index[future]
            try:
                results.append((index, future.result()))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    (
                        index,
                        {
                            "path": tasks[index].path,
                            "status": "error",
                            "error": _error_payload(exc),
                        },
                    )
                )
    return [item for _, item in sorted(results, key=lambda pair: pair[0])]


def _bocpy_parse_behavior(task_state, results_state):  # noqa: ANN001
    task = task_state.value
    payload = _parse_document_task(
        str(task["path"]),
        str(task["workload"]),
        int(task["index"]),
    )
    results_state.value.append((int(task["index"]), payload))


def _run_bocpy(
    tasks: list[_Task], runner: TaskRunner, *, max_workers: int
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    _ = max_workers
    if runner is not _parse_document_task:
        return None, {
            "status": "skipped",
            "reason": "bocpy backend requires the default importable parser task runner.",
        }
    try:
        from bocpy import Cown, start, wait, when  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, {
            "status": "skipped",
            "reason": "bocpy is not installed or cannot be imported.",
            "error": _error_payload(exc),
            "hint": "Run with `uvx --with bocpy mcp-ebook-concurrency-benchmark ...` to test bocpy locally.",
        }

    try:
        start(worker_count=max_workers, module=("__main__", None))
        results = Cown([])
        for task in tasks:
            task_cown = Cown(
                {"index": task.index, "path": task.path, "workload": task.workload}
            )
            when(task_cown, results)(_bocpy_parse_behavior)

        wait()
        ordered = sorted(list(results.value), key=lambda pair: pair[0])
        return [item for _, item in ordered], None
    except Exception as exc:  # noqa: BLE001
        return None, {
            "status": "error",
            "reason": "bocpy backend failed while executing parser tasks.",
            "error": _error_payload_with_traceback(exc),
        }


def _summarize_backend(
    *,
    backend: str,
    workload: str,
    elapsed_seconds: float,
    max_workers: int,
    documents: list[dict[str, Any]],
    backend_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if backend_error is not None:
        return {
            "backend": backend,
            "workload": workload,
            "status": backend_error.get("status", "error"),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "max_workers": max_workers,
            **backend_error,
        }
    ok = [item for item in documents if item.get("status") == "ok"]
    skipped = [item for item in documents if item.get("status") == "skipped"]
    failed = [item for item in documents if item.get("status") == "error"]
    total_text_chars = sum(int(item.get("text_chars") or 0) for item in ok)
    return {
        "backend": backend,
        "workload": workload,
        "status": "ok" if not failed else "completed_with_errors",
        "elapsed_seconds": round(elapsed_seconds, 6),
        "max_workers": max_workers,
        "documents_total": len(documents),
        "documents_ok": len(ok),
        "documents_skipped": len(skipped),
        "documents_failed": len(failed),
        "throughput_docs_per_second": round(len(ok) / elapsed_seconds, 6)
        if elapsed_seconds > 0
        else None,
        "throughput_text_chars_per_second": round(total_text_chars / elapsed_seconds, 6)
        if elapsed_seconds > 0
        else None,
        "totals": {
            "text_chars": total_text_chars,
            "chunks": sum(int(item.get("chunks_count") or 0) for item in ok),
            "formulas": sum(int(item.get("formulas_count") or 0) for item in ok),
            "images": sum(int(item.get("images_count") or 0) for item in ok),
            "tables": sum(int(item.get("tables_count") or 0) for item in ok),
            "figures": sum(int(item.get("figures_count") or 0) for item in ok),
        },
        "documents": documents,
    }


def _recommendation(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [result for result in results if result.get("status") == "ok"]
    if not ok_results:
        return {
            "decision": "insufficient_data",
            "reason": "No backend completed without task errors.",
        }
    baseline = next(
        (result for result in ok_results if result.get("backend") == "sequential"),
        None,
    )
    fastest = min(ok_results, key=lambda item: float(item["elapsed_seconds"]))
    payload = {
        "decision": "prefer_fastest_measured_backend",
        "backend": fastest["backend"],
        "elapsed_seconds": fastest["elapsed_seconds"],
    }
    if baseline is not None and fastest is not baseline:
        baseline_seconds = float(baseline["elapsed_seconds"])
        fastest_seconds = float(fastest["elapsed_seconds"])
        payload["speedup_vs_sequential"] = (
            round(baseline_seconds / fastest_seconds, 6)
            if fastest_seconds > 0
            else None
        )
    if fastest.get("backend") == "bocpy":
        payload["caution"] = (
            "bocpy uses CPython sub-interpreters and version-specific wheels; "
            "validate stability on target machines before making it a default backend."
        )
    return payload


def run_concurrency_benchmark(
    document_paths: list[Path],
    *,
    workload: str = "auto",
    backends: list[str] | None = None,
    max_workers: int | None = None,
    task_runner: TaskRunner | None = None,
) -> dict[str, Any]:
    if workload not in _WORKLOADS:
        raise ValueError(f"Unsupported workload: {workload}")
    selected_backends = backends or ["sequential", "thread", "process", "bocpy"]
    unknown_backends = sorted(set(selected_backends) - _BACKENDS)
    if unknown_backends:
        raise ValueError(f"Unsupported concurrency backends: {unknown_backends}")

    resolved_paths = [
        path.resolve()
        for path in sorted(document_paths)
        if path.is_file() and path.suffix.lower() in _SUFFIXES
    ]
    worker_count = max(1, int(max_workers or min(4, os.cpu_count() or 1)))
    tasks = [
        _Task(index=index, path=str(path), workload=workload)
        for index, path in enumerate(resolved_paths)
    ]
    runner = task_runner or _parse_document_task
    results: list[dict[str, Any]] = []

    for backend in selected_backends:
        started = time.perf_counter()
        backend_error: dict[str, Any] | None = None
        documents: list[dict[str, Any]] = []
        if backend == "sequential":
            documents = _run_sequential(tasks, runner)
        elif backend == "thread":
            documents = _run_thread(tasks, runner, max_workers=worker_count)
        elif backend == "process":
            if task_runner is not None:
                backend_error = {
                    "status": "skipped",
                    "reason": "process backend requires the default pickle-safe task runner.",
                }
            else:
                documents = _run_process(tasks, max_workers=worker_count)
        elif backend == "bocpy":
            documents_or_none, backend_error = _run_bocpy(
                tasks, runner, max_workers=worker_count
            )
            documents = documents_or_none or []
        elapsed = time.perf_counter() - started
        results.append(
            _summarize_backend(
                backend=backend,
                workload=workload,
                elapsed_seconds=elapsed,
                max_workers=worker_count,
                documents=documents,
                backend_error=backend_error,
            )
        )

    return {
        "summary": {
            "workload": workload,
            "documents_total": len(tasks),
            "max_workers": worker_count,
            "backends": selected_backends,
        },
        "recommendation": _recommendation(results),
        "results": results,
    }


def _parse_backends(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-ebook-concurrency-benchmark",
        description=(
            "Compare parser-task concurrency backends on representative EPUB/PDF documents. "
            "This benchmark does not write sidecars."
        ),
    )
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Newline-delimited EPUB/PDF path manifest. Relative paths resolve from the manifest directory.",
    )
    parser.add_argument(
        "--workload",
        choices=sorted(_WORKLOADS),
        default="auto",
        help="Parser workload to schedule across documents.",
    )
    parser.add_argument(
        "--backends",
        default="sequential,thread,process,bocpy",
        help="Comma-separated backend list: sequential,thread,process,bocpy.",
    )
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    documents = collect_documents(
        samples_dir=args.samples_dir,
        manifest=args.manifest,
        suffixes=DOCUMENT_SUFFIXES,
        max_documents=args.max_documents,
    )
    if not documents:
        raise SystemExit(
            "No EPUB/PDF documents found. Pass --samples-dir or --manifest."
        )
    result = run_concurrency_benchmark(
        documents,
        workload=args.workload,
        backends=_parse_backends(args.backends),
        max_workers=args.max_workers or None,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
