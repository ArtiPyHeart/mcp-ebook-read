"""Practical parser-engine benchmark for EPUB/PDF candidates."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import queue
import re
import time
import traceback
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


PDF_SUFFIX = ".pdf"
EPUB_SUFFIX = ".epub"
SUPPORTED_SUFFIXES = {PDF_SUFFIX, EPUB_SUFFIX}
ALL_ENGINES = (
    "project_pdf_docling",
    "docling_raw_pdf_no_formula",
    "project_epub_ebooklib",
    "pymupdf_pdf",
    "pypdfium2_pdf",
    "pdf_oxide_pdf",
    "epub_zip_lxml",
)

SMOKE_RELATIVE_PATHS = (
    "01-高频 MM 报价控制模型/04-Optimal market making.pdf",
    "补充材料-论文/DeepLOB.pdf",
    "books-strategy/06-Limit Order Books.pdf",
    "books-system/Rust atomics and locks _ low-level concurrency in practice.epub",
)

DEFAULT_RELATIVE_PATHS = (
    "01-高频 MM 报价控制模型/04-Optimal market making.pdf",
    "01-高频 MM 报价控制模型/02-High-frequency trading in a limit order book.pdf",
    "02-订单簿 alpha、fair price、短期方向/02-The Micro-Price.pdf",
    "02-订单簿 alpha、fair price、短期方向/04-Hawkes processes in finance.pdf",
    "03-fill probability、queue position、订单簿仿真/02-Simulating and analyzing order book data- The queue-reactive model.pdf",
    "04-反向问题-toxic flow、adverse selection、latency arbitrage/04-Flow Toxicity and Liquidity in a High Frequency World.pdf",
    "补充材料-论文/DeepLOB.pdf",
    "补充材料-论文/The Probability of Backtest Overfitting.pdf",
    "books-strategy/02-The Financial Mathematics of Market Liquidity.pdf",
    "books-strategy/06-Limit Order Books.pdf",
    "books-system/Performance Analysis and Tuning on Modern CPUs_ Learn to write fast software like a pro.pdf",
    "books-strategy/01-Algorithmic and High-Frequency Trading.epub",
    "books-strategy/03-Trades, quotes and prices.epub",
    "books-system/Rust atomics and locks _ low-level concurrency in practice.epub",
    "补充材料/Advances in financial machine learning,López de Prado.epub",
)

DEFAULT_READING_QUERIES = (
    "market making",
    "inventory risk",
    "limit order book",
    "micro price",
    "Hawkes process",
    "queue reactive model",
    "flow toxicity",
    "DeepLOB",
    "backtest overfitting",
    "CPU cache",
    "Rust atomics",
)


def _sha1_short(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token.lower() for token in re.findall(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", value)
    )


def _score_query_against_text(query: str, text: str) -> dict[str, Any]:
    query_tokens = set(_lexical_tokens(query))
    text_tokens = set(_lexical_tokens(text))
    overlap = query_tokens & text_tokens
    overlap_ratio = _safe_div(len(overlap), len(query_tokens)) or 0.0
    normalized_query = _normalize_text(query).lower()
    normalized_text = _normalize_text(text).lower()
    phrase_hit = bool(normalized_query and normalized_query in normalized_text)
    return {
        "score": overlap_ratio + (1.0 if phrase_hit else 0.0),
        "overlap_ratio": overlap_ratio,
        "matched_tokens": sorted(overlap),
        "phrase_hit": phrase_hit,
        "hit": phrase_hit or overlap_ratio >= 0.66,
    }


def _preview_for_query(query: str, text: str, *, max_chars: int = 240) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized

    lower = normalized.lower()
    candidates = [_normalize_text(query).lower(), *_lexical_tokens(query)]
    start = 0
    for candidate in candidates:
        if not candidate:
            continue
        index = lower.find(candidate)
        if index >= 0:
            start = max(0, index - max_chars // 3)
            break
    end = min(len(normalized), start + max_chars)
    preview = normalized[start:end].strip()
    if start > 0:
        preview = "..." + preview
    if end < len(normalized):
        preview += "..."
    return preview


def _query_replay_from_chunks(chunks: list[Any], queries: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        best: dict[str, Any] | None = None
        best_text = ""
        for chunk in chunks:
            text = str(getattr(chunk, "text", "") or "")
            score = _score_query_against_text(query, text)
            if best is None or float(score["score"]) > float(best["score"]):
                locator = getattr(chunk, "locator", None)
                best = {
                    **score,
                    "chunk_id": getattr(chunk, "chunk_id", None),
                    "order_index": getattr(chunk, "order_index", None),
                    "section_path": list(getattr(chunk, "section_path", []) or []),
                    "page_range": getattr(locator, "page_range", None),
                }
                best_text = text
        if best is None:
            best = {
                "score": 0.0,
                "overlap_ratio": 0.0,
                "matched_tokens": [],
                "phrase_hit": False,
                "hit": False,
                "chunk_id": None,
                "order_index": None,
                "section_path": [],
                "page_range": None,
            }
        rows.append(
            {
                "query": query,
                "hit": bool(best["hit"]),
                "best_score": round(float(best["score"]), 6),
                "best_overlap_ratio": round(float(best["overlap_ratio"]), 6),
                "phrase_hit": bool(best["phrase_hit"]),
                "matched_tokens": list(best["matched_tokens"]),
                "best_chunk": {
                    "chunk_id": best["chunk_id"],
                    "order_index": best["order_index"],
                    "section_path": best["section_path"],
                    "page_range": best["page_range"],
                    "preview": _preview_for_query(query, best_text)
                    if best_text
                    else "",
                },
            }
        )

    best_overlap_sum = sum(float(row["best_overlap_ratio"]) for row in rows)
    best_score_sum = sum(float(row["best_score"]) for row in rows)
    hits = sum(1 for row in rows if row["hit"])
    return {
        "queries_total": len(rows),
        "queries_with_hit": hits,
        "hit_rate": _safe_div(hits, len(rows)),
        "avg_best_overlap_ratio": _safe_div(best_overlap_sum, len(rows)),
        "avg_best_score": _safe_div(best_score_sum, len(rows)),
        "queries": rows,
    }


def _query_replay_from_text(text: str, queries: list[str]) -> dict[str, Any]:
    pseudo_chunk = type(
        "BenchmarkChunk",
        (),
        {
            "text": text,
            "chunk_id": None,
            "order_index": 0,
            "section_path": [],
            "locator": None,
        },
    )()
    return _query_replay_from_chunks([pseudo_chunk], queries)


def _metric_summary_from_text(text: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    return {
        "text_chars": len(text),
        "normalized_text_chars": len(normalized),
        "normalized_text_sha256": hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest(),
        "wordish_tokens": len(re.findall(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", text)),
    }


def _len_attr(value: Any, attr: str) -> int:
    item = getattr(value, attr, None)
    if item is None:
        return 0
    try:
        return len(item)
    except TypeError:
        return 0


def _docling_document_metrics(document: Any) -> dict[str, Any]:
    pictures = _len_attr(document, "pictures")
    return {
        "pages": _len_attr(document, "pages"),
        "tables": _len_attr(document, "tables"),
        "figures": pictures,
        "images": pictures,
        "docling_text_items": _len_attr(document, "texts"),
        "docling_groups": _len_attr(document, "groups"),
    }


def _parsed_document_metrics(
    parsed: Any, *, queries: list[str] | None = None
) -> dict[str, Any]:
    metadata = dict(getattr(parsed, "metadata", {}) or {})
    chunks = list(getattr(parsed, "chunks", []) or [])
    outline = list(getattr(parsed, "outline", []) or [])
    formulas = list(getattr(parsed, "formulas", []) or [])
    images = list(getattr(parsed, "images", []) or [])
    reading_markdown = str(getattr(parsed, "reading_markdown", "") or "")
    chunk_text = "\n".join(str(getattr(chunk, "text", "") or "") for chunk in chunks)
    chunk_text_chars = sum(
        len(str(getattr(chunk, "text", "") or "")) for chunk in chunks
    )
    formula_unresolved = int(metadata.get("formula_unresolved") or 0)
    formula_markers_total = int(metadata.get("formula_markers_total") or 0)
    metrics = {
        "title": str(getattr(parsed, "title", "") or ""),
        "parser_chain": list(getattr(parsed, "parser_chain", []) or []),
        "pages": metadata.get("pages"),
        "outline_nodes": len(outline),
        "chunks": len(chunks),
        "formulas": len(formulas),
        "images": len(images),
        "tables": int(metadata.get("pdf_tables_count") or 0),
        "figures": int(metadata.get("pdf_figures_count") or 0),
        "formula_markers_total": formula_markers_total,
        "formula_unresolved": formula_unresolved,
        "formula_unresolved_rate": _safe_div(
            formula_unresolved,
            formula_markers_total,
        ),
        "reading_markdown_chars": len(reading_markdown),
        "chunk_text_chars": chunk_text_chars,
        "metadata": {
            key: metadata.get(key)
            for key in (
                "toc_nodes_raw",
                "toc_nodes_clean",
                "docling_formula_enrichment_enabled",
                "formula_replaced_by_docling_text",
                "formula_replaced_by_pix2text",
                "formula_replaced_by_fallback",
                "formula_records_total",
                "epub_spine_documents",
                "epub_images_total",
                "epub_raw_artifacts_total",
            )
            if key in metadata
        },
    }
    metrics.update(_metric_summary_from_text(chunk_text or reading_markdown))
    if queries:
        metrics["query_replay"] = _query_replay_from_chunks(chunks, queries)
    return metrics


def _run_project_pdf_docling(
    path: Path, queries: list[str] | None = None
) -> dict[str, Any]:
    if path.suffix.lower() != PDF_SUFFIX:
        return {"status": "skipped", "reason": "PDF-only engine"}
    from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser

    parser = DoclingPdfParser()
    try:
        parsed = parser.parse(str(path), f"bench-{_sha1_short(str(path))}")
        return {
            "status": "ok",
            "metrics": _parsed_document_metrics(parsed, queries=queries),
        }
    finally:
        parser.close()


def _run_docling_raw_pdf_no_formula(
    path: Path, queries: list[str] | None = None
) -> dict[str, Any]:
    if path.suffix.lower() != PDF_SUFFIX:
        return {"status": "skipped", "reason": "PDF-only engine"}
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_formula_enrichment = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )
    result = converter.convert(str(path))
    document = result.document
    markdown = str(document.export_to_markdown() or "")
    metrics = {"title": path.stem}
    metrics.update(_metric_summary_from_text(markdown))
    metrics.update(_docling_document_metrics(document))
    if queries:
        metrics["query_replay"] = _query_replay_from_text(markdown, queries)
    metrics["formula_markers_total"] = markdown.count("<!-- formula-not-decoded -->")
    return {"status": "ok", "metrics": metrics}


def _run_project_epub_ebooklib(
    path: Path, queries: list[str] | None = None
) -> dict[str, Any]:
    if path.suffix.lower() != EPUB_SUFFIX:
        return {"status": "skipped", "reason": "EPUB-only engine"}
    from mcp_ebook_read.parsers.epub_ebooklib import EbooklibEpubParser

    parsed = EbooklibEpubParser().parse(str(path), f"bench-{_sha1_short(str(path))}")
    return {
        "status": "ok",
        "metrics": _parsed_document_metrics(parsed, queries=queries),
    }


def _run_pymupdf_pdf(path: Path, queries: list[str] | None = None) -> dict[str, Any]:
    if path.suffix.lower() != PDF_SUFFIX:
        return {"status": "skipped", "reason": "PDF-only engine"}
    import fitz

    text_parts: list[str] = []
    images_total = 0
    block_count = 0
    with fitz.open(str(path)) as doc:
        for page in doc:
            text_parts.append(str(page.get_text("text") or ""))
            images_total += len(page.get_images(full=True))
            try:
                block_count += len(page.get_text("blocks") or [])
            except Exception:  # noqa: BLE001
                pass
        metrics = {
            "title": doc.metadata.get("title") or path.stem,
            "pages": doc.page_count,
            "outline_nodes": len(doc.get_toc() or []),
            "images": images_total,
            "blocks": block_count,
        }
    text = "\n".join(text_parts)
    metrics.update(_metric_summary_from_text(text))
    if queries:
        metrics["query_replay"] = _query_replay_from_text(text, queries)
    return {"status": "ok", "metrics": metrics}


def _run_pypdfium2_pdf(path: Path, queries: list[str] | None = None) -> dict[str, Any]:
    if path.suffix.lower() != PDF_SUFFIX:
        return {"status": "skipped", "reason": "PDF-only engine"}
    from mcp_ebook_read.parsers.pdf_pypdfium2 import Pypdfium2PdfParser

    parsed = Pypdfium2PdfParser().parse(str(path), f"bench-{_sha1_short(str(path))}")
    return {
        "status": "ok",
        "metrics": _parsed_document_metrics(parsed, queries=queries),
    }


def _pdf_oxide_page_count(document: Any) -> int:
    page_count = getattr(document, "page_count", None)
    if callable(page_count):
        return int(page_count())
    if page_count is not None:
        return int(page_count)
    pages = getattr(document, "pages", None)
    if callable(pages):
        return len(list(pages()))
    if pages is not None:
        return len(pages)
    raise RuntimeError("PDF Oxide document does not expose a known page count API")


def _run_pdf_oxide_pdf(path: Path, queries: list[str] | None = None) -> dict[str, Any]:
    if path.suffix.lower() != PDF_SUFFIX:
        return {"status": "skipped", "reason": "PDF-only engine"}
    try:
        from pdf_oxide import PdfDocument
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"pdf_oxide unavailable: {exc}"}

    doc = PdfDocument(str(path))
    try:
        page_count = _pdf_oxide_page_count(doc)
        text_parts = [
            str(doc.extract_text(page_index) or "") for page_index in range(page_count)
        ]
    finally:
        close_doc = getattr(doc, "close", None)
        if callable(close_doc):
            close_doc()

    metrics = {"title": path.stem, "pages": page_count}
    text = "\n".join(text_parts)
    metrics.update(_metric_summary_from_text(text))
    if queries:
        metrics["query_replay"] = _query_replay_from_text(text, queries)
    return {"status": "ok", "metrics": metrics}


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _xml_child_text(root: ElementTree.Element, local_name: str) -> str:
    for node in root.iter():
        if _xml_local_name(node.tag) == local_name and node.text:
            return _normalize_text(node.text)
    return ""


def _find_epub_rootfile(zip_file: zipfile.ZipFile) -> str:
    container = ElementTree.fromstring(zip_file.read("META-INF/container.xml"))
    for node in container.iter():
        if _xml_local_name(node.tag) == "rootfile":
            full_path = node.attrib.get("full-path")
            if full_path:
                return full_path
    raise RuntimeError("EPUB rootfile not found")


def _run_epub_zip_lxml(path: Path, queries: list[str] | None = None) -> dict[str, Any]:
    if path.suffix.lower() != EPUB_SUFFIX:
        return {"status": "skipped", "reason": "EPUB-only engine"}
    from lxml import html

    text_parts: list[str] = []
    image_count = 0
    heading_count = 0
    spine_docs = 0
    with zipfile.ZipFile(path) as archive:
        rootfile = _find_epub_rootfile(archive)
        opf_root = ElementTree.fromstring(archive.read(rootfile))
        opf_dir = str(Path(rootfile).parent)
        if opf_dir == ".":
            opf_dir = ""

        manifest: dict[str, str] = {}
        for node in opf_root.iter():
            if _xml_local_name(node.tag) == "item":
                item_id = node.attrib.get("id")
                href = node.attrib.get("href")
                if item_id and href:
                    manifest[item_id] = href

        spine_ids = [
            node.attrib["idref"]
            for node in opf_root.iter()
            if _xml_local_name(node.tag) == "itemref" and node.attrib.get("idref")
        ]
        title = _xml_child_text(opf_root, "title") or path.stem

        for spine_id in spine_ids:
            href = manifest.get(spine_id)
            if not href:
                continue
            item_path = str(Path(opf_dir, href))
            if item_path not in archive.namelist():
                continue
            raw = archive.read(item_path)
            try:
                node = html.fromstring(raw)
            except Exception:  # noqa: BLE001
                continue
            spine_docs += 1
            text_parts.append(_normalize_text(" ".join(node.xpath("//text()"))))
            image_count += len(node.xpath("//img|//*[local-name()='image']"))
            heading_count += len(node.xpath("//h1|//h2|//h3|//h4|//h5|//h6"))

    metrics = {
        "title": title,
        "spine_documents": spine_docs,
        "images": image_count,
        "headings": heading_count,
    }
    text = "\n".join(text_parts)
    metrics.update(_metric_summary_from_text(text))
    if queries:
        metrics["query_replay"] = _query_replay_from_text(text, queries)
    return {"status": "ok", "metrics": metrics}


ENGINE_RUNNERS: dict[str, Callable[[Path, list[str] | None], dict[str, Any]]] = {
    "project_pdf_docling": _run_project_pdf_docling,
    "docling_raw_pdf_no_formula": _run_docling_raw_pdf_no_formula,
    "project_epub_ebooklib": _run_project_epub_ebooklib,
    "pymupdf_pdf": _run_pymupdf_pdf,
    "pypdfium2_pdf": _run_pypdfium2_pdf,
    "pdf_oxide_pdf": _run_pdf_oxide_pdf,
    "epub_zip_lxml": _run_epub_zip_lxml,
}


def _worker(
    engine: str,
    path_str: str,
    queries: list[str] | None,
    result_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    started = time.perf_counter()
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        runner = ENGINE_RUNNERS[engine]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = runner(Path(path_str), queries)
        result["seconds"] = time.perf_counter() - started
        engine_output = _engine_output_payload(stdout.getvalue(), stderr.getvalue())
        if engine_output:
            result["engine_output"] = engine_output
        result_queue.put(result)
    except Exception as exc:  # noqa: BLE001
        engine_output = _engine_output_payload(stdout.getvalue(), stderr.getvalue())
        result_queue.put(
            {
                "status": "error",
                "seconds": time.perf_counter() - started,
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
                **({"engine_output": engine_output} if engine_output else {}),
            }
        )


def _engine_output_payload(stdout: str, stderr: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if stdout:
        payload["stdout_chars"] = len(stdout)
        payload["stdout_tail"] = stdout[-4000:]
    if stderr:
        payload["stderr_chars"] = len(stderr)
        payload["stderr_tail"] = stderr[-4000:]
    return payload


def _run_engine_with_timeout(
    *,
    engine: str,
    path: Path,
    queries: list[str] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()  # type: ignore[type-arg]
    process = context.Process(
        target=_worker,
        args=(engine, str(path), queries, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "status": "timeout",
            "seconds": timeout_seconds,
            "error": {
                "type": "TimeoutError",
                "message": f"Engine timed out after {timeout_seconds}s",
            },
        }

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return {
            "status": "error",
            "seconds": None,
            "error": {
                "type": "NoResult",
                "message": f"Engine process exited with code {process.exitcode} without result",
            },
        }


def _read_manifest(manifest: Path) -> list[Path]:
    base = manifest.parent
    paths: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        paths.append(candidate.resolve())
    return paths


def _collect_paths(
    *,
    samples_dir: Path,
    preset: str,
    manifest: Path | None,
    paths: list[Path],
) -> list[Path]:
    collected: list[Path] = []
    if manifest is not None:
        collected.extend(_read_manifest(manifest))
    if paths:
        collected.extend(path.expanduser().resolve() for path in paths)
    if not collected:
        relative_paths = (
            SMOKE_RELATIVE_PATHS if preset == "smoke" else DEFAULT_RELATIVE_PATHS
        )
        collected.extend((samples_dir / path).resolve() for path in relative_paths)

    seen: set[Path] = set()
    result: list[Path] = []
    for path in collected:
        if path in seen:
            continue
        seen.add(path)
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            result.append(path)
    return result


def _parse_engines(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_ENGINES)
    engines = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [engine for engine in engines if engine not in ENGINE_RUNNERS]
    if unknown:
        raise ValueError(
            f"Unknown engine(s): {', '.join(unknown)}. Known engines: {', '.join(ALL_ENGINES)}"
        )
    return engines


def run_parser_engine_benchmark(
    *,
    document_paths: list[Path],
    engines: list[str],
    queries: list[str] | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    started = time.perf_counter()
    query_list = list(DEFAULT_READING_QUERIES if queries is None else queries)
    documents: list[dict[str, Any]] = []
    for path in document_paths:
        path_result: dict[str, Any] = {
            "path": str(path),
            "suffix": path.suffix.lower(),
            "exists": path.exists(),
            "engines": [],
        }
        if not path.exists():
            path_result["status"] = "missing"
            documents.append(path_result)
            continue

        for engine in engines:
            engine_result = _run_engine_with_timeout(
                engine=engine,
                path=path,
                queries=query_list,
                timeout_seconds=timeout_seconds,
            )
            engine_result["engine"] = engine
            path_result["engines"].append(engine_result)
        _annotate_document_comparisons(path_result)
        path_result["status"] = "ok"
        documents.append(path_result)

    engine_summary: dict[str, dict[str, Any]] = {}
    for engine in engines:
        rows = [
            engine_result
            for document in documents
            for engine_result in document.get("engines", [])
            if engine_result.get("engine") == engine
        ]
        ok_rows = [row for row in rows if row.get("status") == "ok"]
        seconds = [
            float(row["seconds"])
            for row in ok_rows
            if isinstance(row.get("seconds"), int | float)
        ]
        text_chars = [
            int(row.get("metrics", {}).get("normalized_text_chars") or 0)
            for row in ok_rows
        ]
        query_replays = [
            row.get("metrics", {}).get("query_replay", {})
            for row in ok_rows
            if row.get("metrics", {}).get("query_replay")
        ]
        queries_total = sum(
            int(item.get("queries_total") or 0) for item in query_replays
        )
        queries_with_hit = sum(
            int(item.get("queries_with_hit") or 0) for item in query_replays
        )
        avg_overlap_values = [
            float(item.get("avg_best_overlap_ratio"))
            for item in query_replays
            if isinstance(item.get("avg_best_overlap_ratio"), int | float)
        ]
        engine_summary[engine] = {
            "runs_total": len(rows),
            "ok": len(ok_rows),
            "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
            "errors": sum(1 for row in rows if row.get("status") == "error"),
            "timeouts": sum(1 for row in rows if row.get("status") == "timeout"),
            "seconds_total": sum(seconds),
            "seconds_avg": _safe_div(sum(seconds), len(seconds)),
            "normalized_text_chars_total": sum(text_chars),
            "query_replay_hit_rate": _safe_div(queries_with_hit, queries_total),
            "query_replay_avg_best_overlap_ratio": _safe_div(
                sum(avg_overlap_values), len(avg_overlap_values)
            ),
        }

    return {
        "summary": {
            "documents_total": len(document_paths),
            "engines": engines,
            "queries": query_list,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "engine_summary": engine_summary,
        "documents": documents,
    }


def _annotate_document_comparisons(document: dict[str, Any]) -> None:
    ok_rows = [row for row in document.get("engines", []) if row.get("status") == "ok"]
    if not ok_rows:
        return

    max_chars = max(
        int(row.get("metrics", {}).get("normalized_text_chars") or 0) for row in ok_rows
    )
    by_engine = {str(row.get("engine")): row for row in ok_rows}
    pymupdf_chars = int(
        by_engine.get("pymupdf_pdf", {}).get("metrics", {}).get("normalized_text_chars")
        or 0
    )
    pypdfium2_chars = int(
        by_engine.get("pypdfium2_pdf", {})
        .get("metrics", {})
        .get("normalized_text_chars")
        or 0
    )

    for row in ok_rows:
        metrics = row.get("metrics", {})
        chars = int(metrics.get("normalized_text_chars") or 0)
        metrics["normalized_text_ratio_to_document_max"] = _safe_div(chars, max_chars)
        if pymupdf_chars:
            metrics["normalized_text_ratio_to_pymupdf"] = _safe_div(
                chars, pymupdf_chars
            )
        if pypdfium2_chars:
            metrics["normalized_text_ratio_to_pypdfium2"] = _safe_div(
                chars, pypdfium2_chars
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-ebook-parser-engine-benchmark",
        description="Compare EPUB/PDF parser engines on representative documents.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("tests/samples/more"),
        help="Root directory used by --preset smoke/default.",
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "default"),
        default="smoke",
        help="Representative corpus preset used when no manifest/path is provided.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Newline-delimited document path manifest. Relative paths resolve from the manifest directory.",
    )
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        default=[],
        help="Explicit document path. May be passed multiple times.",
    )
    parser.add_argument(
        "--engines",
        default="all",
        help=f"Comma-separated engines or 'all'. Known: {', '.join(ALL_ENGINES)}.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Reading query used for lexical replay. May be passed multiple times.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Newline-delimited reading queries. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--no-query-replay",
        action="store_true",
        help="Disable reading-query replay metrics.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _read_queries_file(path: Path) -> list[str]:
    queries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            queries.append(line)
    return queries


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        engines = _parse_engines(args.engines)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    document_paths = _collect_paths(
        samples_dir=args.samples_dir.expanduser().resolve(),
        preset=args.preset,
        manifest=args.manifest.expanduser().resolve() if args.manifest else None,
        paths=args.path,
    )
    if args.max_documents and args.max_documents > 0:
        document_paths = document_paths[: args.max_documents]
    if args.no_query_replay:
        queries: list[str] = []
    else:
        queries = list(DEFAULT_READING_QUERIES)
        if args.queries_file:
            queries = _read_queries_file(args.queries_file.expanduser().resolve())
        if args.query:
            queries = [*queries, *args.query]

    result = run_parser_engine_benchmark(
        document_paths=document_paths,
        engines=engines,
        queries=queries,
        timeout_seconds=max(1, int(args.timeout_seconds)),
    )
    result["environment"] = {
        "cwd": os.getcwd(),
        "python": os.sys.version,
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
