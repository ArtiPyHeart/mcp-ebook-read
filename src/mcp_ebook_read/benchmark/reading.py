"""No-label reading benchmark for parsed EPUB/PDF structure stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol

from mcp_ebook_read.benchmark.paths import DOCUMENT_SUFFIXES, collect_documents
from mcp_ebook_read.benchmark.pdf_formula import _error_payload
from mcp_ebook_read.parsers.epub_ebooklib import EbooklibEpubParser
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.schema.models import ChunkRecord, ParsedDocument

_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")


class ReadingParserProtocol(Protocol):
    """Type protocol for parsers used by no-label reading benchmark."""

    def parse(self, path: str, doc_id: str) -> ParsedDocument:
        """Parse a document into normalized reading structure."""


class ReadingServiceProtocol(Protocol):
    """Type protocol for service-side reading companion benchmark."""

    def storage_list_sidecars(self, *, root: str, limit: int) -> dict[str, Any]: ...

    def library_explore(
        self, *, root: str, query: str, top_k: int = 12
    ) -> dict[str, Any]: ...

    def get_outline(self, doc_id: str) -> dict[str, Any]: ...

    def document_explore(
        self, doc_id: str, query: str, top_k: int = 8
    ) -> dict[str, Any]: ...

    def document_node(self, doc_id: str, node_id: str) -> dict[str, Any]: ...

    def read_outline_node(
        self,
        *,
        doc_id: str,
        node_id: str,
        out_format: str,
        max_chunks: int = 120,
    ) -> dict[str, Any]: ...

    def epub_list_images(
        self, *, doc_id: str, node_id: str | None, limit: int
    ) -> dict[str, Any]: ...

    def epub_read_image(self, *, doc_id: str, image_id: str) -> dict[str, Any]: ...

    def pdf_list_images(
        self, *, doc_id: str, node_id: str | None, limit: int
    ) -> dict[str, Any]: ...

    def pdf_read_image(self, *, doc_id: str, image_id: str) -> dict[str, Any]: ...

    def pdf_list_tables(
        self, *, doc_id: str, node_id: str | None, limit: int
    ) -> dict[str, Any]: ...

    def pdf_read_table(self, *, doc_id: str, table_id: str) -> dict[str, Any]: ...

    def pdf_list_figures(
        self, *, doc_id: str, node_id: str | None, limit: int
    ) -> dict[str, Any]: ...

    def pdf_read_figure(self, *, doc_id: str, figure_id: str) -> dict[str, Any]: ...

    def pdf_book_list_formulas(
        self, *, doc_id: str, node_id: str | None, limit: int, status: str | None
    ) -> dict[str, Any]: ...

    def pdf_book_read_formula(
        self, *, doc_id: str, formula_id: str
    ) -> dict[str, Any]: ...

    def pdf_paper_list_formulas(
        self, *, doc_id: str, node_id: str | None, limit: int, status: str | None
    ) -> dict[str, Any]: ...

    def pdf_paper_read_formula(
        self, *, doc_id: str, formula_id: str
    ) -> dict[str, Any]: ...


class _DefaultReadingParser:
    def __init__(self) -> None:
        self.pdf_parser = DoclingPdfParser()
        self.epub_parser = EbooklibEpubParser()

    def parse(self, path: str, doc_id: str) -> ParsedDocument:
        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            return self.pdf_parser.parse(path, doc_id)
        if suffix == ".epub":
            return self.epub_parser.parse(path, doc_id)
        raise ValueError(f"Unsupported benchmark document type: {suffix}")


def _stable_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _query_from_chunk(chunk: ChunkRecord) -> str:
    tokens = list(_tokens(chunk.search_text or chunk.text))
    return " ".join(tokens[: min(4, len(tokens))])


def _lexical_search_signature(
    chunks: list[ChunkRecord],
    *,
    max_queries: int,
    top_k: int = 3,
) -> dict[str, Any]:
    queries = [_query_from_chunk(chunk) for chunk in chunks[:max_queries]]
    queries = [query for query in queries if query]
    results: list[dict[str, Any]] = []
    for query in queries:
        query_tokens = _tokens(query)
        scored: list[tuple[int, int, str]] = []
        for chunk in chunks:
            overlap = len(query_tokens & _tokens(chunk.search_text or chunk.text))
            if overlap:
                scored.append((-overlap, chunk.order_index, chunk.chunk_id))
        scored.sort()
        results.append(
            {
                "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "chunk_order_indexes": [
                    order_index for _, order_index, _ in scored[:top_k]
                ],
            }
        )
    return {
        "queries_count": len(queries),
        "results": results,
        "signature": _stable_digest({"results": results}),
    }


def summarize_parsed_reading_quality(
    parsed: ParsedDocument,
    *,
    max_search_queries: int = 3,
) -> dict[str, Any]:
    outline_payload = [
        {"id": node.id, "title": node.title, "level": node.level}
        for node in parsed.outline
    ]
    search_replay = _lexical_search_signature(
        parsed.chunks,
        max_queries=max_search_queries,
    )
    payload = {
        "outline": outline_payload,
        "chunks": [
            {
                "order_index": chunk.order_index,
                "section_path": chunk.section_path,
            }
            for chunk in parsed.chunks
        ],
        "formulas_count": len(parsed.formulas),
        "images_count": len(parsed.images),
        "tables_count": int(parsed.metadata.get("pdf_tables_count") or 0),
        "search_replay_signature": search_replay["signature"],
    }
    return {
        "outline_nodes_count": len(parsed.outline),
        "outline_max_depth": max((node.level for node in parsed.outline), default=0),
        "chunks_count": len(parsed.chunks),
        "formulas_count": len(parsed.formulas),
        "images_count": len(parsed.images),
        "tables_count": payload["tables_count"],
        "search_replay": search_replay,
        "reading_signature": _stable_digest(payload),
    }


def run_reading_benchmark(
    document_paths: list[Path],
    *,
    parser: ReadingParserProtocol | None = None,
    passes: int = 2,
    min_stability_rate: float = 1.0,
) -> dict[str, Any]:
    runner = parser or _DefaultReadingParser()
    pass_count = max(1, int(passes))
    documents: list[dict[str, Any]] = []

    for path in sorted(document_paths):
        path_str = str(path.resolve())
        if not path.exists():
            documents.append(
                {
                    "path": path_str,
                    "status": "error",
                    "error": {
                        "type": "FileNotFoundError",
                        "message": f"Document not found: {path_str}",
                    },
                }
            )
            continue

        pass_results: list[dict[str, Any]] = []
        signatures: list[str] = []
        document_error: dict[str, Any] | None = None
        for pass_index in range(pass_count):
            doc_id = hashlib.sha1(
                f"{path_str}:{pass_index}".encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:16]
            try:
                parsed = runner.parse(path_str, doc_id)
                metrics = summarize_parsed_reading_quality(parsed)
                metrics["pass_index"] = pass_index
                pass_results.append(metrics)
                signatures.append(metrics["reading_signature"])
            except Exception as exc:  # noqa: BLE001
                document_error = _error_payload(exc)
                break

        if document_error is not None:
            documents.append(
                {"path": path_str, "status": "error", "error": document_error}
            )
            continue

        first_pass = pass_results[0]
        documents.append(
            {
                "path": path_str,
                "status": "ok",
                "first_pass": first_pass,
                "stability": {
                    "passes": pass_count,
                    "exact_match": len(set(signatures)) == 1,
                    "unique_signatures": len(set(signatures)),
                },
                "passes": pass_results,
            }
        )

    ok_docs = [item for item in documents if item["status"] == "ok"]
    stability_rate = (
        sum(1 for item in ok_docs if item["stability"]["exact_match"]) / len(ok_docs)
        if ok_docs
        else 0.0
    )
    summary = {
        "docs_total": len(documents),
        "docs_ok": len(ok_docs),
        "docs_failed": len(documents) - len(ok_docs),
        "stability_exact_match_rate": stability_rate,
        "chunks_total": sum(item["first_pass"]["chunks_count"] for item in ok_docs),
        "outline_nodes_total": sum(
            item["first_pass"]["outline_nodes_count"] for item in ok_docs
        ),
        "formulas_total": sum(item["first_pass"]["formulas_count"] for item in ok_docs),
        "images_total": sum(item["first_pass"]["images_count"] for item in ok_docs),
        "tables_total": sum(item["first_pass"]["tables_count"] for item in ok_docs),
    }
    thresholds = {
        "min_stability_rate": min_stability_rate,
        "passed": stability_rate >= min_stability_rate and summary["docs_failed"] == 0,
    }
    return {
        "summary": summary,
        "thresholds": thresholds,
        "documents": documents,
    }


def _enum_token(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").split(".")[-1].lower()


def _ready_documents(
    storage_payload: dict[str, Any], *, max_docs: int
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for sidecar in storage_payload.get("sidecars") or []:
        sidecar_path = sidecar.get("sidecar_path")
        for doc in sidecar.get("documents") or []:
            if _enum_token(doc.get("status")) != "ready":
                continue
            payload = dict(doc)
            payload["sidecar_path"] = sidecar_path
            docs.append(payload)
            if len(docs) >= max_docs:
                return docs
    return docs


def _task_result(
    name: str,
    *,
    status: str,
    elapsed_seconds: float = 0.0,
    summary: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 6),
    }
    if summary is not None:
        payload["summary"] = summary
    if error is not None:
        payload["error"] = error
    return payload


def _record_call(
    tasks: list[dict[str, Any]],
    name: str,
    call: Any,
    validator: Any | None = None,
    summarizer: Any | None = None,
) -> Any:
    started = time.perf_counter()
    try:
        payload = call()
        if validator is not None and not validator(payload):
            tasks.append(
                _task_result(
                    name,
                    status="failed",
                    elapsed_seconds=time.perf_counter() - started,
                    summary=summarizer(payload) if summarizer else None,
                    error={
                        "type": "AssertionError",
                        "message": "Tool output failed benchmark validation.",
                    },
                )
            )
            return payload
        tasks.append(
            _task_result(
                name,
                status="ok",
                elapsed_seconds=time.perf_counter() - started,
                summary=summarizer(payload) if summarizer else None,
            )
        )
        return payload
    except Exception as exc:  # noqa: BLE001
        tasks.append(
            _task_result(
                name,
                status="failed",
                elapsed_seconds=time.perf_counter() - started,
                error=_error_payload(exc),
            )
        )
        return None


def _record_skipped(
    tasks: list[dict[str, Any]],
    name: str,
    *,
    reason: str,
) -> None:
    tasks.append(_task_result(name, status="skipped", summary={"reason": reason}))


def _first_id(items: list[dict[str, Any]], key: str) -> str | None:
    if not items:
        return None
    value = items[0].get(key)
    return str(value) if value else None


def _doc_query(
    doc: dict[str, Any], outline_payload: dict[str, Any], fallback: str
) -> str:
    nodes = outline_payload.get("nodes") or []
    if nodes and nodes[0].get("title"):
        return str(nodes[0]["title"])
    path = str(doc.get("path") or "")
    if path:
        return Path(path).stem
    return fallback


def _evaluate_pdf_evidence(
    service: ReadingServiceProtocol,
    *,
    doc: dict[str, Any],
    tasks: list[dict[str, Any]],
    evidence_counts: dict[str, int],
) -> None:
    doc_id = str(doc["doc_id"])
    profile = _enum_token(doc.get("profile"))
    if profile == "paper":
        list_formula = service.pdf_paper_list_formulas
        read_formula = service.pdf_paper_read_formula
        formula_prefix = "pdf_paper"
    else:
        list_formula = service.pdf_book_list_formulas
        read_formula = service.pdf_book_read_formula
        formula_prefix = "pdf_book"

    formulas_payload = _record_call(
        tasks,
        f"{formula_prefix}_list_formulas",
        lambda: list_formula(doc_id=doc_id, node_id=None, limit=3, status=None),
        validator=lambda payload: isinstance(payload.get("formulas"), list),
        summarizer=lambda payload: {
            "formulas_count": payload.get("formulas_count"),
            "returned": len(payload.get("formulas") or []),
        },
    )
    formula_id = _first_id((formulas_payload or {}).get("formulas") or [], "formula_id")
    if formula_id:
        evidence_counts["formula_lists_nonempty"] += 1
        read_payload = _record_call(
            tasks,
            f"{formula_prefix}_read_formula",
            lambda: read_formula(doc_id=doc_id, formula_id=formula_id),
            validator=lambda payload: bool((payload.get("formula") or {}).get("latex")),
            summarizer=lambda payload: {
                "formula_id": (payload.get("formula") or {}).get("formula_id"),
                "has_evidence": payload.get("evidence") is not None,
            },
        )
        if read_payload is not None:
            evidence_counts["evidence_reads_ok"] += 1
    else:
        _record_skipped(
            tasks,
            f"{formula_prefix}_read_formula",
            reason="No formulas returned by list tool.",
        )

    image_payload = _record_call(
        tasks,
        "pdf_list_images",
        lambda: service.pdf_list_images(doc_id=doc_id, node_id=None, limit=3),
        validator=lambda payload: isinstance(payload.get("images"), list),
        summarizer=lambda payload: {
            "images_count": payload.get("images_count"),
            "returned": len(payload.get("images") or []),
        },
    )
    image_id = _first_id((image_payload or {}).get("images") or [], "image_id")
    if image_id:
        evidence_counts["image_lists_nonempty"] += 1
        read_payload = _record_call(
            tasks,
            "pdf_read_image",
            lambda: service.pdf_read_image(doc_id=doc_id, image_id=image_id),
            validator=lambda payload: bool(payload.get("image")),
            summarizer=lambda payload: {
                "image_id": (payload.get("image") or {}).get("image_id"),
                "has_context": payload.get("context") is not None,
            },
        )
        if read_payload is not None:
            evidence_counts["evidence_reads_ok"] += 1
    else:
        _record_skipped(tasks, "pdf_read_image", reason="No PDF images returned.")

    table_payload = _record_call(
        tasks,
        "pdf_list_tables",
        lambda: service.pdf_list_tables(doc_id=doc_id, node_id=None, limit=3),
        validator=lambda payload: isinstance(payload.get("tables"), list),
        summarizer=lambda payload: {
            "tables_count": payload.get("tables_count"),
            "returned": len(payload.get("tables") or []),
        },
    )
    table_id = _first_id((table_payload or {}).get("tables") or [], "table_id")
    if table_id:
        evidence_counts["table_lists_nonempty"] += 1
        read_payload = _record_call(
            tasks,
            "pdf_read_table",
            lambda: service.pdf_read_table(doc_id=doc_id, table_id=table_id),
            validator=lambda payload: bool(payload.get("table")),
            summarizer=lambda payload: {
                "table_id": (payload.get("table") or {}).get("table_id"),
                "has_context": payload.get("context") is not None,
            },
        )
        if read_payload is not None:
            evidence_counts["evidence_reads_ok"] += 1
    else:
        _record_skipped(tasks, "pdf_read_table", reason="No PDF tables returned.")

    figure_payload = _record_call(
        tasks,
        "pdf_list_figures",
        lambda: service.pdf_list_figures(doc_id=doc_id, node_id=None, limit=3),
        validator=lambda payload: isinstance(payload.get("figures"), list),
        summarizer=lambda payload: {
            "figures_count": payload.get("figures_count"),
            "returned": len(payload.get("figures") or []),
        },
    )
    figure_id = _first_id((figure_payload or {}).get("figures") or [], "figure_id")
    if figure_id:
        evidence_counts["figure_lists_nonempty"] += 1
        read_payload = _record_call(
            tasks,
            "pdf_read_figure",
            lambda: service.pdf_read_figure(doc_id=doc_id, figure_id=figure_id),
            validator=lambda payload: bool(payload.get("figure")),
            summarizer=lambda payload: {
                "figure_id": (payload.get("figure") or {}).get("figure_id"),
                "has_context": payload.get("context") is not None,
            },
        )
        if read_payload is not None:
            evidence_counts["evidence_reads_ok"] += 1
    else:
        _record_skipped(tasks, "pdf_read_figure", reason="No PDF figures returned.")


def _evaluate_epub_evidence(
    service: ReadingServiceProtocol,
    *,
    doc_id: str,
    tasks: list[dict[str, Any]],
    evidence_counts: dict[str, int],
) -> None:
    image_payload = _record_call(
        tasks,
        "epub_list_images",
        lambda: service.epub_list_images(doc_id=doc_id, node_id=None, limit=3),
        validator=lambda payload: isinstance(payload.get("images"), list),
        summarizer=lambda payload: {
            "images_count": payload.get("images_count"),
            "returned": len(payload.get("images") or []),
        },
    )
    image_id = _first_id((image_payload or {}).get("images") or [], "image_id")
    if image_id:
        evidence_counts["image_lists_nonempty"] += 1
        read_payload = _record_call(
            tasks,
            "epub_read_image",
            lambda: service.epub_read_image(doc_id=doc_id, image_id=image_id),
            validator=lambda payload: bool(payload.get("image")),
            summarizer=lambda payload: {
                "image_id": (payload.get("image") or {}).get("image_id"),
                "has_context": payload.get("context") is not None,
            },
        )
        if read_payload is not None:
            evidence_counts["evidence_reads_ok"] += 1
    else:
        _record_skipped(tasks, "epub_read_image", reason="No EPUB images returned.")


def run_reading_service_benchmark(
    root: Path,
    *,
    service: ReadingServiceProtocol | None = None,
    query: str = "formula figure table introduction method results",
    top_k: int = 8,
    max_docs: int = 20,
    min_task_pass_rate: float = 1.0,
) -> dict[str, Any]:
    """Run a product-path benchmark over existing sidecar catalogs."""

    if service is None:
        from mcp_ebook_read.service import AppService

        service = AppService.from_env()

    root_path = root.expanduser().resolve()
    started = time.perf_counter()
    root_tasks: list[dict[str, Any]] = []
    storage_payload = _record_call(
        root_tasks,
        "storage_list_sidecars",
        lambda: service.storage_list_sidecars(root=str(root_path), limit=max_docs),
        validator=lambda payload: int(payload.get("sidecars_count") or 0) > 0,
        summarizer=lambda payload: {
            "sidecars_count": payload.get("sidecars_count"),
            "documents_count": payload.get("documents_count"),
        },
    )
    storage_payload = storage_payload or {"sidecars": [], "documents_count": 0}
    ready_docs = _ready_documents(storage_payload, max_docs=max_docs)

    library_payload = _record_call(
        root_tasks,
        "library_explore",
        lambda: service.library_explore(root=str(root_path), query=query, top_k=top_k),
        validator=lambda payload: (
            int((payload.get("retrieval") or {}).get("searched_documents_count") or 0)
            > 0
        ),
        summarizer=lambda payload: {
            "documents": len(payload.get("documents") or []),
            "selected_results": len(payload.get("selected_results") or []),
            "hits": len(payload.get("hits") or []),
            "diagnostics": len(payload.get("diagnostics") or []),
        },
    )

    evidence_counts = {
        "formula_lists_nonempty": 0,
        "image_lists_nonempty": 0,
        "table_lists_nonempty": 0,
        "figure_lists_nonempty": 0,
        "evidence_reads_ok": 0,
    }
    documents: list[dict[str, Any]] = []

    for doc in ready_docs:
        doc_id = str(doc["doc_id"])
        doc_tasks: list[dict[str, Any]] = []
        outline_payload = _record_call(
            doc_tasks,
            "get_outline",
            lambda doc_id=doc_id: service.get_outline(doc_id),
            validator=lambda payload: isinstance(payload.get("nodes"), list),
            summarizer=lambda payload: {
                "nodes": len(payload.get("nodes") or []),
                "title": payload.get("title"),
            },
        )
        outline_payload = outline_payload or {"nodes": []}
        nodes = outline_payload.get("nodes") or []
        outline_node_id = str(nodes[0]["id"]) if nodes and nodes[0].get("id") else None
        doc_query = _doc_query(doc, outline_payload, query)
        explore_payload = _record_call(
            doc_tasks,
            "document_explore",
            lambda doc_id=doc_id, doc_query=doc_query: service.document_explore(
                doc_id, doc_query, top_k=top_k
            ),
            validator=lambda payload: bool(payload.get("document")),
            summarizer=lambda payload: {
                "selected_nodes": len(payload.get("selected_nodes") or []),
                "hits": len(payload.get("hits") or []),
                "diagnostics": len(payload.get("diagnostics") or []),
            },
        )
        selected_nodes = (explore_payload or {}).get("selected_nodes") or []
        selected_node_id = (
            str(selected_nodes[0]["node_id"])
            if selected_nodes and selected_nodes[0].get("node_id")
            else None
        )
        if selected_node_id:
            _record_call(
                doc_tasks,
                "document_node",
                lambda doc_id=doc_id, selected_node_id=selected_node_id: (
                    service.document_node(doc_id, selected_node_id)
                ),
                validator=lambda payload: bool(payload.get("node")),
                summarizer=lambda payload: {
                    "node_id": (payload.get("node") or {}).get("node_id"),
                    "neighbors": len(payload.get("neighbors") or []),
                },
            )
        else:
            _record_skipped(
                doc_tasks,
                "document_node",
                reason="document_explore returned no selected nodes.",
            )

        if outline_node_id:
            _record_call(
                doc_tasks,
                "read_outline_node",
                lambda doc_id=doc_id, outline_node_id=outline_node_id: (
                    service.read_outline_node(
                        doc_id=doc_id,
                        node_id=outline_node_id,
                        out_format="markdown",
                        max_chunks=20,
                    )
                ),
                validator=lambda payload: bool(payload.get("content")),
                summarizer=lambda payload: {
                    "chunks_count": payload.get("chunks_count"),
                    "truncated": payload.get("truncated"),
                },
            )
        else:
            _record_skipped(
                doc_tasks,
                "read_outline_node",
                reason="Document has no outline nodes.",
            )

        doc_type = _enum_token(doc.get("type"))
        if doc_type == "pdf":
            _evaluate_pdf_evidence(
                service,
                doc=doc,
                tasks=doc_tasks,
                evidence_counts=evidence_counts,
            )
        elif doc_type == "epub":
            _evaluate_epub_evidence(
                service,
                doc_id=doc_id,
                tasks=doc_tasks,
                evidence_counts=evidence_counts,
            )
        else:
            _record_skipped(
                doc_tasks,
                "evidence_tools",
                reason=f"Unsupported document type for evidence smoke: {doc_type}",
            )

        failed = sum(1 for task in doc_tasks if task["status"] == "failed")
        documents.append(
            {
                "doc_id": doc_id,
                "path": doc.get("path"),
                "type": doc_type,
                "profile": _enum_token(doc.get("profile")),
                "sidecar_path": doc.get("sidecar_path"),
                "tasks": doc_tasks,
                "tasks_total": len(doc_tasks),
                "tasks_failed": failed,
                "tasks_ok": sum(1 for task in doc_tasks if task["status"] == "ok"),
                "tasks_skipped": sum(
                    1 for task in doc_tasks if task["status"] == "skipped"
                ),
            }
        )

    all_tasks = root_tasks + [
        task for document in documents for task in document["tasks"]
    ]
    tasks_total = len(all_tasks)
    tasks_failed = sum(1 for task in all_tasks if task["status"] == "failed")
    tasks_ok = sum(1 for task in all_tasks if task["status"] == "ok")
    tasks_skipped = sum(1 for task in all_tasks if task["status"] == "skipped")
    evaluated_tasks = max(1, tasks_total - tasks_skipped)
    task_pass_rate = tasks_ok / evaluated_tasks
    storage_diagnostics_count = sum(
        int(sidecar.get("diagnostics_count") or 0)
        for sidecar in storage_payload.get("sidecars") or []
    )
    summary = {
        "mode": "service_sidecar",
        "root": str(root_path),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "sidecars_count": storage_payload.get("sidecars_count", 0),
        "documents_total": storage_payload.get("documents_count", 0),
        "documents_ready": len(ready_docs),
        "documents_evaluated": len(documents),
        "tasks_total": tasks_total,
        "tasks_ok": tasks_ok,
        "tasks_failed": tasks_failed,
        "tasks_skipped": tasks_skipped,
        "task_pass_rate": task_pass_rate,
        "storage_diagnostics_count": storage_diagnostics_count,
        "library_selected_results": len(
            (library_payload or {}).get("selected_results") or []
        ),
        **evidence_counts,
    }
    thresholds = {
        "min_task_pass_rate": min_task_pass_rate,
        "passed": (
            len(ready_docs) > 0
            and tasks_failed == 0
            and task_pass_rate >= min_task_pass_rate
        ),
    }
    return {
        "summary": summary,
        "thresholds": thresholds,
        "root_tasks": root_tasks,
        "documents": documents,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-ebook-reading-benchmark",
        description=(
            "Run no-label reading stability benchmark on EPUB/PDF samples, or "
            "service-side reading companion smoke checks over existing sidecars."
        ),
    )
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Newline-delimited EPUB/PDF path manifest. Relative paths resolve from the manifest directory.",
    )
    parser.add_argument(
        "--service-root",
        type=Path,
        help="Run service-side benchmark against existing .mcp-ebook-read sidecars under this root.",
    )
    parser.add_argument(
        "--query",
        default="formula figure table introduction method results",
        help="Reading query used for service-side library/document exploration.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--min-stability-rate", type=float, default=1.0)
    parser.add_argument("--min-task-pass-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.service_root:
        result = run_reading_service_benchmark(
            args.service_root,
            query=args.query,
            top_k=args.top_k,
            max_docs=args.max_docs,
            min_task_pass_rate=args.min_task_pass_rate,
        )
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return 0 if result["thresholds"]["passed"] else 2

    document_paths = collect_documents(
        samples_dir=args.samples_dir,
        manifest=args.manifest,
        suffixes=DOCUMENT_SUFFIXES,
    )
    if not document_paths:
        raise SystemExit(
            "No EPUB/PDF documents found. Pass --samples-dir or --manifest."
        )
    result = run_reading_benchmark(
        document_paths,
        passes=args.passes,
        min_stability_rate=args.min_stability_rate,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if result["thresholds"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
