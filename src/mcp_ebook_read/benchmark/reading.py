"""No-label reading benchmark for parsed EPUB/PDF structure stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from mcp_ebook_read.benchmark.pdf_formula import _error_payload
from mcp_ebook_read.parsers.epub_ebooklib import EbooklibEpubParser
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.schema.models import ChunkRecord, ParsedDocument

_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+")


class ReadingParserProtocol(Protocol):
    """Type protocol for parsers used by no-label reading benchmark."""

    def parse(self, path: str, doc_id: str) -> ParsedDocument:
        """Parse a document into normalized reading structure."""


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


def _collect_documents(samples_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(samples_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".pdf", ".epub"}
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-ebook-reading-benchmark",
        description="Run no-label reading stability benchmark on EPUB/PDF samples.",
    )
    parser.add_argument("--samples-dir", required=True, type=Path)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--min-stability-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_reading_benchmark(
        _collect_documents(args.samples_dir),
        passes=args.passes,
        min_stability_rate=args.min_stability_rate,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if result["thresholds"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
