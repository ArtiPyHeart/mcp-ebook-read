"""No-label benchmark utilities for PDF formula extraction quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from mcp_ebook_read.errors import AppError
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.schema.models import ParsedDocument

_BLOCK_FORMULA_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_BEGIN_PATTERN = re.compile(r"\\begin\{([^{}]+)\}")
_END_PATTERN = re.compile(r"\\end\{([^{}]+)\}")
_UNRESOLVED_HINT = "formula unresolved"


class PdfParserProtocol(Protocol):
    """Type protocol for parser inputs used by benchmark runner."""

    def parse(self, pdf_path: str, doc_id: str) -> ParsedDocument:
        """Parse PDF into normalized document structure."""


def _safe_div(numerator: int, denominator: int, *, zero_value: float = 0.0) -> float:
    if denominator <= 0:
        return zero_value
    return numerator / denominator


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _normalize_latex(latex: str) -> str:
    return " ".join(latex.split())


def _balanced_delimiters(text: str) -> bool:
    pairs = {"{": "}", "(": ")", "[": "]"}
    closings = set(pairs.values())
    stack: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in pairs:
            stack.append(pairs[ch])
            continue
        if ch in closings:
            if not stack or stack[-1] != ch:
                return False
            stack.pop()
    return not stack


def extract_block_latex(text: str) -> list[str]:
    """Extract block-level LaTeX expressions from markdown-like text."""
    formulas: list[str] = []
    for matched in _BLOCK_FORMULA_PATTERN.findall(text):
        cleaned = matched.strip()
        if cleaned:
            formulas.append(cleaned)
    return formulas


def is_latex_heuristically_valid(latex: str) -> bool:
    """Fast structural quality check for LaTeX without external renderer."""
    normalized = _normalize_latex(latex)
    if not normalized:
        return False

    lowered = normalized.lower()
    if _UNRESOLVED_HINT in lowered:
        return False

    if normalized.count(r"\left") != normalized.count(r"\right"):
        return False

    begins = _BEGIN_PATTERN.findall(normalized)
    ends = _END_PATTERN.findall(normalized)
    if len(begins) != len(ends):
        return False
    for expected, actual in zip(begins, ends, strict=True):
        if expected != actual:
            return False

    return _balanced_delimiters(normalized)


def summarize_parsed_formula_quality(parsed: ParsedDocument) -> dict[str, Any]:
    """Compute no-label formula quality metrics for a parsed document."""
    markers_total = _to_int(parsed.metadata.get("formula_markers_total"))
    replaced_by_engine = _to_int(parsed.metadata.get("formula_replaced_by_pix2text"))
    replaced_by_fallback = _to_int(parsed.metadata.get("formula_replaced_by_fallback"))
    unresolved = _to_int(parsed.metadata.get("formula_unresolved"))
    recovered_total = replaced_by_engine + replaced_by_fallback

    latex_blocks: list[str] = []
    for chunk in parsed.chunks:
        latex_blocks.extend(extract_block_latex(chunk.text))

    valid_blocks = sum(
        1 for latex in latex_blocks if is_latex_heuristically_valid(latex)
    )
    signature_payload = {
        "formula_stats": {
            "markers_total": markers_total,
            "replaced_by_engine": replaced_by_engine,
            "replaced_by_fallback": replaced_by_fallback,
            "unresolved": unresolved,
        },
        "latex_blocks": [_normalize_latex(item) for item in latex_blocks],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()

    return {
        "formula_markers_total": markers_total,
        "formula_recovered_total": recovered_total,
        "formula_replaced_by_pix2text": replaced_by_engine,
        "formula_replaced_by_fallback": replaced_by_fallback,
        "formula_unresolved": unresolved,
        "formula_recovered_rate": _safe_div(
            recovered_total, markers_total, zero_value=0.0
        ),
        "formula_unresolved_rate": _safe_div(unresolved, markers_total, zero_value=0.0),
        "latex_blocks_total": len(latex_blocks),
        "latex_blocks_heuristic_valid": valid_blocks,
        "latex_blocks_heuristic_valid_rate": _safe_div(
            valid_blocks, len(latex_blocks), zero_value=1.0
        ),
        "formula_signature": signature,
    }


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AppError):
        return {
            "type": "AppError",
            "code": exc.code,
            "message": exc.message,
            "details": exc.details or None,
        }
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }


def run_pdf_formula_benchmark(
    pdf_paths: list[Path],
    *,
    parser: PdfParserProtocol | None = None,
    passes: int = 2,
    max_unresolved_rate: float = 0.15,
    min_latex_valid_rate: float = 0.85,
    min_stability_rate: float = 1.0,
) -> dict[str, Any]:
    """Run no-label quality benchmark on a list of non-scanned PDF files."""
    runner = parser or DoclingPdfParser()
    pass_count = max(1, int(passes))
    documents: list[dict[str, Any]] = []

    for path in sorted(pdf_paths):
        path_str = str(path.resolve())
        if not path.exists():
            documents.append(
                {
                    "path": path_str,
                    "status": "error",
                    "error": {
                        "type": "FileNotFoundError",
                        "message": f"PDF not found: {path_str}",
                    },
                }
            )
            continue

        pass_results: list[dict[str, Any]] = []
        signatures: list[str] = []
        document_error: dict[str, Any] | None = None

        for pass_index in range(pass_count):
            doc_id = hashlib.sha1(
                f"{path_str}:{pass_index}".encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:16]
            try:
                parsed = runner.parse(path_str, doc_id)
                metrics = summarize_parsed_formula_quality(parsed)
                metrics["pass_index"] = pass_index
                pass_results.append(metrics)
                signatures.append(metrics["formula_signature"])
            except Exception as exc:  # noqa: BLE001
                document_error = _error_payload(exc)
                break

        if document_error is not None:
            documents.append(
                {
                    "path": path_str,
                    "status": "error",
                    "error": document_error,
                }
            )
            continue

        assert pass_results
        first_pass = pass_results[0]
        unique_signatures = len(set(signatures))
        documents.append(
            {
                "path": path_str,
                "status": "ok",
                "first_pass": first_pass,
                "stability": {
                    "passes": pass_count,
                    "exact_match": unique_signatures == 1,
                    "unique_signatures": unique_signatures,
                },
                "all_passes": pass_results,
            }
        )

    ok_docs = [item for item in documents if item.get("status") == "ok"]
    markers_total = sum(item["first_pass"]["formula_markers_total"] for item in ok_docs)
    recovered_total = sum(
        item["first_pass"]["formula_recovered_total"] for item in ok_docs
    )
    unresolved_total = sum(item["first_pass"]["formula_unresolved"] for item in ok_docs)
    latex_blocks_total = sum(
        item["first_pass"]["latex_blocks_total"] for item in ok_docs
    )
    latex_valid_total = sum(
        item["first_pass"]["latex_blocks_heuristic_valid"] for item in ok_docs
    )
    stable_docs = sum(1 for item in ok_docs if item["stability"]["exact_match"])

    summary = {
        "docs_total": len(documents),
        "docs_ok": len(ok_docs),
        "docs_failed": len(documents) - len(ok_docs),
        "formula_markers_total": markers_total,
        "formula_recovered_total": recovered_total,
        "formula_unresolved_total": unresolved_total,
        "formula_recovered_rate": _safe_div(
            recovered_total, markers_total, zero_value=0.0
        ),
        "formula_unresolved_rate": _safe_div(
            unresolved_total, markers_total, zero_value=0.0
        ),
        "latex_blocks_total": latex_blocks_total,
        "latex_blocks_heuristic_valid_total": latex_valid_total,
        "latex_blocks_heuristic_valid_rate": _safe_div(
            latex_valid_total, latex_blocks_total, zero_value=1.0
        ),
        "stability_exact_match_rate": _safe_div(
            stable_docs, len(ok_docs), zero_value=0.0
        ),
    }

    thresholds = {
        "max_unresolved_rate": max_unresolved_rate,
        "min_latex_valid_rate": min_latex_valid_rate,
        "min_stability_rate": min_stability_rate,
        "passed": (
            summary["formula_unresolved_rate"] <= max_unresolved_rate
            and summary["latex_blocks_heuristic_valid_rate"] >= min_latex_valid_rate
            and summary["stability_exact_match_rate"] >= min_stability_rate
            and summary["docs_failed"] == 0
        ),
    }

    return {
        "summary": summary,
        "thresholds": thresholds,
        "documents": documents,
    }


def _discover_pdf_paths(samples_dir: Path) -> list[Path]:
    if not samples_dir.exists():
        return []
    return sorted(path for path in samples_dir.rglob("*.pdf") if path.is_file())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-ebook-formula-benchmark",
        description="Run no-label formula quality benchmark on sample PDFs.",
    )
    parser.add_argument(
        "--samples-dir",
        default="tests/samples/pdf-papers",
        help="Directory containing non-scanned PDF files.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Repeat parse count for stability check.",
    )
    parser.add_argument(
        "--max-unresolved-rate",
        type=float,
        default=0.15,
        help="Fail threshold: unresolved marker rate upper bound.",
    )
    parser.add_argument(
        "--min-latex-valid-rate",
        type=float,
        default=0.85,
        help="Fail threshold: heuristic LaTeX validity lower bound.",
    )
    parser.add_argument(
        "--min-stability-rate",
        type=float,
        default=1.0,
        help="Fail threshold: exact signature stability lower bound.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write benchmark JSON to file path when set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    pdf_paths = _discover_pdf_paths(Path(args.samples_dir).resolve())
    if not pdf_paths:
        parser.error(f"No PDF files found under: {args.samples_dir}")

    result = run_pdf_formula_benchmark(
        pdf_paths,
        passes=args.passes,
        max_unresolved_rate=args.max_unresolved_rate,
        min_latex_valid_rate=args.min_latex_valid_rate,
        min_stability_rate=args.min_stability_rate,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)

    return 0 if result["thresholds"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
