from __future__ import annotations

from pathlib import Path

from mcp_ebook_read.benchmark.pdf_formula import (
    extract_block_latex,
    is_latex_heuristically_valid,
    run_pdf_formula_benchmark,
    summarize_parsed_formula_quality,
)
from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import ChunkRecord, Locator, ParsedDocument


def _chunk(doc_id: str, chunk_id: str, text: str) -> ChunkRecord:
    locator = Locator(
        doc_id=doc_id,
        chunk_id=chunk_id,
        section_path=["Section"],
        page_range=[1, 1],
        method="docling",
        confidence=None,
    )
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        order_index=0,
        section_path=["Section"],
        text=text,
        search_text=" ".join(text.split()),
        locator=locator,
        method="docling",
        confidence=None,
    )


def _parsed(
    doc_id: str,
    *,
    chunk_text: str,
    markers_total: int,
    replaced: int,
    fallback: int,
    unresolved: int,
) -> ParsedDocument:
    return ParsedDocument(
        title="Doc",
        parser_chain=["docling"],
        metadata={
            "formula_markers_total": markers_total,
            "formula_replaced_by_pix2text": replaced,
            "formula_replaced_by_fallback": fallback,
            "formula_unresolved": unresolved,
        },
        outline=[],
        chunks=[_chunk(doc_id, f"{doc_id}-chunk", chunk_text)],
        reading_markdown=chunk_text,
        raw_artifacts={},
        overall_confidence=None,
    )


def test_extract_block_latex() -> None:
    text = "intro\n$$\\frac{a}{b}$$\ntext\n$$\\sum_{i=1}^{n} x_i$$"
    assert extract_block_latex(text) == [r"\frac{a}{b}", r"\sum_{i=1}^{n} x_i"]


def test_is_latex_heuristically_valid() -> None:
    assert is_latex_heuristically_valid(r"\frac{a}{b}")
    assert not is_latex_heuristically_valid(r"\left( a + b")
    assert not is_latex_heuristically_valid(
        "[Formula unresolved. Use render_pdf_page for verification.]"
    )


def test_summarize_parsed_formula_quality() -> None:
    parsed = _parsed(
        "doc1",
        chunk_text="$$\\frac{a}{b}$$\n$$\\left( x + y $$",
        markers_total=3,
        replaced=2,
        fallback=0,
        unresolved=1,
    )
    metrics = summarize_parsed_formula_quality(parsed)

    assert metrics["formula_markers_total"] == 3
    assert metrics["formula_recovered_total"] == 2
    assert metrics["formula_unresolved"] == 1
    assert metrics["formula_unresolved_rate"] == 1 / 3
    assert metrics["latex_blocks_total"] == 2
    assert metrics["latex_blocks_heuristic_valid"] == 1
    assert len(metrics["formula_signature"]) == 64


def test_run_pdf_formula_benchmark_with_stability_and_error(tmp_path: Path) -> None:
    good_pdf = tmp_path / "good.pdf"
    unstable_pdf = tmp_path / "unstable.pdf"
    broken_pdf = tmp_path / "broken.pdf"
    good_pdf.write_bytes(b"%PDF-1.7")
    unstable_pdf.write_bytes(b"%PDF-1.7")
    broken_pdf.write_bytes(b"%PDF-1.7")

    class FakeParser:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def parse(self, pdf_path: str, doc_id: str) -> ParsedDocument:
            call_index = self.calls.get(pdf_path, 0)
            self.calls[pdf_path] = call_index + 1

            if pdf_path.endswith("broken.pdf"):
                raise AppError(
                    ErrorCode.INGEST_PDF_DOCLING_FAILED,
                    "broken parser",
                )

            if pdf_path.endswith("unstable.pdf") and call_index == 1:
                return _parsed(
                    doc_id,
                    chunk_text="$$x^2$$",
                    markers_total=1,
                    replaced=1,
                    fallback=0,
                    unresolved=0,
                )

            return _parsed(
                doc_id,
                chunk_text="$$\\frac{a}{b}$$",
                markers_total=1,
                replaced=1,
                fallback=0,
                unresolved=0,
            )

    result = run_pdf_formula_benchmark(
        [good_pdf, unstable_pdf, broken_pdf],
        parser=FakeParser(),
        passes=2,
        max_unresolved_rate=0.1,
        min_latex_valid_rate=0.9,
        min_stability_rate=1.0,
    )

    assert result["summary"]["docs_total"] == 3
    assert result["summary"]["docs_ok"] == 2
    assert result["summary"]["docs_failed"] == 1
    assert result["summary"]["formula_unresolved_rate"] == 0.0
    assert result["summary"]["stability_exact_match_rate"] == 0.5
    assert result["thresholds"]["passed"] is False
