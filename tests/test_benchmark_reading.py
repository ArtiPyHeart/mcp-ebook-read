from __future__ import annotations

from pathlib import Path

from mcp_ebook_read.benchmark.reading import (
    run_reading_benchmark,
    summarize_parsed_reading_quality,
)
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    ExtractedImage,
    FormulaRecord,
    Locator,
    OutlineNode,
    ParsedDocument,
)


def _chunk(doc_id: str, chunk_id: str, text: str, order_index: int = 0) -> ChunkRecord:
    locator = Locator(
        doc_id=doc_id,
        chunk_id=chunk_id,
        section_path=["Chapter"],
        page_range=[1, 1],
        method="test",
    )
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        order_index=order_index,
        section_path=["Chapter"],
        text=text,
        search_text=text,
        locator=locator,
        method="test",
    )


def _parsed(doc_id: str, *, variant: str = "stable") -> ParsedDocument:
    chunks = [
        _chunk(doc_id, f"{doc_id}-c0", f"alpha beta gamma {variant}", 0),
        _chunk(doc_id, f"{doc_id}-c1", "delta epsilon", 1),
    ]
    if variant == "changed":
        chunks.append(_chunk(doc_id, f"{doc_id}-c2", "zeta eta", 2))
    return ParsedDocument(
        title="Doc",
        parser_chain=["test"],
        metadata={"pdf_tables_count": 1},
        outline=[OutlineNode(id="n1", title="Chapter", level=1)],
        chunks=chunks,
        formulas=[
            FormulaRecord(
                formula_id=f"{doc_id}-f0",
                doc_id=doc_id,
                section_path=["Chapter"],
                latex="x=y",
                source="test",
                status="resolved",
            )
        ],
        images=[
            ExtractedImage(
                image_id=f"{doc_id}-i0",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter"],
                media_type="image/png",
                data=b"png",
            )
        ],
        reading_markdown="text",
    )


def test_summarize_parsed_reading_quality() -> None:
    parsed = _parsed("doc1")
    metrics = summarize_parsed_reading_quality(parsed)

    assert metrics["outline_nodes_count"] == 1
    assert metrics["chunks_count"] == 2
    assert metrics["formulas_count"] == 1
    assert metrics["images_count"] == 1
    assert metrics["tables_count"] == 1
    assert metrics["search_replay"]["queries_count"] == 2
    assert len(metrics["reading_signature"]) == 64


def test_run_reading_benchmark_reports_stability_and_errors(tmp_path: Path) -> None:
    stable = tmp_path / "stable.epub"
    unstable = tmp_path / "unstable.pdf"
    missing = tmp_path / "missing.pdf"
    stable.write_bytes(b"epub")
    unstable.write_bytes(b"pdf")

    class FakeParser:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def parse(self, path: str, doc_id: str) -> ParsedDocument:
            call_index = self.calls.get(path, 0)
            self.calls[path] = call_index + 1
            variant = (
                "changed" if path.endswith("unstable.pdf") and call_index else "stable"
            )
            return _parsed(doc_id, variant=variant)

    result = run_reading_benchmark(
        [stable, unstable, missing],
        parser=FakeParser(),
        passes=2,
        min_stability_rate=1.0,
    )

    assert result["summary"]["docs_total"] == 3
    assert result["summary"]["docs_ok"] == 2
    assert result["summary"]["docs_failed"] == 1
    assert result["summary"]["stability_exact_match_rate"] == 0.5
    assert result["summary"]["chunks_total"] == 4
    assert result["thresholds"]["passed"] is False
