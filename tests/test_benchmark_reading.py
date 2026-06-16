from __future__ import annotations

from pathlib import Path

from mcp_ebook_read.benchmark.reading import (
    run_reading_benchmark,
    run_reading_service_benchmark,
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


def test_run_reading_service_benchmark_exercises_mcp_reading_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "library"
    root.mkdir()

    class FakeService:
        def storage_list_sidecars(self, *, root: str, limit: int) -> dict:
            return {
                "root": root,
                "sidecars_count": 1,
                "documents_count": 1,
                "sidecars": [
                    {
                        "sidecar_path": str(tmp_path / ".mcp-ebook-read"),
                        "diagnostics_count": 0,
                        "documents": [
                            {
                                "doc_id": "pdf-book",
                                "path": str(tmp_path / "book.pdf"),
                                "type": "pdf",
                                "profile": "book",
                                "status": "ready",
                            }
                        ],
                    }
                ],
            }

        def library_explore(self, *, root: str, query: str, top_k: int = 12) -> dict:
            return {
                "retrieval": {"searched_documents_count": 1},
                "documents": [{"doc_id": "pdf-book"}],
                "selected_results": [
                    {
                        "document": {"doc_id": "pdf-book"},
                        "node": {"node_id": "node-1"},
                    }
                ],
                "hits": [{"source_id": "chunk-1"}],
                "diagnostics": [],
            }

        def get_outline(self, doc_id: str) -> dict:
            return {"title": "Book", "nodes": [{"id": "chapter-1", "title": "Intro"}]}

        def document_explore(self, doc_id: str, query: str, top_k: int = 8) -> dict:
            return {
                "document": {"doc_id": doc_id},
                "selected_nodes": [{"node_id": "node-1"}],
                "hits": [{"source_id": "chunk-1"}],
                "diagnostics": [],
            }

        def document_node(self, doc_id: str, node_id: str) -> dict:
            return {"node": {"node_id": node_id}, "neighbors": []}

        def read_outline_node(
            self,
            *,
            doc_id: str,
            node_id: str,
            out_format: str,
            max_chunks: int = 120,
        ) -> dict:
            return {"content": "# Intro", "chunks_count": 1, "truncated": False}

        def pdf_book_list_formulas(
            self,
            *,
            doc_id: str,
            node_id: str | None,
            limit: int,
            status: str | None,
        ) -> dict:
            return {
                "formulas": [{"formula_id": "formula-1", "latex": "x=y"}],
                "formulas_count": 1,
            }

        def pdf_book_read_formula(self, *, doc_id: str, formula_id: str) -> dict:
            return {
                "formula": {"formula_id": formula_id, "latex": "x=y"},
                "evidence": {"image_path": "formula.png"},
            }

        def pdf_paper_list_formulas(self, **kwargs) -> dict:  # noqa: ANN003
            raise AssertionError("book profile should not call paper formula list")

        def pdf_paper_read_formula(self, **kwargs) -> dict:  # noqa: ANN003
            raise AssertionError("book profile should not call paper formula read")

        def pdf_list_images(
            self, *, doc_id: str, node_id: str | None, limit: int
        ) -> dict:
            return {"images": [{"image_id": "image-1"}], "images_count": 1}

        def pdf_read_image(self, *, doc_id: str, image_id: str) -> dict:
            return {"image": {"image_id": image_id}, "context": None}

        def pdf_list_tables(
            self, *, doc_id: str, node_id: str | None, limit: int
        ) -> dict:
            return {"tables": [{"table_id": "table-1"}], "tables_count": 1}

        def pdf_read_table(self, *, doc_id: str, table_id: str) -> dict:
            return {"table": {"table_id": table_id}, "context": None}

        def pdf_list_figures(
            self, *, doc_id: str, node_id: str | None, limit: int
        ) -> dict:
            return {"figures": [{"figure_id": "figure-1"}], "figures_count": 1}

        def pdf_read_figure(self, *, doc_id: str, figure_id: str) -> dict:
            return {"figure": {"figure_id": figure_id}, "context": None}

    result = run_reading_service_benchmark(root, service=FakeService())

    assert result["thresholds"]["passed"] is True
    assert result["summary"]["documents_evaluated"] == 1
    assert result["summary"]["tasks_failed"] == 0
    assert result["summary"]["task_pass_rate"] == 1.0
    assert result["summary"]["formula_lists_nonempty"] == 1
    assert result["summary"]["image_lists_nonempty"] == 1
    assert result["summary"]["table_lists_nonempty"] == 1
    assert result["summary"]["figure_lists_nonempty"] == 1
    assert result["summary"]["evidence_reads_ok"] == 4
