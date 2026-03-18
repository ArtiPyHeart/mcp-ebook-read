from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    Locator,
    OutlineNode,
    PdfFigureRecord,
    PdfTableRecord,
    TableSegmentRecord,
)
from mcp_ebook_read.service import AppService


class _DummyVectorIndex:
    def rebuild_document(
        self, doc_id: str, title: str, chunks: list[ChunkRecord]
    ) -> None:  # noqa: ARG002
        return None

    def search(self, query: str, top_k: int = 20, doc_ids: list[str] | None = None):  # noqa: ARG002
        return []


class _DummyParser:
    def parse(self, path: str, doc_id: str):  # noqa: ARG002
        raise NotImplementedError

    def close(self) -> None:
        return None


class _DummyGrobid:
    def parse_fulltext(self, path: str):  # noqa: ARG002
        raise NotImplementedError


class _ObservabilityVisualExtractor:
    def __init__(
        self,
        *,
        tables: list[PdfTableRecord],
        figures: list[PdfFigureRecord],
        diagnostics: dict,
    ) -> None:
        self.tables = tables
        self.figures = figures
        self.diagnostics = diagnostics

    def extract(
        self,
        *,
        pdf_path: str,  # noqa: ARG002
        doc_id: str,  # noqa: ARG002
        chunks: list[ChunkRecord],  # noqa: ARG002
        tables_dir: Path,
        figures_dir: Path,
    ):
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        generated_tables: list[PdfTableRecord] = []
        for table in self.tables:
            image_path = tables_dir / f"{table.table_id}.png"
            image_path.write_bytes(b"\x89PNG")
            segments = []
            for idx, segment in enumerate(table.segments):
                segment_path = tables_dir / f"{table.table_id}_segment_{idx}.png"
                segment_path.write_bytes(b"\x89PNG")
                segments.append(
                    segment.model_copy(update={"file_path": str(segment_path)})
                )
            generated_tables.append(
                table.model_copy(
                    update={
                        "file_path": str(image_path),
                        "segments": segments,
                    }
                )
            )

        generated_figures: list[PdfFigureRecord] = []
        for figure in self.figures:
            image_path = figures_dir / f"{figure.figure_id}.png"
            image_path.write_bytes(b"\x89PNG")
            generated_figures.append(
                figure.model_copy(update={"file_path": str(image_path)})
            )

        return SimpleNamespace(
            tables=generated_tables,
            figures=generated_figures,
            diagnostics=self.diagnostics,
        )


def _build_service(
    tmp_path: Path, visual_extractor: _ObservabilityVisualExtractor
) -> AppService:
    return AppService(
        sidecar_dir_name=".mcp-ebook-read",
        vector_index=_DummyVectorIndex(),
        pdf_parser=_DummyParser(),
        pdf_visual_extractor=visual_extractor,
        grobid_client=_DummyGrobid(),
        epub_parser=_DummyParser(),
    )


def _register_ready_pdf(service: AppService, doc_id: str, pdf_path: Path) -> None:
    catalog = service._catalog_for_document_path(pdf_path)
    catalog.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str(pdf_path.resolve()),
            type=DocumentType.PDF,
            sha256="a" * 64,
            mtime=1.0,
        )
    )
    catalog.save_document_parse_output(
        doc_id=doc_id,
        title="Observed PDF",
        parser_chain=["docling"],
        metadata={"source": "docling"},
        outline=[
            OutlineNode(
                id="toc-1", title="Chapter 1", level=1, page_start=1, page_end=3
            )
        ],
        overall_confidence=0.9,
        status=DocumentStatus.READY,
    )
    catalog.replace_chunks(
        doc_id,
        [
            ChunkRecord(
                chunk_id="c0",
                doc_id=doc_id,
                order_index=0,
                section_path=["Chapter 1"],
                text="Observed context",
                search_text="Observed context",
                locator=Locator(
                    doc_id=doc_id,
                    chunk_id="c0",
                    section_path=["Chapter 1"],
                    page_range=[1, 2],
                    method="docling",
                ),
                method="docling",
            )
        ],
    )


def test_pdf_table_tools_surface_diagnostics_and_persist_summary(
    tmp_path: Path,
) -> None:
    doc_id = "doc-visual-diag-table"
    pdf_path = tmp_path / "observed.pdf"
    pdf_path.write_bytes(b"pdf")

    table = PdfTableRecord(
        table_id="table-1",
        doc_id=doc_id,
        order_index=0,
        section_path=["Chapter 1"],
        page_range=[1, 2],
        bbox=None,
        caption=None,
        headers=["Name", "Value"],
        rows=[["alpha", "1"]],
        markdown="| Name | Value |\n| --- | --- |\n| alpha | 1 |",
        html="<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody><tr><td>alpha</td><td>1</td></tr></tbody></table>",
        file_path="placeholder.png",
        width=300,
        height=120,
        merged=True,
        merge_confidence=0.91,
        segments=[
            TableSegmentRecord(
                page=1,
                bbox=[10.0, 10.0, 100.0, 100.0],
                caption=None,
                file_path="placeholder-segment.png",
                width=300,
                height=120,
            )
        ],
    )
    diagnostics = {
        "extractor": "docling-visuals",
        "summary": {
            "tables_detected_raw": 1,
            "tables_returned": 1,
            "figures_returned": 0,
            "merged_tables_count": 1,
            "issues_count": 1,
            "warning_count": 1,
            "error_count": 0,
            "info_count": 0,
        },
        "issues": [
            {
                "code": "PDF_TABLE_CAPTION_MISSING",
                "severity": "warning",
                "message": "A table was extracted without caption text.",
                "hint": "Escalate the sample if this ambiguity blocks reading.",
                "details": {"table_id": "table-1", "page": 1},
            }
        ],
        "merge_decisions": [
            {
                "decision": "merged",
                "score": 0.91,
                "reasons": ["matching_headers", "page_edge_alignment_ok"],
                "details": {"pages": [1, 2]},
            }
        ],
        "tables": {
            "table-1": {
                "issues": [
                    {
                        "code": "PDF_TABLE_CAPTION_MISSING",
                        "severity": "warning",
                        "message": "A table was extracted without caption text.",
                        "hint": "Escalate the sample if this ambiguity blocks reading.",
                        "details": {"table_id": "table-1", "page": 1},
                    }
                ],
                "merge": {
                    "decision": "merged",
                    "score": 0.91,
                    "reason": "adjacent table segments passed all merge checks",
                },
            }
        },
        "figures": {},
    }
    service = _build_service(
        tmp_path,
        _ObservabilityVisualExtractor(
            tables=[table], figures=[], diagnostics=diagnostics
        ),
    )
    _register_ready_pdf(service, doc_id, pdf_path)

    listed = service.pdf_list_tables(doc_id=doc_id, node_id=None, limit=20)
    assert listed["diagnostics"]["summary"]["issues_count"] == 1
    assert listed["tables"][0]["issues"][0]["code"] == "PDF_TABLE_CAPTION_MISSING"
    assert listed["tables"][0]["merge_diagnostics"]["decision"] == "merged"

    read = service.pdf_read_table(doc_id=doc_id, table_id="table-1")
    assert (
        read["diagnostics"]["document"]["issues"][0]["code"]
        == "PDF_TABLE_CAPTION_MISSING"
    )
    assert read["diagnostics"]["table"]["merge"]["decision"] == "merged"

    catalog = service._catalog_for_document_path(pdf_path)
    loaded = catalog.get_document_by_id(doc_id)
    assert loaded is not None
    assert loaded.metadata["pdf_visuals_diagnostics"]["summary"]["issues_count"] == 1


def test_pdf_figure_tools_surface_item_issues(tmp_path: Path) -> None:
    doc_id = "doc-visual-diag-figure"
    pdf_path = tmp_path / "observed_figure.pdf"
    pdf_path.write_bytes(b"pdf")

    figure = PdfFigureRecord(
        figure_id="figure-1",
        doc_id=doc_id,
        order_index=0,
        section_path=["Chapter 1"],
        page=1,
        bbox=None,
        caption=None,
        kind="chart",
        file_path="placeholder.png",
        width=200,
        height=100,
    )
    diagnostics = {
        "extractor": "docling-visuals",
        "summary": {
            "tables_detected_raw": 0,
            "tables_returned": 0,
            "figures_returned": 1,
            "merged_tables_count": 0,
            "issues_count": 1,
            "warning_count": 1,
            "error_count": 0,
            "info_count": 0,
        },
        "issues": [
            {
                "code": "PDF_FIGURE_CAPTION_MISSING",
                "severity": "warning",
                "message": "A figure was extracted without caption text.",
                "details": {"figure_id": "figure-1", "page": 1},
            }
        ],
        "merge_decisions": [],
        "tables": {},
        "figures": {
            "figure-1": {
                "issues": [
                    {
                        "code": "PDF_FIGURE_CAPTION_MISSING",
                        "severity": "warning",
                        "message": "A figure was extracted without caption text.",
                        "details": {"figure_id": "figure-1", "page": 1},
                    }
                ]
            }
        },
    }
    service = _build_service(
        tmp_path,
        _ObservabilityVisualExtractor(
            tables=[], figures=[figure], diagnostics=diagnostics
        ),
    )
    _register_ready_pdf(service, doc_id, pdf_path)

    listed = service.pdf_list_figures(doc_id=doc_id, node_id=None, limit=20)
    assert listed["diagnostics"]["summary"]["issues_count"] == 1
    assert listed["figures"][0]["issues"][0]["code"] == "PDF_FIGURE_CAPTION_MISSING"

    read = service.pdf_read_figure(doc_id=doc_id, figure_id="figure-1")
    assert (
        read["diagnostics"]["figure"]["issues"][0]["code"]
        == "PDF_FIGURE_CAPTION_MISSING"
    )
