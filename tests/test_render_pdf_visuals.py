from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from PIL import Image

from mcp_ebook_read.render.pdf_visuals import DoclingPdfVisualExtractor
from mcp_ebook_read.schema.models import ChunkRecord, Locator


def _chunk(doc_id: str, chunk_id: str, page_start: int, page_end: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        order_index=0,
        section_path=["Chapter 1"],
        text="context",
        search_text="context",
        locator=Locator(
            doc_id=doc_id,
            chunk_id=chunk_id,
            section_path=["Chapter 1"],
            page_range=[page_start, page_end],
            method="docling",
        ),
        method="docling",
    )


class _FakeValues:
    def __init__(self, rows: list[list[str]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[str]]:
        return self._rows


class _FakeFrame:
    def __init__(self, headers: list[str], rows: list[list[str]]) -> None:
        self.columns = headers
        self._rows = rows

    def fillna(self, _value: str) -> "_FakeFrame":
        return self

    @property
    def values(self) -> _FakeValues:
        return _FakeValues(self._rows)


class _FakeBBox:
    def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
        self.l = left
        self.t = top
        self.r = right
        self.b = bottom

    def to_top_left_origin(self, *, page_height: float) -> "_FakeBBox":  # noqa: ARG002
        return self


class _FakeTableItem:
    def __init__(
        self,
        *,
        page: int,
        bbox: tuple[float, float, float, float] | None,
        headers: list[str],
        rows: list[list[str]],
        image: Image.Image | None,
        caption_refs: list[str] | None = None,
        html_headers: list[str] | None = None,
        html_rows: list[list[str]] | None = None,
        dataframe_error: Exception | None = None,
    ) -> None:
        self.prov = [
            SimpleNamespace(
                page_no=page,
                bbox=_FakeBBox(*bbox) if bbox is not None else None,
            )
        ]
        self._headers = headers
        self._rows = rows
        self._image = image
        self._html_headers = html_headers or headers
        self._html_rows = html_rows or rows
        self._dataframe_error = dataframe_error
        self.captions = [SimpleNamespace(cref=ref) for ref in (caption_refs or [])]

    def get_image(self, _document) -> Image.Image | None:  # noqa: ANN001
        return self._image.copy() if self._image is not None else None

    def export_to_dataframe(self, doc=None):  # noqa: ANN001, ARG002
        if self._dataframe_error is not None:
            raise self._dataframe_error
        return _FakeFrame(self._headers, self._rows)

    def export_to_html(self, doc=None, add_caption: bool = False) -> str:  # noqa: ANN001, ARG002
        caption_html = "<caption>caption</caption>" if add_caption else ""
        header_html = "".join(f"<th>{header}</th>" for header in self._html_headers)
        row_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in self._html_rows
        )
        return (
            f"<table>{caption_html}<thead><tr>{header_html}</tr></thead>"
            f"<tbody>{row_html}</tbody></table>"
        )


class _FakePictureItem:
    def __init__(
        self,
        *,
        page: int,
        bbox: tuple[float, float, float, float] | None,
        image: Image.Image | None,
        kind: str,
        caption_refs: list[str] | None = None,
    ) -> None:
        self.prov = [
            SimpleNamespace(
                page_no=page,
                bbox=_FakeBBox(*bbox) if bbox is not None else None,
            )
        ]
        self._image = image
        self.label = SimpleNamespace(value=kind)
        self.captions = [SimpleNamespace(cref=ref) for ref in (caption_refs or [])]

    def get_image(self, _document) -> Image.Image | None:  # noqa: ANN001
        return self._image.copy() if self._image is not None else None


class _FakeTextItem:
    def __init__(
        self,
        ref: str | None,
        text: str,
        *,
        page: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.self_ref = ref
        self.text = text
        self.prov = []
        if page is not None:
            self.prov = [
                SimpleNamespace(
                    page_no=page,
                    bbox=_FakeBBox(*bbox) if bbox is not None else None,
                )
            ]


class _FakeDocument:
    def __init__(
        self,
        *,
        tables: list[_FakeTableItem],
        pictures: list[_FakePictureItem],
        caption_items: list[_FakeTextItem],
    ) -> None:
        self.tables = tables
        self.pictures = pictures
        self.pages = {
            1: SimpleNamespace(size=SimpleNamespace(width=600.0, height=800.0)),
            2: SimpleNamespace(size=SimpleNamespace(width=600.0, height=800.0)),
        }
        self._caption_items = caption_items

    def iterate_items(self):
        for item in self._caption_items:
            yield item, 0


class _FakeConverter:
    def __init__(self, document: _FakeDocument) -> None:
        self._document = document

    def convert(self, _path: str):
        return SimpleNamespace(document=self._document)


def test_docling_pdf_visual_extractor_merges_adjacent_tables_and_extracts_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "visuals.pdf"
    pdf_path.write_bytes(b"pdf")
    extractor = DoclingPdfVisualExtractor()

    document = _FakeDocument(
        tables=[
            _FakeTableItem(
                page=1,
                bbox=(10.0, 640.0, 540.0, 790.0),
                headers=["Name", "Value"],
                rows=[["alpha", "1"]],
                image=Image.new("RGB", (300, 120), "white"),
                caption_refs=["#/captions/0"],
            ),
            _FakeTableItem(
                page=2,
                bbox=(12.0, 18.0, 538.0, 250.0),
                headers=["Name", "Value"],
                rows=[["beta", "2"]],
                image=Image.new("RGB", (300, 120), "white"),
            ),
        ],
        pictures=[
            _FakePictureItem(
                page=1,
                bbox=(100.0, 100.0, 250.0, 220.0),
                image=Image.new("RGB", (160, 120), "blue"),
                kind="chart",
                caption_refs=["#/captions/1"],
            )
        ],
        caption_items=[
            _FakeTextItem("#/captions/0", "Table 1: Metrics"),
            _FakeTextItem("#/captions/1", "Figure 1: Trend"),
        ],
    )
    monkeypatch.setattr(
        extractor,
        "_build_docling_converter",
        lambda: _FakeConverter(document),
    )

    result = extractor.extract(
        pdf_path=str(pdf_path),
        doc_id="doc1",
        chunks=[_chunk("doc1", "chunk-1", 1, 2)],
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )

    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.merged is True
    assert table.page_range == [1, 2]
    assert table.caption == "Table 1: Metrics"
    assert table.rows == [["alpha", "1"], ["beta", "2"]]
    assert len(table.segments) == 2
    assert Path(table.file_path).exists()
    assert result.diagnostics["summary"]["merged_tables_count"] == 1
    assert result.diagnostics["merge_decisions"][0]["decision"] == "merged"
    assert result.diagnostics["tables"][table.table_id]["merge"]["decision"] == "merged"

    assert len(result.figures) == 1
    figure = result.figures[0]
    assert figure.kind == "chart"
    assert figure.caption == "Figure 1: Trend"
    assert figure.section_path == ["Chapter 1"]
    assert Path(figure.file_path).exists()
    assert result.diagnostics["figures"][figure.figure_id]["issues"] == []


def test_docling_pdf_visual_extractor_keeps_tables_separate_when_headers_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "visuals_headers.pdf"
    pdf_path.write_bytes(b"pdf")
    extractor = DoclingPdfVisualExtractor()

    document = _FakeDocument(
        tables=[
            _FakeTableItem(
                page=1,
                bbox=(10.0, 640.0, 540.0, 790.0),
                headers=["Name", "Value"],
                rows=[["alpha", "1"]],
                image=Image.new("RGB", (300, 120), "white"),
            ),
            _FakeTableItem(
                page=2,
                bbox=(12.0, 18.0, 538.0, 250.0),
                headers=["Name", "Score"],
                rows=[["beta", "2"]],
                image=Image.new("RGB", (300, 120), "white"),
            ),
        ],
        pictures=[],
        caption_items=[],
    )
    monkeypatch.setattr(
        extractor,
        "_build_docling_converter",
        lambda: _FakeConverter(document),
    )

    result = extractor.extract(
        pdf_path=str(pdf_path),
        doc_id="doc1",
        chunks=[_chunk("doc1", "chunk-1", 1, 2)],
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )

    assert len(result.tables) == 2
    assert all(table.merged is False for table in result.tables)
    assert result.diagnostics["merge_decisions"][0]["decision"] == "rejected"
    assert "headers_mismatch" in result.diagnostics["merge_decisions"][0]["reasons"]


def test_docling_pdf_visual_extractor_recovers_captions_from_page_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "visuals_page_text.pdf"
    pdf_path.write_bytes(b"pdf")
    extractor = DoclingPdfVisualExtractor()

    document = _FakeDocument(
        tables=[
            _FakeTableItem(
                page=1,
                bbox=(10.0, 480.0, 540.0, 640.0),
                headers=["Name", "Value"],
                rows=[["alpha", "1"]],
                image=Image.new("RGB", (300, 120), "white"),
            )
        ],
        pictures=[
            _FakePictureItem(
                page=2,
                bbox=(100.0, 120.0, 300.0, 260.0),
                image=Image.new("RGB", (160, 120), "blue"),
                kind="chart",
            )
        ],
        caption_items=[
            _FakeTextItem(
                None,
                "Table 3: Heuristic metrics",
                page=1,
                bbox=(10.0, 648.0, 540.0, 690.0),
            ),
            _FakeTextItem(
                None,
                "Figure 2: Trend overview",
                page=2,
                bbox=(100.0, 268.0, 300.0, 305.0),
            ),
        ],
    )
    monkeypatch.setattr(
        extractor,
        "_build_docling_converter",
        lambda: _FakeConverter(document),
    )

    result = extractor.extract(
        pdf_path=str(pdf_path),
        doc_id="doc1",
        chunks=[_chunk("doc1", "chunk-1", 1, 2)],
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )

    table = result.tables[0]
    table_observation = result.diagnostics["tables"][table.table_id]
    assert table.caption == "Table 3: Heuristic metrics"
    assert table_observation["caption"]["source"] == "page_text_pattern"
    assert table_observation["caption"]["needs_review"] is True
    assert table_observation["issues"][0]["code"] == "PDF_TABLE_CAPTION_HEURISTIC_MATCH"

    figure = result.figures[0]
    figure_observation = result.diagnostics["figures"][figure.figure_id]
    assert figure.caption == "Figure 2: Trend overview"
    assert figure_observation["caption"]["source"] == "page_text_pattern"
    assert figure_observation["caption"]["needs_review"] is True
    assert (
        figure_observation["issues"][0]["code"] == "PDF_FIGURE_CAPTION_HEURISTIC_MATCH"
    )
    assert result.diagnostics["summary"]["info_count"] == 2


def test_docling_pdf_visual_extractor_keeps_ambiguous_caption_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "visuals_ambiguous_caption.pdf"
    pdf_path.write_bytes(b"pdf")
    extractor = DoclingPdfVisualExtractor()

    document = _FakeDocument(
        tables=[
            _FakeTableItem(
                page=1,
                bbox=None,
                headers=["Name", "Value"],
                rows=[["alpha", "1"]],
                image=Image.new("RGB", (300, 120), "white"),
            )
        ],
        pictures=[],
        caption_items=[
            _FakeTextItem(
                None,
                "Baseline results summary",
                page=1,
                bbox=(20.0, 80.0, 260.0, 120.0),
            )
        ],
    )
    monkeypatch.setattr(
        extractor,
        "_build_docling_converter",
        lambda: _FakeConverter(document),
    )

    result = extractor.extract(
        pdf_path=str(pdf_path),
        doc_id="doc1",
        chunks=[_chunk("doc1", "chunk-1", 1, 1)],
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )

    table = result.tables[0]
    table_observation = result.diagnostics["tables"][table.table_id]
    issue_codes = {issue["code"] for issue in table_observation["issues"]}

    assert table.caption is None
    assert "PDF_TABLE_CAPTION_AMBIGUOUS" in issue_codes
    assert "PDF_TABLE_BBOX_MISSING" in issue_codes
    assert table_observation["caption"]["text"] is None
    assert table_observation["caption"]["needs_review"] is True
    assert (
        table_observation["caption"]["candidates"][0]["text"]
        == "Baseline results summary"
    )


def test_docling_pdf_visual_extractor_falls_back_to_html_when_dataframe_export_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "visuals_html_fallback.pdf"
    pdf_path.write_bytes(b"pdf")
    extractor = DoclingPdfVisualExtractor()

    document = _FakeDocument(
        tables=[
            _FakeTableItem(
                page=1,
                bbox=(10.0, 640.0, 540.0, 790.0),
                headers=["ignored"],
                rows=[["ignored"]],
                html_headers=["Metric", "Value"],
                html_rows=[["fallback", "7"]],
                image=Image.new("RGB", (300, 120), "white"),
                caption_refs=["#/captions/0"],
                dataframe_error=RuntimeError("dataframe export failed"),
            )
        ],
        pictures=[],
        caption_items=[_FakeTextItem("#/captions/0", "Table 7: HTML fallback")],
    )
    monkeypatch.setattr(
        extractor,
        "_build_docling_converter",
        lambda: _FakeConverter(document),
    )

    result = extractor.extract(
        pdf_path=str(pdf_path),
        doc_id="doc1",
        chunks=[_chunk("doc1", "chunk-1", 1, 1)],
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )

    table = result.tables[0]
    issue_codes = {
        issue["code"]
        for issue in result.diagnostics["tables"][table.table_id]["issues"]
    }

    assert table.caption == "Table 7: HTML fallback"
    assert table.headers == ["Metric", "Value"]
    assert table.rows == [["fallback", "7"]]
    assert "PDF_TABLE_DATAFRAME_FALLBACK_HTML" in issue_codes
