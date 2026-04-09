from __future__ import annotations

import builtins
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.parsers.pdf_docling import (
    _FORMULA_MARKER,
    DoclingPdfParser,
)
from mcp_ebook_read.schema.models import (
    PdfParserBenchmarkRow,
    PdfParserPerformanceConfig,
)


class FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, _format: str = "text") -> str:
        return self._text


class FakePixmap:
    def tobytes(self, _format: str = "png") -> bytes:
        image = Image.new("RGB", (8, 8), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class FakeRenderedPage:
    def get_pixmap(self, matrix=None, alpha=False):  # noqa: ANN001, ARG002
        return FakePixmap()


class FakePdfDoc:
    def __init__(
        self,
        title: str = "PDF Title",
        page_count: int = 3,
        toc: list[list[object]] | None = None,
        page_texts: list[str] | None = None,
    ) -> None:
        self.metadata = {"title": title}
        self.page_count = page_count
        self._toc = toc or [[1, "Intro", 1], [1, "Method", 2]]
        self._pages = [
            FakePdfPage(text)
            for text in (page_texts or [f"Page {idx}" for idx in range(page_count)])
        ]

    def get_toc(self) -> list[list[object]]:
        return self._toc

    def __iter__(self):
        return iter(self._pages)

    def load_page(self, _idx: int) -> FakeRenderedPage:
        return FakeRenderedPage()

    def __enter__(self) -> "FakePdfDoc":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def close(self) -> None:
        return None

    def insert_pdf(
        self, _source: "FakePdfDoc", *, from_page: int, to_page: int
    ) -> None:
        self.page_count = max(0, to_page - from_page + 1)

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"%PDF-1.4")


class CountingLoadPagePdfDoc(FakePdfDoc):
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.load_page_calls = 0

    def load_page(self, _idx: int) -> FakeRenderedPage:
        self.load_page_calls += 1
        return FakeRenderedPage()


class FakeDocument:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class FakeConvertResult:
    def __init__(self, markdown: str) -> None:
        self.document = FakeDocument(markdown)


def install_fake_docling(monkeypatch: pytest.MonkeyPatch, converter_cls: type) -> None:
    docling_pkg = types.ModuleType("docling")
    docling_pkg.__path__ = []

    datamodel_pkg = types.ModuleType("docling.datamodel")
    datamodel_pkg.__path__ = []
    base_models_module = types.ModuleType("docling.datamodel.base_models")
    accelerator_options_module = types.ModuleType(
        "docling.datamodel.accelerator_options"
    )
    pipeline_options_module = types.ModuleType("docling.datamodel.pipeline_options")

    class InputFormat:
        PDF = "pdf"

    class AcceleratorOptions:
        def __init__(self, *, num_threads: int = 4, device: str = "auto") -> None:
            self.num_threads = num_threads
            self.device = device

    class PdfPipelineOptions:
        def __init__(self) -> None:
            self.do_formula_enrichment = False
            self.accelerator_options = None
            self.ocr_batch_size = 4
            self.layout_batch_size = 4
            self.table_batch_size = 4

    base_models_module.InputFormat = InputFormat
    accelerator_options_module.AcceleratorOptions = AcceleratorOptions
    pipeline_options_module.PdfPipelineOptions = PdfPipelineOptions

    converter_module = types.ModuleType("docling.document_converter")

    class ConverterAdapter(converter_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.init_args = args
            self.init_kwargs = kwargs
            super().__init__()

    class PdfFormatOption:
        def __init__(self, pipeline_options=None) -> None:  # noqa: ANN001
            self.pipeline_options = pipeline_options

    converter_module.DocumentConverter = ConverterAdapter
    converter_module.PdfFormatOption = PdfFormatOption

    monkeypatch.setitem(sys.modules, "docling", docling_pkg)
    monkeypatch.setitem(sys.modules, "docling.datamodel", datamodel_pkg)
    monkeypatch.setitem(
        sys.modules, "docling.datamodel.base_models", base_models_module
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.accelerator_options",
        accelerator_options_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        pipeline_options_module,
    )
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)


def install_fake_pix2text(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outputs: list[dict[str, object]],
    state: dict[str, object] | None = None,
) -> None:
    pix2text_module = types.ModuleType("pix2text")

    class FakePix2Text:
        @classmethod
        def from_config(cls, **_kwargs):  # noqa: ANN003
            if state is not None:
                state["from_config_calls"] = int(state.get("from_config_calls", 0)) + 1
            return cls()

        def recognize_text_formula(
            self,
            _image,  # noqa: ANN001
            *,
            return_text: bool = False,
            mfr_batch_size: int = 1,
        ):
            assert return_text is False
            if state is not None:
                calls = state.setdefault("batch_sizes", [])
                assert isinstance(calls, list)
                calls.append(mfr_batch_size)
            return outputs

    pix2text_module.Pix2Text = FakePix2Text
    monkeypatch.setitem(sys.modules, "pix2text", pix2text_module)


class SuccessfulConverter:
    def convert(self, _path: str) -> FakeConvertResult:
        return FakeConvertResult("# Intro\nAlpha\n## Method\nBeta\n### Deep\nGamma")


class FailingConverter:
    def convert(self, _path: str) -> FakeConvertResult:
        raise RuntimeError("docling failed")


class FormulaConverter:
    def convert(self, _path: str) -> FakeConvertResult:
        return FakeConvertResult("# Intro\nx(t) = sin(t)\n<!-- formula-not-decoded -->")


class MultiFormulaConverter:
    def convert(self, _path: str) -> FakeConvertResult:
        return FakeConvertResult(
            f"# Intro\nx(t) = sin(t)\n{_FORMULA_MARKER}\nMore text\n{_FORMULA_MARKER}"
        )


def test_docling_parse_missing_file(tmp_path: Path) -> None:
    parser = DoclingPdfParser()
    with pytest.raises(AppError) as exc:
        parser.parse(str(tmp_path / "missing.pdf"), "doc1")
    assert exc.value.code == ErrorCode.INGEST_DOC_NOT_FOUND


def test_docling_import_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"pdf")

    real_import = builtins.__import__

    def fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ARG001
        if name == "docling.datamodel.base_models":
            raise ImportError("missing docling")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(AppError) as exc:
        parser.parse(str(pdf), "doc1")

    assert exc.value.code == ErrorCode.INGEST_PDF_DOCLING_FAILED


def test_docling_parse_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, FailingConverter)
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open", lambda _path: FakePdfDoc()
    )

    with pytest.raises(AppError) as exc:
        parser.parse(str(pdf), "doc1")

    assert exc.value.code == ErrorCode.INGEST_PDF_DOCLING_FAILED


def test_docling_parse_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, SuccessfulConverter)
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(
            title="My PDF",
            page_count=4,
            toc=[[1, "Intro", 1], [2, "Method", 2]],
            page_texts=["Intro text", "Method and Deep text", "More Deep text", "end"],
        ),
    )

    parsed = parser.parse(str(pdf), "doc1")

    assert parsed.title == "My PDF"
    assert parsed.parser_chain == ["docling"]
    assert len(parsed.outline) == 2
    assert len(parsed.chunks) == 3
    assert parsed.chunks[0].section_path == ["Intro"]
    assert parsed.chunks[1].section_path == ["Intro", "Method"]
    assert parsed.chunks[0].locator.page_range == [1, 1]
    assert parsed.chunks[1].locator.page_range == [2, 2]
    assert parsed.chunks[2].locator.page_range == [2, 4]
    assert parsed.outline[0].page_end == 4
    assert parsed.outline[1].page_end == 4
    assert parsed.metadata["pages"] == 4
    assert parsed.metadata["toc_nodes_raw"] == 2
    assert parsed.metadata["toc_nodes_clean"] == 2
    assert parsed.metadata["formula_markers_total"] == 0
    assert parsed.formulas == []


def test_docling_parse_skips_page_loading_without_formula_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "no_formula.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, SuccessfulConverter)
    fake_doc = CountingLoadPagePdfDoc(
        title="No Formula PDF",
        page_count=5,
        toc=[[1, "Intro", 1], [2, "Method", 3]],
        page_texts=["a", "b", "c", "d", "e"],
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: fake_doc,
    )

    parsed = parser.parse(str(pdf), "doc-no-formula")

    assert parsed.metadata["formula_markers_total"] == 0
    assert fake_doc.load_page_calls == 0


def test_docling_parse_replaces_formula_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, FormulaConverter)
    install_fake_pix2text(
        monkeypatch,
        outputs=[
            {
                "type": "isolated",
                "text": r"\frac{a}{b}",
                "score": 0.98,
                "position": [[0, 0], [20, 0], [20, 10], [0, 10]],
            }
        ],
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(
            title="Formula PDF",
            page_count=2,
            toc=[[1, "Intro", 1]],
            page_texts=["x(t) = sin(t)", "y = mx + b"],
        ),
    )

    parsed = parser.parse(str(pdf), "doc-formula")

    assert parsed.parser_chain == ["docling", "pix2text"]
    assert parsed.metadata["formula_markers_total"] == 1
    assert parsed.metadata["formula_replaced_by_pix2text"] == 1
    assert parsed.metadata["formula_records_total"] == 1
    assert "<!-- formula-not-decoded -->" not in parsed.chunks[0].text
    assert r"\frac{a}{b}" in parsed.chunks[0].text
    assert len(parsed.formulas) == 1
    assert parsed.formulas[0].status == "resolved"
    assert parsed.formulas[0].source == "pix2text"
    assert parsed.formulas[0].latex == r"\frac{a}{b}"
    assert parsed.formulas[0].chunk_id == parsed.chunks[0].chunk_id


def test_docling_parse_formula_fallback_uses_page_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "formula_fallback.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, FormulaConverter)
    install_fake_pix2text(monkeypatch, outputs=[])
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(
            title="Formula Fallback PDF",
            page_count=2,
            toc=[[1, "Intro", 1]],
            page_texts=["x(t) = sin(t)", "plain text"],
        ),
    )

    parsed = parser.parse(str(pdf), "doc-formula-fallback")

    assert parsed.metadata["formula_markers_total"] == 1
    assert parsed.metadata["formula_replaced_by_pix2text"] == 0
    assert parsed.metadata["formula_replaced_by_fallback"] == 1
    assert "Formula fallback text" in parsed.chunks[0].text
    assert len(parsed.formulas) == 1
    assert parsed.formulas[0].status == "fallback_text"
    assert "x(t) = sin(t)" in parsed.formulas[0].latex


def test_docling_parse_formula_unresolved_when_no_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "formula_unresolved.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, FormulaConverter)
    install_fake_pix2text(monkeypatch, outputs=[])
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(
            title="Formula Unresolved PDF",
            page_count=2,
            toc=[[1, "Intro", 1]],
            page_texts=["plain intro text", "plain method text"],
        ),
    )

    parsed = parser.parse(str(pdf), "doc-formula-unresolved")

    assert parsed.metadata["formula_markers_total"] == 1
    assert parsed.metadata["formula_replaced_by_fallback"] == 0
    assert parsed.metadata["formula_unresolved"] == 1
    assert len(parsed.formulas) == 1
    assert parsed.formulas[0].status == "unresolved"


def test_docling_parse_formula_engine_missing_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser(require_formula_engine=True)
    pdf = tmp_path / "f2.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, FormulaConverter)
    monkeypatch.setattr(
        parser.formula_extractor,
        "_ensure_engine",
        lambda: (_ for _ in ()).throw(
            AppError(
                ErrorCode.INGEST_PDF_FORMULA_ENGINE_UNAVAILABLE,
                "missing formula engine",
            )
        ),
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(
            title="Formula PDF",
            page_count=2,
            toc=[[1, "Intro", 1]],
            page_texts=["x(t) = sin(t)", "y = mx + b"],
        ),
    )

    with pytest.raises(AppError) as exc:
        parser.parse(str(pdf), "doc-formula-missing")

    assert exc.value.code == ErrorCode.INGEST_PDF_FORMULA_ENGINE_UNAVAILABLE


def test_docling_parse_filters_noisy_toc_titles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "n.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, SuccessfulConverter)
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(
            title="Noisy PDF",
            page_count=3,
            toc=[
                [1, "Intro", 1],
                [2, "def foo(x): return x class A: pass", 1],
                [2, "Method", 2],
            ],
        ),
    )

    parsed = parser.parse(str(pdf), "doc-noisy")

    assert [node.title for node in parsed.outline] == ["Intro", "Method"]


def test_docling_parse_passes_formula_batch_size_to_pix2text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state: dict[str, object] = {}
    parser = DoclingPdfParser(formula_batch_size=5)
    pdf = tmp_path / "formula_batch.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, FormulaConverter)
    install_fake_pix2text(
        monkeypatch,
        outputs=[
            {
                "type": "isolated",
                "text": r"\sigma_t",
                "score": 0.91,
                "position": [[0, 0], [20, 0], [20, 10], [0, 10]],
            }
        ],
        state=state,
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(page_count=1, toc=[[1, "Intro", 1]]),
    )

    parsed = parser.parse(str(pdf), "doc-formula-batch")

    assert parsed.metadata["formula_replaced_by_pix2text"] == 1
    assert state["batch_sizes"] == [5]


def test_docling_parse_caches_formula_candidates_per_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "formula_cache.pdf"
    pdf.write_bytes(b"pdf")

    install_fake_docling(monkeypatch, MultiFormulaConverter)
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(page_count=1, toc=[[1, "Intro", 1]]),
    )
    calls: list[int] = []

    def fake_extract(_image, page: int):  # noqa: ANN001
        calls.append(page)
        return [
            types.SimpleNamespace(
                latex="a_1",
                page=page,
                score=0.9,
                bbox=[0.0, 0.0, 1.0, 1.0],
                source="pix2text",
            ),
            types.SimpleNamespace(
                latex="a_2",
                page=page,
                score=0.8,
                bbox=[0.0, 2.0, 1.0, 3.0],
                source="pix2text",
            ),
        ]

    monkeypatch.setattr(parser.formula_extractor, "_ensure_engine", lambda: object())
    monkeypatch.setattr(parser.formula_extractor, "extract", fake_extract)

    parsed = parser.parse(str(pdf), "doc-formula-cache")

    assert calls == [1]
    assert len(parsed.formulas) == 2
    assert parsed.formulas[0].latex == "a_1"
    assert parsed.formulas[1].latex == "a_2"


def test_docling_parse_caches_converter_between_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "cache.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        lambda _path: FakePdfDoc(page_count=1, toc=[[1, "Intro", 1]]),
    )

    build_calls: list[str] = []

    def fake_build() -> SuccessfulConverter:
        build_calls.append("build")
        return SuccessfulConverter()

    monkeypatch.setattr(parser, "_build_docling_converter", fake_build)

    parser.parse(str(pdf), "doc-cache-1")
    parser.parse(str(pdf), "doc-cache-2")

    assert build_calls == ["build"]


def test_docling_autotune_selects_fastest_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = DoclingPdfParser()
    pdf = tmp_path / "autotune.pdf"
    pdf.write_bytes(b"pdf")
    cleanup_calls: list[str] = []

    def fake_fitz_open(path: str | None = None):  # noqa: ANN001
        if path is None:
            return FakePdfDoc(page_count=0, toc=[])
        return FakePdfDoc(page_count=5, toc=[[1, "Intro", 1]])

    def make_converter(label: str) -> object:
        class FakeEngine:
            def cleanup(self) -> None:
                cleanup_calls.append(label)

        pipeline = types.SimpleNamespace(
            preprocessing_model=None,
            ocr_model=None,
            layout_model=None,
            table_model=None,
            assemble_model=None,
            reading_order_model=None,
            build_pipe=[],
            enrichment_pipe=[types.SimpleNamespace(engine=FakeEngine())],
        )
        return types.SimpleNamespace(
            initialized_pipelines={("pipeline", label): pipeline}
        )

    slow_converter = make_converter("slow")
    fast_converter = make_converter("fast")

    rows = iter(
        [
            (
                PdfParserBenchmarkRow(
                    label="warmup",
                    config=PdfParserPerformanceConfig(num_threads=4),
                    convert_seconds=0.1,
                    export_seconds=0.01,
                    markdown_chars=10,
                    formula_markers=0,
                ),
                None,
            ),
            (
                PdfParserBenchmarkRow(
                    label="threads=4",
                    config=PdfParserPerformanceConfig(num_threads=4),
                    convert_seconds=2.0,
                    export_seconds=0.1,
                    markdown_chars=10,
                    formula_markers=0,
                ),
                slow_converter,
            ),
            (
                PdfParserBenchmarkRow(
                    label="threads=8",
                    config=PdfParserPerformanceConfig(num_threads=8),
                    convert_seconds=1.0,
                    export_seconds=0.1,
                    markdown_chars=10,
                    formula_markers=0,
                ),
                fast_converter,
            ),
        ]
    )

    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_docling.fitz.open",
        fake_fitz_open,
    )
    monkeypatch.setattr(
        parser,
        "_benchmark_docling_config",
        lambda **_kwargs: next(rows),
    )

    profile = parser.autotune(
        pdf_path=str(pdf),
        sample_pages=3,
        candidate_configs=[
            PdfParserPerformanceConfig(num_threads=4),
            PdfParserPerformanceConfig(num_threads=8),
        ],
    )

    assert profile.sample_pages == 3
    assert profile.selected_config.num_threads == 8
    assert len(profile.benchmarks) == 2
    assert parser._pending_converter is fast_converter
    assert cleanup_calls == ["slow"]

    parser.set_performance_config(profile.selected_config)

    assert parser._converter is fast_converter
    assert parser._pending_converter is None


def test_docling_close_cleans_pipeline_engines_and_clears_cache() -> None:
    parser = DoclingPdfParser()
    engine_calls: list[str] = []

    class FakeEngine:
        def cleanup(self) -> None:
            engine_calls.append("cleanup")

    model = types.SimpleNamespace(engine=FakeEngine())
    pipeline = types.SimpleNamespace(
        preprocessing_model=None,
        ocr_model=None,
        layout_model=None,
        table_model=None,
        assemble_model=None,
        reading_order_model=None,
        build_pipe=[],
        enrichment_pipe=[model],
    )
    parser._converter = types.SimpleNamespace(
        initialized_pipelines={("pipeline", "hash"): pipeline}
    )

    parser.close()

    assert engine_calls == ["cleanup"]
    assert model.engine is None
    assert parser._converter is None
