from __future__ import annotations

import builtins
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser


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
    pipeline_options_module = types.ModuleType("docling.datamodel.pipeline_options")

    class InputFormat:
        PDF = "pdf"

    class PdfPipelineOptions:
        def __init__(self) -> None:
            self.do_formula_enrichment = False

    base_models_module.InputFormat = InputFormat
    pipeline_options_module.PdfPipelineOptions = PdfPipelineOptions

    converter_module = types.ModuleType("docling.document_converter")

    class ConverterAdapter(converter_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
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
        "docling.datamodel.pipeline_options",
        pipeline_options_module,
    )
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)


def install_fake_pix2text(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outputs: list[dict[str, object]],
) -> None:
    pix2text_module = types.ModuleType("pix2text")

    class FakePix2Text:
        @classmethod
        def from_config(cls, **_kwargs):  # noqa: ANN003
            return cls()

        def recognize_text_formula(
            self,
            _image,  # noqa: ANN001
            *,
            return_text: bool = False,
            mfr_batch_size: int = 1,  # noqa: ARG002
        ):
            assert return_text is False
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
    assert "<!-- formula-not-decoded -->" not in parsed.chunks[0].text
    assert r"\frac{a}{b}" in parsed.chunks[0].text


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
