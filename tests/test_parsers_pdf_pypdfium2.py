from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mcp_ebook_read.errors import ErrorCode
from mcp_ebook_read.parsers.pdf_pypdfium2 import Pypdfium2PdfParser


class FakeTextPage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False

    def get_text_range(self) -> str:
        return self.text

    def close(self) -> None:
        self.closed = True


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.closed = False

    def get_textpage(self) -> FakeTextPage:
        return FakeTextPage(self.text)

    def close(self) -> None:
        self.closed = True


class FakeDest:
    def __init__(self, index: int) -> None:
        self.index = index

    def get_index(self) -> int:
        return self.index


class FakeBookmark:
    def __init__(self, title: str, level: int, page_index: int) -> None:
        self.title = title
        self.level = level
        self.page_index = page_index

    def get_title(self) -> str:
        return self.title

    def get_dest(self) -> FakeDest:
        return FakeDest(self.page_index)


class FakePdfDocument:
    instances: list["FakePdfDocument"] = []

    def __init__(self, _path: str) -> None:
        self.closed = False
        self.pages = [
            FakePage("Intro page text"),
            FakePage("Method page text"),
            FakePage("Deep method page text"),
        ]
        self.toc = [
            FakeBookmark("Intro", 0, 0),
            FakeBookmark("Method", 0, 1),
            FakeBookmark("Deep", 1, 2),
        ]
        FakePdfDocument.instances.append(self)

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> FakePage:
        return self.pages[index]

    def get_metadata_dict(self) -> dict[str, str]:
        return {"Title": "Fast PDF", "Author": "Tester"}

    def get_toc(self):  # noqa: ANN201
        return iter(self.toc)

    def close(self) -> None:
        self.closed = True


def install_fake_pypdfium2(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("pypdfium2")
    module.PdfDocument = FakePdfDocument
    monkeypatch.setitem(sys.modules, "pypdfium2", module)
    FakePdfDocument.instances.clear()


def test_pypdfium2_fast_parser_builds_parsed_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_pypdfium2(monkeypatch)
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF")

    parsed = Pypdfium2PdfParser().parse(str(pdf), "doc-fast")

    assert parsed.title == "Fast PDF"
    assert parsed.parser_chain == ["pypdfium2-fast"]
    assert parsed.metadata["fast_parser"] is True
    assert parsed.metadata["fidelity_lane"] == "preview_text_only"
    assert parsed.metadata["pages"] == 3
    assert parsed.metadata["toc_nodes_clean"] == 3
    assert parsed.metadata["pdf_parse_phase_seconds"]["total"] >= 0
    assert [node.title for node in parsed.outline] == ["Intro", "Method", "Deep"]
    assert parsed.outline[0].page_start == 1
    assert parsed.outline[0].page_end == 1
    assert parsed.outline[1].page_start == 2
    assert parsed.outline[1].page_end == 3
    assert parsed.chunks
    assert parsed.chunks[0].method == "pypdfium2-fast"
    assert "Intro page text" in parsed.reading_markdown
    assert FakePdfDocument.instances[-1].closed is True


def test_pypdfium2_fast_parser_missing_file(tmp_path: Path) -> None:
    with pytest.raises(Exception) as exc_info:
        Pypdfium2PdfParser().parse(str(tmp_path / "missing.pdf"), "doc-fast")

    assert getattr(exc_info.value, "code") == ErrorCode.INGEST_DOC_NOT_FOUND


def test_pypdfium2_fast_parser_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF")

    with pytest.raises(Exception) as exc_info:
        Pypdfium2PdfParser().parse(str(pdf), "doc-fast")

    assert getattr(exc_info.value, "code") == ErrorCode.INGEST_PDF_FAST_PARSE_FAILED


def test_pypdfium2_fast_parser_falls_back_to_page_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_pypdfium2(monkeypatch)
    pdf = tmp_path / "no-toc.pdf"
    pdf.write_bytes(b"%PDF")

    original_init = FakePdfDocument.__init__

    def init_no_toc(self: FakePdfDocument, path: str) -> None:
        original_init(self, path)
        self.toc = []

    monkeypatch.setattr(FakePdfDocument, "__init__", init_no_toc)

    parsed = Pypdfium2PdfParser().parse(str(pdf), "doc-pages")

    assert parsed.outline == []
    assert len(parsed.chunks) == 3
    assert parsed.chunks[0].section_path == ["Page 1"]
    assert parsed.chunks[0].locator.page_range == [1, 1]
