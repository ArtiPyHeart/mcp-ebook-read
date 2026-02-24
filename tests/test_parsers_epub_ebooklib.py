from __future__ import annotations

from pathlib import Path

import pytest
from ebooklib import epub

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.parsers.epub_ebooklib import EbooklibEpubParser, ITEM_DOCUMENT


class FakeEpubItem:
    def __init__(
        self,
        content: bytes,
        item_type: int = ITEM_DOCUMENT,
        name: str = "item.xhtml",
        media_type: str | None = None,
    ) -> None:
        self._content = content
        self._item_type = item_type
        self._name = name
        self.media_type = media_type

    def get_content(self) -> bytes:
        return self._content

    def get_type(self) -> int:
        return self._item_type

    def get_name(self) -> str:
        return self._name


class FakeEpubBook:
    def __init__(
        self,
        title: str,
        spine: list[tuple[str, str]],
        items: dict[str, FakeEpubItem],
        toc: list[object] | None = None,
    ) -> None:
        self._title = title
        self.spine = spine
        self._items = items
        self.toc = toc or []

    def get_metadata(self, _ns: str, _name: str) -> list[tuple[str, dict[str, str]]]:  # noqa: ARG002
        return [(self._title, {})]

    def get_item_with_id(self, spine_id: str) -> FakeEpubItem | None:
        return self._items.get(spine_id)

    def get_items(self):
        return list(self._items.values())


def test_epub_parse_missing_file(tmp_path: Path) -> None:
    parser = EbooklibEpubParser()
    with pytest.raises(AppError) as exc:
        parser.parse(str(tmp_path / "missing.epub"), "doc1")
    assert exc.value.code == ErrorCode.INGEST_DOC_NOT_FOUND


def test_epub_parse_read_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = EbooklibEpubParser()
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    def _raise(_path: str) -> FakeEpubBook:  # noqa: ARG001
        raise RuntimeError("bad epub")

    monkeypatch.setattr("mcp_ebook_read.parsers.epub_ebooklib.epub.read_epub", _raise)

    with pytest.raises(AppError) as exc:
        parser.parse(str(epub_path), "doc1")

    assert exc.value.code == ErrorCode.INGEST_EPUB_PARSE_FAILED


def test_epub_parse_invalid_xhtml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = EbooklibEpubParser()
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    book = FakeEpubBook(
        title="Book",
        spine=[("chap1", "yes")],
        items={"chap1": FakeEpubItem(b"<html></html>")},
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.epub_ebooklib.epub.read_epub", lambda _path: book
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.epub_ebooklib.html.fromstring",
        lambda _content: (_ for _ in ()).throw(ValueError("invalid xhtml")),
    )

    with pytest.raises(AppError) as exc:
        parser.parse(str(epub_path), "doc1")

    assert exc.value.code == ErrorCode.INGEST_EPUB_PARSE_FAILED


def test_epub_parse_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parser = EbooklibEpubParser()
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    chapter_html = b"""
    <html><body>
      <h1 id="ch1">Chapter One</h1>
      <p>First paragraph.</p>
      <h2 id="s11">Section 1.1</h2>
      <p>Second paragraph.</p>
      <li>List item.</li>
    </body></html>
    """
    appendix_html = b"<html><body><p>Appendix text.</p></body></html>"
    book = FakeEpubBook(
        title="EPUB Title",
        spine=[("chap1", "yes"), ("appendix", "yes")],
        items={
            "chap1": FakeEpubItem(chapter_html, name="chap1.xhtml"),
            "appendix": FakeEpubItem(appendix_html, name="appendix.xhtml"),
        },
        toc=[
            (
                epub.Section("Chapter One", "chap1.xhtml"),
                [epub.Link("chap1.xhtml#s11", "Section 1.1", None)],
            ),
            epub.Link("appendix.xhtml", "Appendix", None),
        ],
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.epub_ebooklib.epub.read_epub", lambda _path: book
    )

    parsed = parser.parse(str(epub_path), "doc-epub")

    assert parsed.title == "EPUB Title"
    assert parsed.parser_chain == ["ebooklib"]
    assert len(parsed.chunks) == 3
    assert parsed.chunks[0].section_path == ["Chapter One"]
    assert parsed.chunks[1].section_path == ["Chapter One", "Section 1.1"]
    assert parsed.chunks[1].locator.epub_locator == {
        "spine_id": "chap1",
        "href": "chap1.xhtml",
        "anchor": "s11",
    }
    assert parsed.chunks[2].section_path == ["Appendix"]
    assert parsed.outline[0].title == "Chapter One"
    assert parsed.outline[0].children[0].title == "Section 1.1"
    assert parsed.outline[0].children[0].level == 2
    assert parsed.metadata["spine_items"] == 2
    assert parsed.metadata["toc_nodes"] == 3
    assert parsed.metadata["chunking"] == "heading_sections"
    assert parsed.metadata["images_extracted"] == 0
    assert "Chapter One / Section 1.1" in parsed.reading_markdown


def test_epub_parse_extracts_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parser = EbooklibEpubParser()
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"epub")

    chapter_html = b"""
    <html><body>
      <h1 id="ch1">Chapter One</h1>
      <p>Before image.</p>
      <figure>
        <img id="img-anchor" src="images/plot.png" alt="Performance chart" width="640" />
        <figcaption>Training loss plot</figcaption>
      </figure>
      <h2 id="s11">Section 1.1</h2>
      <p>After image.</p>
    </body></html>
    """
    book = FakeEpubBook(
        title="Image EPUB",
        spine=[("chap1", "yes")],
        items={
            "chap1": FakeEpubItem(chapter_html, name="chapters/chap1.xhtml"),
            "img1": FakeEpubItem(
                b"\x89PNG\r\n\x1a\n\x00\x00",
                item_type=9,
                name="chapters/images/plot.png",
                media_type="image/png",
            ),
        },
        toc=[epub.Link("chapters/chap1.xhtml", "Chapter One", None)],
    )
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.epub_ebooklib.epub.read_epub", lambda _path: book
    )

    parsed = parser.parse(str(epub_path), "doc-epub-image")

    assert parsed.metadata["images_extracted"] == 1
    assert len(parsed.images) == 1
    image = parsed.images[0]
    assert image.doc_id == "doc-epub-image"
    assert image.section_path == ["Chapter One"]
    assert image.spine_id == "chap1"
    assert image.href == "chapters/images/plot.png"
    assert image.anchor == "img-anchor"
    assert image.alt == "Performance chart"
    assert image.caption == "Training loss plot"
    assert image.media_type == "image/png"
    assert image.extension == ".png"
    assert image.width == 640
    assert image.data.startswith(b"\x89PNG")
