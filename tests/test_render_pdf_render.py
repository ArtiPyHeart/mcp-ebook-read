from __future__ import annotations

from pathlib import Path

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.render.pdf_render import render_pdf_page


class FakePixmap:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self.saved_path: str | None = None

    def save(self, path: str) -> None:
        self.saved_path = path
        Path(path).write_bytes(b"png")


class FakePage:
    def get_pixmap(self, dpi: int) -> FakePixmap:  # noqa: ARG002
        return FakePixmap()


class FakeDoc:
    def __init__(self, page_count: int) -> None:
        self.page_count = page_count
        self.closed = False

    def __getitem__(self, idx: int) -> FakePage:  # noqa: ARG002
        return FakePage()

    def close(self) -> None:
        self.closed = True


def test_render_pdf_page_invalid_page() -> None:
    with pytest.raises(AppError) as exc:
        render_pdf_page("/tmp/x.pdf", Path("/tmp/out.png"), page=0)
    assert exc.value.code == ErrorCode.RENDER_PAGE_FAILED


def test_render_pdf_page_out_of_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mcp_ebook_read.render.pdf_render.fitz.open",
        lambda _path: FakeDoc(page_count=2),
    )

    with pytest.raises(AppError) as exc:
        render_pdf_page("/tmp/x.pdf", tmp_path / "out.png", page=3)

    assert exc.value.code == ErrorCode.RENDER_PAGE_FAILED
    assert "out of range" in exc.value.message


def test_render_pdf_page_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mcp_ebook_read.render.pdf_render.fitz.open",
        lambda _path: FakeDoc(page_count=5),
    )

    out = tmp_path / "render" / "page.png"
    width, height = render_pdf_page("/tmp/x.pdf", out, page=1, dpi=120)

    assert (width, height) == (640, 480)
    assert out.exists()
