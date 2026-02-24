"""PDF page rendering."""

from __future__ import annotations

from pathlib import Path

import fitz

from mcp_ebook_read.errors import AppError, ErrorCode


def render_pdf_page(
    pdf_path: str, out_path: Path, page: int, dpi: int = 200
) -> tuple[int, int]:
    """Render a single 1-based PDF page to PNG."""
    if page <= 0:
        raise AppError(ErrorCode.RENDER_PAGE_FAILED, "Page must be >= 1")

    try:
        doc = fitz.open(pdf_path)
        if page > doc.page_count:
            doc.close()
            raise AppError(
                ErrorCode.RENDER_PAGE_FAILED,
                f"Page {page} out of range (1..{doc.page_count})",
            )
        target = doc[page - 1]
        pix = target.get_pixmap(dpi=dpi)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        width, height = pix.width, pix.height
        doc.close()
        return width, height
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AppError(
            ErrorCode.RENDER_PAGE_FAILED,
            f"Failed to render page {page}",
        ) from exc
