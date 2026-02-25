"""PDF page rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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


def render_pdf_region(
    pdf_path: str,
    out_path: Path,
    *,
    page: int,
    bbox: Sequence[float],
    dpi: int = 220,
    padding_pt: float = 4.0,
) -> tuple[int, int]:
    """Render one PDF bbox region on a 1-based page to PNG."""
    if page <= 0:
        raise AppError(ErrorCode.RENDER_PAGE_FAILED, "Page must be >= 1")
    if len(bbox) != 4:
        raise AppError(
            ErrorCode.RENDER_PAGE_FAILED,
            "bbox must be [x0, y0, x1, y1]",
        )

    try:
        x0 = float(bbox[0])
        y0 = float(bbox[1])
        x1 = float(bbox[2])
        y1 = float(bbox[3])
    except (TypeError, ValueError) as exc:
        raise AppError(
            ErrorCode.RENDER_PAGE_FAILED,
            "bbox values must be numeric",
        ) from exc

    if x1 <= x0 or y1 <= y0:
        raise AppError(
            ErrorCode.RENDER_PAGE_FAILED,
            "bbox must define a positive area",
        )

    try:
        doc = fitz.open(pdf_path)
        if page > doc.page_count:
            doc.close()
            raise AppError(
                ErrorCode.RENDER_PAGE_FAILED,
                f"Page {page} out of range (1..{doc.page_count})",
            )

        target = doc[page - 1]
        page_rect = target.rect
        clip = fitz.Rect(
            x0 - padding_pt,
            y0 - padding_pt,
            x1 + padding_pt,
            y1 + padding_pt,
        )
        clip = fitz.Rect(
            max(clip.x0, page_rect.x0),
            max(clip.y0, page_rect.y0),
            min(clip.x1, page_rect.x1),
            min(clip.y1, page_rect.y1),
        )
        if clip.width <= 0 or clip.height <= 0:
            doc.close()
            raise AppError(
                ErrorCode.RENDER_PAGE_FAILED,
                "bbox does not overlap the target PDF page",
            )

        zoom = max(float(dpi), 36.0) / 72.0
        pix = target.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            clip=clip,
            alpha=False,
        )
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
            f"Failed to render region on page {page}",
        ) from exc
