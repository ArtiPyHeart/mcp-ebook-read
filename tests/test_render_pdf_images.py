from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

from mcp_ebook_read.render.pdf_images import PdfImageExtractor
from mcp_ebook_read.schema.models import ChunkRecord, Locator


def _png_bytes(color: str, size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_pdf_image_extractor_respects_area_threshold(tmp_path: Path) -> None:
    pdf_path = tmp_path / "figures.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=800)
        large_img = _png_bytes("red", (400, 400))
        small_img = _png_bytes("blue", (20, 20))
        page.insert_image(fitz.Rect(100, 100, 400, 400), stream=large_img)
        page.insert_image(fitz.Rect(10, 10, 30, 30), stream=small_img)
        page.insert_text((100, 430), "Figure 1: Performance comparison")
        doc.save(str(pdf_path))

    chunks = [
        ChunkRecord(
            chunk_id="chunk-1",
            doc_id="doc-pdf-1",
            order_index=0,
            section_path=["Chapter 1"],
            text="context",
            search_text="context",
            locator=Locator(
                doc_id="doc-pdf-1",
                chunk_id="chunk-1",
                section_path=["Chapter 1"],
                page_range=[1, 1],
                method="docling",
            ),
            method="docling",
        )
    ]
    extractor = PdfImageExtractor(min_area_ratio=0.01)
    out_dir = tmp_path / "assets"
    images = extractor.extract(
        pdf_path=str(pdf_path),
        doc_id="doc-pdf-1",
        chunks=chunks,
        out_dir=out_dir,
    )

    assert len(images) == 1
    image = images[0]
    assert image.page == 1
    assert image.section_path == ["Chapter 1"]
    assert image.caption is not None
    assert image.caption.startswith("Figure 1")
    assert image.file_path.endswith(".png")
    assert Path(image.file_path).exists()
    assert image.width > 0
    assert image.height > 0
    assert image.bbox is not None
    assert len(image.bbox) == 4
