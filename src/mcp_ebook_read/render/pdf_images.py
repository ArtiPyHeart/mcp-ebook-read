"""Extract figure-like images from PDF pages for multimodal reading."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import fitz

from mcp_ebook_read.schema.models import ChunkRecord, ImageRecord

_CAPTION_PREFIX = re.compile(r"^(figure|fig\.?|table)\s*\d+", re.IGNORECASE)


class PdfImageExtractor:
    """Extract and crop PDF images with basic caption and section mapping."""

    def __init__(self, *, min_area_ratio: float = 0.01, zoom: float = 2.0) -> None:
        self.min_area_ratio = max(0.0, min_area_ratio)
        self.zoom = max(1.0, zoom)

    def _section_path_for_page(self, chunks: list[ChunkRecord], page: int) -> list[str]:
        for chunk in chunks:
            page_range = chunk.locator.page_range
            if not page_range or len(page_range) != 2:
                continue
            if page_range[0] <= page <= page_range[1]:
                return chunk.section_path
        return []

    def _caption_for_rect(
        self,
        blocks: list[tuple[Any, ...]],
        rect: fitz.Rect,
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for block in blocks:
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            content = " ".join(str(text).split())
            if not content:
                continue

            if y0 >= rect.y1:
                distance = y0 - rect.y1
                if distance > 140:
                    continue
                score = float(distance)
            elif y1 <= rect.y0:
                distance = rect.y0 - y1
                if distance > 90:
                    continue
                score = float(distance + 30)
            else:
                continue

            overlap = max(0.0, min(rect.x1, x1) - max(rect.x0, x0))
            overlap_ratio = overlap / max(rect.width, 1.0)
            score -= min(overlap_ratio, 1.0) * 20.0
            if _CAPTION_PREFIX.search(content):
                score -= 40.0
            candidates.append((score, content))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1][:400]

    def extract(
        self,
        *,
        pdf_path: str,
        doc_id: str,
        chunks: list[ChunkRecord],
        out_dir: Path,
    ) -> list[ImageRecord]:
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[ImageRecord] = []
        seen_rects: set[tuple[int, float, float, float, float]] = set()
        section_cache: dict[int, list[str]] = {}
        order_index = 0

        with fitz.open(pdf_path) as pdf_doc:
            for page_index in range(pdf_doc.page_count):
                page = pdf_doc.load_page(page_index)
                page_num = page_index + 1
                page_area = max(page.rect.width * page.rect.height, 1.0)
                images = page.get_images(full=True)
                caption_blocks = page.get_text("blocks")
                if page_num not in section_cache:
                    section_cache[page_num] = self._section_path_for_page(
                        chunks, page_num
                    )

                for image_idx, image_info in enumerate(images):
                    if not image_info:
                        continue
                    xref = int(image_info[0])
                    if xref <= 0:
                        continue
                    for rect_idx, rect in enumerate(page.get_image_rects(xref)):
                        key = (
                            page_num,
                            round(rect.x0, 2),
                            round(rect.y0, 2),
                            round(rect.x1, 2),
                            round(rect.y1, 2),
                        )
                        if key in seen_rects:
                            continue
                        seen_rects.add(key)

                        ratio = (rect.width * rect.height) / page_area
                        if ratio < self.min_area_ratio:
                            continue

                        clip = fitz.Rect(
                            max(0, rect.x0),
                            max(0, rect.y0),
                            min(page.rect.x1, rect.x1),
                            min(page.rect.y1, rect.y1),
                        )
                        if clip.width <= 0 or clip.height <= 0:
                            continue

                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(self.zoom, self.zoom),
                            clip=clip,
                            alpha=False,
                        )
                        png_bytes = pix.tobytes("png")
                        digest = hashlib.sha1(
                            png_bytes, usedforsecurity=False
                        ).hexdigest()[:8]
                        image_id = hashlib.sha1(
                            f"{doc_id}:{page_num}:{image_idx}:{rect_idx}:{key}".encode(),
                            usedforsecurity=False,
                        ).hexdigest()[:16]

                        output_path = (
                            out_dir
                            / f"page_{page_num:04d}_{order_index:04d}_{digest}.png"
                        )
                        output_path.write_bytes(png_bytes)

                        records.append(
                            ImageRecord(
                                image_id=image_id,
                                doc_id=doc_id,
                                order_index=order_index,
                                section_path=section_cache[page_num],
                                page=page_num,
                                bbox=[
                                    round(float(clip.x0), 2),
                                    round(float(clip.y0), 2),
                                    round(float(clip.x1), 2),
                                    round(float(clip.y1), 2),
                                ],
                                caption=self._caption_for_rect(caption_blocks, clip),
                                media_type="image/png",
                                file_path=str(output_path),
                                width=pix.width,
                                height=pix.height,
                                source="pdf-image-extractor",
                                status="ready",
                            )
                        )
                        order_index += 1

        return records
