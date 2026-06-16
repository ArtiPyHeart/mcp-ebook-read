"""Fast PDF parser based on pypdfium2 for preview/search lanes."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    Locator,
    OutlineNode,
    ParsedDocument,
)


@dataclass(slots=True)
class _PageText:
    page: int
    text: str


def _sanitize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _outline_page_end(
    nodes: list[OutlineNode], index: int, page_count: int
) -> int | None:
    current = nodes[index]
    if current.page_start is None:
        return None
    for next_node in nodes[index + 1 :]:
        if next_node.page_start is None:
            continue
        if next_node.level <= current.level:
            return max(current.page_start, next_node.page_start - 1)
    return page_count


def _build_section_path(nodes: list[OutlineNode], index: int) -> list[str]:
    current = nodes[index]
    path: list[str] = []
    for node in reversed(nodes[: index + 1]):
        if node.level <= current.level and (not path or node.level < current.level):
            path.insert(0, node.title)
            current = node
    return path or [nodes[index].title]


class Pypdfium2PdfParser:
    """Fast text-first PDF parser.

    This parser is designed for fast preview, discovery, and coarse local search.
    It intentionally does not recover formulas, tables, figures, or visual evidence.
    Use Docling-based parsing when high-fidelity reading is required.
    """

    method = "pypdfium2-fast"

    def parse(self, pdf_path: str, doc_id: str) -> ParsedDocument:
        started = time.perf_counter()
        path = Path(pdf_path)
        if not path.exists():
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                f"Document not found: {pdf_path}",
            )

        try:
            import pypdfium2 as pdfium
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_FAST_PARSE_FAILED,
                "pypdfium2 is unavailable for fast PDF parsing.",
            ) from exc

        phase_seconds: dict[str, float] = {}

        def record_phase(name: str, phase_started: float) -> None:
            phase_seconds[name] = time.perf_counter() - phase_started

        pdf: Any | None = None
        try:
            phase_started = time.perf_counter()
            pdf = pdfium.PdfDocument(str(path))
            record_phase("open_document", phase_started)

            page_count = len(pdf)
            metadata = self._metadata(pdf)
            title = metadata.get("Title") or metadata.get("title") or path.stem

            phase_started = time.perf_counter()
            outline = self._outline(pdf, page_count)
            record_phase("outline", phase_started)

            phase_started = time.perf_counter()
            pages = self._pages(pdf, page_count)
            record_phase("page_text", phase_started)
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_FAST_PARSE_FAILED,
                f"pypdfium2 fast parse failed for {pdf_path}",
            ) from exc
        finally:
            if pdf is not None:
                close_pdf = getattr(pdf, "close", None)
                if callable(close_pdf):
                    close_pdf()

        phase_started = time.perf_counter()
        chunks = self._chunks(doc_id=doc_id, outline=outline, pages=pages)
        record_phase("chunk_assembly", phase_started)
        phase_seconds["total"] = time.perf_counter() - started

        reading_markdown = self._reading_markdown(chunks)
        return ParsedDocument(
            title=str(title),
            parser_chain=[self.method],
            metadata={
                "pages": page_count,
                "toc_nodes_raw": len(outline),
                "toc_nodes_clean": len(outline),
                "formula_markers_total": 0,
                "formula_replaced_by_pix2text": 0,
                "formula_replaced_by_fallback": 0,
                "formula_unresolved": 0,
                "formula_records_total": 0,
                "fast_parser": True,
                "fidelity_lane": "preview_text_only",
                "pdf_parse_phase_seconds": {
                    key: round(value, 6) for key, value in phase_seconds.items()
                },
                "pypdfium2_metadata": metadata,
            },
            outline=outline,
            chunks=chunks,
            formulas=[],
            images=[],
            reading_markdown=reading_markdown,
            raw_artifacts={},
            overall_confidence=None,
        )

    def _metadata(self, pdf: Any) -> dict[str, str]:
        get_metadata_dict = getattr(pdf, "get_metadata_dict", None)
        if not callable(get_metadata_dict):
            return {}
        raw = get_metadata_dict() or {}
        return {str(key): str(value) for key, value in dict(raw).items() if value}

    def _outline(self, pdf: Any, page_count: int) -> list[OutlineNode]:
        get_toc = getattr(pdf, "get_toc", None)
        if not callable(get_toc):
            return []

        nodes: list[OutlineNode] = []
        for idx, bookmark in enumerate(list(get_toc() or [])):
            title_getter = getattr(bookmark, "get_title", None)
            title = str(title_getter() if callable(title_getter) else "").strip()
            if not title:
                title = f"section-{idx + 1}"
            level = int(getattr(bookmark, "level", 0) or 0) + 1
            page_start: int | None = None
            dest_getter = getattr(bookmark, "get_dest", None)
            dest = dest_getter() if callable(dest_getter) else None
            index_getter = getattr(dest, "get_index", None)
            if callable(index_getter):
                try:
                    raw_index = int(index_getter())
                    if 0 <= raw_index < page_count:
                        page_start = raw_index + 1
                except Exception:  # noqa: BLE001
                    page_start = None
            nodes.append(
                OutlineNode(
                    id=f"toc-{idx + 1}",
                    title=title,
                    level=level,
                    page_start=page_start,
                    page_end=None,
                )
            )

        for idx, node in enumerate(nodes):
            node.page_end = _outline_page_end(nodes, idx, page_count)
        return nodes

    def _pages(self, pdf: Any, page_count: int) -> list[_PageText]:
        pages: list[_PageText] = []
        for page_index in range(page_count):
            page = pdf[page_index]
            textpage = None
            try:
                textpage = page.get_textpage()
                text = str(textpage.get_text_range() or "")
            finally:
                close_textpage = getattr(textpage, "close", None)
                if callable(close_textpage):
                    close_textpage()
                close_page = getattr(page, "close", None)
                if callable(close_page):
                    close_page()
            pages.append(_PageText(page=page_index + 1, text=text))
        return pages

    def _chunks(
        self, *, doc_id: str, outline: list[OutlineNode], pages: list[_PageText]
    ) -> list[ChunkRecord]:
        return self._page_chunks(doc_id=doc_id, outline=outline, pages=pages)

    def _page_chunks(
        self, *, doc_id: str, outline: list[OutlineNode], pages: list[_PageText]
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for page in pages:
            if not page.text.strip():
                continue
            chunks.append(
                self._chunk(
                    doc_id=doc_id,
                    order_index=len(chunks),
                    section_path=self._section_path_for_page(outline, page.page),
                    page_range=[page.page, page.page],
                    text=page.text,
                )
            )
        return chunks

    def _section_path_for_page(
        self, outline: list[OutlineNode], page: int
    ) -> list[str]:
        if not outline:
            return [f"Page {page}"]
        best_index: int | None = None
        best_level = -1
        for idx, node in enumerate(outline):
            if node.page_start is None:
                continue
            page_end = node.page_end or node.page_start
            if node.page_start <= page <= page_end and node.level >= best_level:
                best_index = idx
                best_level = node.level
        if best_index is None:
            return [f"Page {page}"]
        return _build_section_path(outline, best_index)

    def _chunk(
        self,
        *,
        doc_id: str,
        order_index: int,
        section_path: list[str],
        page_range: list[int],
        text: str,
    ) -> ChunkRecord:
        chunk_id = hashlib.sha1(
            f"{doc_id}:{order_index}:{section_path}:{page_range}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:16]
        return ChunkRecord(
            chunk_id=chunk_id,
            doc_id=doc_id,
            order_index=order_index,
            section_path=section_path,
            text=text.strip(),
            search_text=_sanitize_text(text),
            locator=Locator(
                doc_id=doc_id,
                chunk_id=chunk_id,
                section_path=section_path,
                page_range=page_range,
                method=self.method,
                confidence=None,
            ),
            method=self.method,
            confidence=None,
        )

    def _reading_markdown(self, chunks: list[ChunkRecord]) -> str:
        parts: list[str] = []
        for chunk in chunks:
            title = " > ".join(chunk.section_path) or f"Chunk {chunk.order_index + 1}"
            parts.append(f"## {title}\n\n{chunk.text.strip()}")
        return "\n\n".join(parts).strip()
