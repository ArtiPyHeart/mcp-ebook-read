"""PDF parser using Docling for structure and Pix2Text for formula recovery."""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fitz
from PIL import Image

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    FormulaRecord,
    Locator,
    OutlineNode,
    ParsedDocument,
)


_FORMULA_MARKER = "<!-- formula-not-decoded -->"
_MAX_HEADING_LENGTH = 160
_CODE_LIKE_TOKENS = ("def ", "class ", "return ", "import ", "while ", "for ")
_FORMULA_SOURCE = "pix2text"
_UNRESOLVED_FORMULA = "[Formula unresolved. Use render_pdf_page for verification.]"


@dataclass(slots=True)
class _SectionBlock:
    path: list[str]
    title: str
    level: int
    text: str


@dataclass(slots=True)
class _FormulaCandidate:
    latex: str
    page: int
    score: float | None
    bbox: list[float] | None
    source: str = _FORMULA_SOURCE


@dataclass(slots=True)
class _FormulaReplacementStats:
    markers_total: int = 0
    replaced_by_engine: int = 0
    replaced_by_fallback: int = 0
    unresolved: int = 0


@dataclass(slots=True)
class _ResolvedFormula:
    latex: str
    page: int | None
    bbox: list[float] | None
    source: str
    confidence: float | None
    status: str


def _sanitize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = _sanitize_text(normalized).lower()
    lowered = re.sub(r"^(chapter|section|part)\s+\d+[:.\-]?\s*", "", lowered)
    lowered = re.sub(r"^\d+(?:[.\-]\d+)*\s*", "", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return _sanitize_text(lowered)


def _is_noisy_heading(title: str) -> bool:
    if not title:
        return True
    if len(title) > _MAX_HEADING_LENGTH:
        return True
    lowered = title.lower()
    if "```" in title:
        return True
    code_hits = sum(token in lowered for token in _CODE_LIKE_TOKENS)
    if code_hits >= 2:
        return True
    if title.count(":") >= 4 and len(title.split()) >= 18:
        return True
    return False


def _sanitize_heading(raw: str, *, fallback_idx: int | None = None) -> str:
    cleaned = _sanitize_text(raw.strip("#").strip())
    if not cleaned:
        if fallback_idx is None:
            return ""
        return f"Section {fallback_idx}"
    if _is_noisy_heading(cleaned):
        if fallback_idx is None:
            return ""
        return f"Section {fallback_idx}"
    return cleaned


def _split_markdown_into_sections(markdown: str) -> list[_SectionBlock]:
    sections: list[_SectionBlock] = []
    heading_stack: list[str] = []
    current_path: list[str] = ["Document"]
    current_level = 1
    buffer: list[str] = []
    fallback_idx = 1

    def flush_section() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            sections.append(
                _SectionBlock(
                    path=current_path[:],
                    title=current_path[-1],
                    level=current_level,
                    text=text,
                )
            )
        buffer = []

    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            marker = stripped[:level]
            if marker and set(marker) == {"#"}:
                flush_section()
                title = _sanitize_heading(
                    stripped[level:].strip(), fallback_idx=fallback_idx
                )
                fallback_idx += 1
                while len(heading_stack) >= max(level, 1):
                    heading_stack.pop()
                heading_stack.append(title)
                current_path = heading_stack[:] if heading_stack else ["Document"]
                current_level = max(1, min(level, 6))
                continue
        buffer.append(line)

    flush_section()
    if not sections:
        text = markdown.strip()
        if text:
            sections.append(
                _SectionBlock(path=["Document"], title="Document", level=1, text=text)
            )

    return sections


def _sections_to_markdown(sections: list[_SectionBlock]) -> str:
    blocks: list[str] = []
    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        if section.title == "Document" and section.path == ["Document"]:
            blocks.append(text)
            continue
        heading = "#" * max(1, min(section.level, 6))
        blocks.append(f"{heading} {section.title}\n\n{text}")
    return "\n\n".join(blocks).strip()


def _build_outline_from_toc(
    toc: list[list[object]], page_count: int
) -> list[OutlineNode]:
    outline: list[OutlineNode] = []
    for item in toc:
        if len(item) < 3:
            continue
        level_raw, heading_raw, page_raw = item[:3]
        try:
            level = int(level_raw)
            page = int(page_raw)
        except (TypeError, ValueError):
            continue
        heading = _sanitize_heading(str(heading_raw), fallback_idx=None)
        if not heading:
            continue
        level = max(1, min(level, 6))
        page = max(1, min(page, page_count))
        if outline and level > outline[-1].level + 1:
            level = outline[-1].level + 1
        if (
            outline
            and outline[-1].title == heading
            and outline[-1].level == level
            and outline[-1].page_start == page
        ):
            continue
        outline.append(
            OutlineNode(
                id=f"toc-{len(outline)}",
                title=heading,
                level=level,
                page_start=page,
                page_end=page,
            )
        )

    for idx, node in enumerate(outline):
        node_start = node.page_start or 1
        boundary_page = page_count + 1
        for sibling in outline[idx + 1 :]:
            sibling_start = sibling.page_start or node_start
            if sibling.level <= node.level and sibling_start >= node_start:
                boundary_page = sibling_start
                break
        node.page_end = max(node_start, min(page_count, boundary_page - 1))

    return outline


def _build_outline_from_sections(
    sections: list[_SectionBlock], page_ranges: list[list[int]]
) -> list[OutlineNode]:
    outline: list[OutlineNode] = []
    for idx, (section, page_range) in enumerate(
        zip(sections, page_ranges, strict=True)
    ):
        outline.append(
            OutlineNode(
                id=f"sec-{idx}",
                title=section.title,
                level=max(1, min(section.level, 6)),
                page_start=page_range[0],
                page_end=page_range[1],
            )
        )
    return outline


def _build_toc_page_index(outline: list[OutlineNode]) -> dict[str, list[int]]:
    pages: dict[str, list[int]] = {}
    for node in outline:
        key = _normalize_key(node.title)
        if not key:
            continue
        start = node.page_start or 1
        pages.setdefault(key, []).append(start)
    return pages


def _assign_section_page_ranges(
    sections: list[_SectionBlock],
    page_count: int,
    pages_by_title: dict[str, list[int]],
) -> list[list[int]]:
    start_pages: list[int] = []
    title_offsets: dict[str, int] = {}
    cursor = 1
    for section in sections:
        key = _normalize_key(section.title)
        chosen = None
        if key and key in pages_by_title:
            pages = pages_by_title[key]
            offset = title_offsets.get(key, 0)
            if offset < len(pages):
                chosen = pages[offset]
                title_offsets[key] = offset + 1

        if chosen is None:
            chosen = cursor

        chosen = max(1, min(page_count, chosen))
        if chosen < cursor:
            chosen = cursor
        start_pages.append(chosen)
        cursor = chosen

    ranges: list[list[int]] = []
    for idx, start in enumerate(start_pages):
        if idx + 1 < len(start_pages):
            next_start = max(start, start_pages[idx + 1])
            end = max(start, next_start - 1)
        else:
            end = page_count
        ranges.append([start, min(page_count, end)])

    return ranges


def _extract_formula_candidates_from_text(
    page_text: str, *, limit: int = 3
) -> list[str]:
    candidates: list[str] = []
    for raw_line in page_text.splitlines():
        line = _sanitize_text(raw_line)
        if not line or len(line) > 180:
            continue
        math_chars = sum(ch in line for ch in ("=", "≤", "≥", "∈", "⊂", "⊆", "→", "^"))
        looks_math = bool(re.search(r"[A-Za-z]\s*=\s*|[0-9]\s*[+\-/*]\s*[0-9]", line))
        if math_chars == 0 and not looks_math:
            continue
        if line in candidates:
            continue
        candidates.append(line)
        if len(candidates) >= limit:
            break
    return candidates


def _looks_like_latex(text: str) -> bool:
    tokens = (
        "\\frac",
        "\\sum",
        "\\int",
        "\\alpha",
        "\\beta",
        "\\gamma",
        "_{",
        "^{",
        "\\left",
        "\\right",
    )
    if any(token in text for token in tokens):
        return True
    return bool(re.search(r"[A-Za-z0-9]\s*=\s*[A-Za-z0-9]", text))


def _coerce_bbox(position: Any) -> list[float] | None:
    if not isinstance(position, list) or len(position) != 4:
        return None
    points: list[tuple[float, float]] = []
    for point in position:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return [min(xs), min(ys), max(xs), max(ys)]


class _Pix2TextFormulaExtractor:
    def __init__(self, *, batch_size: int = 1) -> None:
        self.batch_size = max(1, batch_size)
        self._engine: Any | None = None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from pix2text import Pix2Text  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_FORMULA_ENGINE_UNAVAILABLE,
                "Pix2Text is required for formula reconstruction but is not available.",
                details={
                    "engine": _FORMULA_SOURCE,
                    "install_hint": "Install dependency: uv add pix2text",
                },
            ) from exc
        self._engine = Pix2Text.from_config()
        return self._engine

    def _normalize_output(self, raw: Any) -> list[dict[str, Any]]:
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            for key in ("outs", "results", "elements", "blocks", "res"):
                value = raw.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        for attr in ("outs", "results", "elements", "blocks"):
            value = getattr(raw, attr, None)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    def extract(self, image: Image.Image, page: int) -> list[_FormulaCandidate]:
        engine = self._ensure_engine()
        try:
            if hasattr(engine, "recognize_text_formula"):
                raw = engine.recognize_text_formula(
                    image,
                    return_text=False,
                    mfr_batch_size=self.batch_size,
                )
            elif hasattr(engine, "recognize"):
                raw = engine.recognize(
                    image,
                    file_type="text_formula",
                    return_text=False,
                    mfr_batch_size=self.batch_size,
                )
            else:
                raise AppError(
                    ErrorCode.INGEST_PDF_FORMULA_ENGINE_UNAVAILABLE,
                    "Pix2Text API mismatch: no compatible recognize method found.",
                    details={"engine": _FORMULA_SOURCE},
                )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_FORMULA_ENGINE_UNAVAILABLE,
                "Pix2Text failed while recognizing formulas.",
                details={"engine": _FORMULA_SOURCE, "page": page},
            ) from exc

        candidates: list[_FormulaCandidate] = []
        for item in self._normalize_output(raw):
            text = _sanitize_text(str(item.get("text") or ""))
            if not text:
                continue
            kind = str(item.get("type") or "").lower()
            if kind not in {
                "formula",
                "isolated",
                "embedding",
            } and not _looks_like_latex(text):
                continue

            score_raw = item.get("score")
            score = None
            if isinstance(score_raw, (int, float)):
                score = float(score_raw)

            candidates.append(
                _FormulaCandidate(
                    latex=text,
                    page=page,
                    score=score,
                    bbox=_coerce_bbox(item.get("position")),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.bbox[1] if candidate.bbox else 0.0,
                candidate.bbox[0] if candidate.bbox else 0.0,
            )
        )
        return candidates


class DoclingPdfParser:
    """Extract PDF content and structure from Docling output."""

    method = "docling"

    def __init__(
        self,
        *,
        enable_docling_formula_enrichment: bool = True,
        require_formula_engine: bool = True,
        formula_batch_size: int = 1,
    ) -> None:
        self.enable_docling_formula_enrichment = enable_docling_formula_enrichment
        self.require_formula_engine = require_formula_engine
        self.formula_extractor = _Pix2TextFormulaExtractor(
            batch_size=formula_batch_size
        )

    def _build_docling_converter(self) -> Any:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_FAILED,
                "Docling formula pipeline dependencies are unavailable.",
            ) from exc

        options = PdfPipelineOptions()
        options.do_formula_enrichment = self.enable_docling_formula_enrichment
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )

    def _render_page_image(self, pdf_doc: fitz.Document, page: int) -> Image.Image:
        page_obj = pdf_doc.load_page(page - 1)
        pixmap = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()
        return image

    def _format_formula_candidate(self, candidate: _FormulaCandidate) -> str:
        score = "n/a" if candidate.score is None else f"{candidate.score:.3f}"
        return (
            "$$\n"
            f"{candidate.latex}\n"
            "$$\n"
            f"<!-- formula-source: {candidate.source}; page={candidate.page}; score={score} -->"
        )

    def _pop_formula_candidate(
        self,
        *,
        pdf_doc: fitz.Document,
        page_range: list[int],
        formula_cache: dict[int, list[_FormulaCandidate]],
        formula_offsets: dict[int, int],
    ) -> _FormulaCandidate | None:
        start, end = page_range
        for page in range(start, end + 1):
            if page not in formula_cache:
                image = self._render_page_image(pdf_doc, page)
                formula_cache[page] = self.formula_extractor.extract(image, page)
            candidates = formula_cache[page]
            offset = formula_offsets.get(page, 0)
            if offset < len(candidates):
                formula_offsets[page] = offset + 1
                return candidates[offset]
        return None

    def _replace_formula_markers(
        self,
        *,
        text: str,
        page_range: list[int],
        get_page_text: Callable[[int], str],
        pdf_doc: fitz.Document,
        formula_cache: dict[int, list[_FormulaCandidate]],
        formula_offsets: dict[int, int],
    ) -> tuple[str, _FormulaReplacementStats, list[_ResolvedFormula]]:
        stats = _FormulaReplacementStats(markers_total=text.count(_FORMULA_MARKER))
        if stats.markers_total == 0:
            return text, stats, []

        replaced = text
        formulas: list[_ResolvedFormula] = []
        for _ in range(stats.markers_total):
            candidate = self._pop_formula_candidate(
                pdf_doc=pdf_doc,
                page_range=page_range,
                formula_cache=formula_cache,
                formula_offsets=formula_offsets,
            )
            if candidate is not None:
                replacement = self._format_formula_candidate(candidate)
                stats.replaced_by_engine += 1
                formulas.append(
                    _ResolvedFormula(
                        latex=candidate.latex,
                        page=candidate.page,
                        bbox=candidate.bbox,
                        source=candidate.source,
                        confidence=candidate.score,
                        status="resolved",
                    )
                )
            else:
                start, end = page_range
                fallback_lines: list[str] = []
                fallback_page: int | None = None
                for page in range(start, end + 1):
                    page_text = get_page_text(page)
                    if not page_text:
                        continue
                    fallback_lines.extend(
                        _extract_formula_candidates_from_text(page_text, limit=2)
                    )
                    if fallback_lines:
                        fallback_page = page
                        break
                if fallback_lines:
                    replacement = "\n".join(
                        f"- Formula fallback text: {line}"
                        for line in fallback_lines[:2]
                    )
                    stats.replaced_by_fallback += 1
                    formulas.append(
                        _ResolvedFormula(
                            latex="\n".join(fallback_lines[:2]),
                            page=fallback_page,
                            bbox=None,
                            source="page_text_fallback",
                            confidence=None,
                            status="fallback_text",
                        )
                    )
                else:
                    replacement = _UNRESOLVED_FORMULA
                    stats.unresolved += 1
                    formulas.append(
                        _ResolvedFormula(
                            latex=_UNRESOLVED_FORMULA,
                            page=start,
                            bbox=None,
                            source="unresolved",
                            confidence=None,
                            status="unresolved",
                        )
                    )

            replaced = replaced.replace(_FORMULA_MARKER, replacement, 1)

        return replaced, stats, formulas

    def parse(self, pdf_path: str, doc_id: str) -> ParsedDocument:
        path = Path(pdf_path)
        if not path.exists():
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                f"Document not found: {pdf_path}",
            )

        try:
            converter = self._build_docling_converter()
            result = converter.convert(str(path))
            document = result.document
            markdown = document.export_to_markdown()
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_FAILED,
                f"Docling parse failed for {pdf_path}",
            ) from exc

        sections = _split_markdown_into_sections(markdown)
        formula_markers_total = sum(
            section.text.count(_FORMULA_MARKER) for section in sections
        )
        if formula_markers_total > 0 and self.require_formula_engine:
            self.formula_extractor._ensure_engine()

        with fitz.open(str(path)) as pdf_doc:
            title = pdf_doc.metadata.get("title") or path.stem
            page_count = pdf_doc.page_count
            toc = pdf_doc.get_toc()
            page_text_cache: dict[int, str] = {}
            iter_pages_cache: list[Any] | None = None

            def get_page_text(page: int) -> str:
                nonlocal iter_pages_cache
                if page < 1 or page > page_count:
                    return ""
                cached = page_text_cache.get(page)
                if cached is not None:
                    return cached

                text = ""
                try:
                    page_obj = pdf_doc.load_page(page - 1)
                    get_text = getattr(page_obj, "get_text", None)
                    if callable(get_text):
                        text = str(get_text("text"))
                except Exception:  # noqa: BLE001
                    text = ""

                if not text:
                    if iter_pages_cache is None:
                        try:
                            iter_pages_cache = list(pdf_doc)
                        except Exception:  # noqa: BLE001
                            iter_pages_cache = []
                    index = page - 1
                    if index < len(iter_pages_cache):
                        get_text = getattr(iter_pages_cache[index], "get_text", None)
                        if callable(get_text):
                            try:
                                text = str(get_text("text"))
                            except Exception:  # noqa: BLE001
                                text = ""

                page_text_cache[page] = text
                return text

            outline = _build_outline_from_toc(toc, page_count)
            pages_by_title = _build_toc_page_index(outline)
            page_ranges = _assign_section_page_ranges(
                sections=sections,
                page_count=page_count,
                pages_by_title=pages_by_title,
            )

            formula_cache: dict[int, list[_FormulaCandidate]] = {}
            formula_offsets: dict[int, int] = {}
            formula_stats = _FormulaReplacementStats()
            enhanced_sections: list[_SectionBlock] = []
            section_formulas: list[list[_ResolvedFormula]] = []

            for section, page_range in zip(sections, page_ranges, strict=True):
                enhanced_text, section_stats, recovered_formulas = (
                    self._replace_formula_markers(
                        text=section.text,
                        page_range=page_range,
                        get_page_text=get_page_text,
                        pdf_doc=pdf_doc,
                        formula_cache=formula_cache,
                        formula_offsets=formula_offsets,
                    )
                )
                formula_stats.markers_total += section_stats.markers_total
                formula_stats.replaced_by_engine += section_stats.replaced_by_engine
                formula_stats.replaced_by_fallback += section_stats.replaced_by_fallback
                formula_stats.unresolved += section_stats.unresolved
                section_formulas.append(recovered_formulas)
                enhanced_sections.append(
                    _SectionBlock(
                        path=section.path,
                        title=section.title,
                        level=section.level,
                        text=enhanced_text,
                    )
                )

        chunks: list[ChunkRecord] = []
        chunk_ids_by_section_index: dict[int, str] = {}
        for idx, (section, page_range) in enumerate(
            zip(enhanced_sections, page_ranges, strict=True)
        ):
            normalized = _sanitize_text(section.text)
            if not normalized:
                continue
            chunk_id = hashlib.sha1(
                f"{doc_id}:{idx}:{section.path}".encode(), usedforsecurity=False
            ).hexdigest()[:16]
            locator = Locator(
                doc_id=doc_id,
                chunk_id=chunk_id,
                section_path=section.path,
                page_range=page_range,
                method=self.method,
                confidence=None,
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    order_index=len(chunks),
                    section_path=section.path,
                    text=section.text.strip(),
                    search_text=normalized,
                    locator=locator,
                    method=self.method,
                    confidence=None,
                )
            )
            chunk_ids_by_section_index[idx] = chunk_id

        if not outline:
            outline = _build_outline_from_sections(enhanced_sections, page_ranges)

        formulas: list[FormulaRecord] = []
        for section_idx, (section, page_range, recovered_formulas) in enumerate(
            zip(enhanced_sections, page_ranges, section_formulas, strict=True)
        ):
            chunk_id = chunk_ids_by_section_index.get(section_idx)
            for formula_idx, recovered in enumerate(recovered_formulas):
                page = recovered.page if recovered.page is not None else page_range[0]
                formula_identity = (
                    f"{doc_id}:{section_idx}:{formula_idx}:{page}:"
                    f"{recovered.bbox}:{recovered.status}:{recovered.latex}"
                )
                formula_id = hashlib.sha1(
                    formula_identity.encode(), usedforsecurity=False
                ).hexdigest()[:16]
                formulas.append(
                    FormulaRecord(
                        formula_id=formula_id,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        section_path=section.path,
                        page=page,
                        bbox=recovered.bbox,
                        latex=recovered.latex,
                        source=recovered.source,
                        confidence=recovered.confidence,
                        status=recovered.status,
                    )
                )

        parser_chain = [self.method]
        if formula_stats.markers_total > 0:
            parser_chain.append(_FORMULA_SOURCE)

        return ParsedDocument(
            title=title,
            parser_chain=parser_chain,
            metadata={
                "pages": page_count,
                "toc_nodes_raw": len(toc),
                "toc_nodes_clean": len(outline),
                "formula_markers_total": formula_stats.markers_total,
                "formula_replaced_by_pix2text": formula_stats.replaced_by_engine,
                "formula_replaced_by_fallback": formula_stats.replaced_by_fallback,
                "formula_unresolved": formula_stats.unresolved,
                "formula_records_total": len(formulas),
                "formula_engine": _FORMULA_SOURCE
                if formula_stats.markers_total
                else None,
                "docling_formula_enrichment_enabled": self.enable_docling_formula_enrichment,
            },
            outline=outline,
            chunks=chunks,
            formulas=formulas,
            reading_markdown=_sections_to_markdown(enhanced_sections),
            raw_artifacts={},
            overall_confidence=None,
        )
