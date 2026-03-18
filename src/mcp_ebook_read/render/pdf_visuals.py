"""Docling-based PDF table and figure extraction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import html as lxml_html
from PIL import Image

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.parsers.pdf_docling import _configure_docling_runtime_logging
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    PdfFigureRecord,
    PdfParserPerformanceConfig,
    PdfTableRecord,
    TableSegmentRecord,
)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _stringify_cell(value: Any) -> str:
    if value is None:
        return ""
    text = _normalize_text(str(value))
    return "" if text.lower() == "nan" else text


def _normalize_headers(headers: list[str]) -> list[str]:
    return [_normalize_text(header).casefold() for header in headers]


_TABLE_CAPTION_PATTERN = re.compile(
    r"^(table|tab\.?)(?:\s+|\s*[-.:])\d+", re.IGNORECASE
)
_FIGURE_CAPTION_PATTERN = re.compile(
    r"^(figure|fig\.?|chart)(?:\s+|\s*[-.:])\d+",
    re.IGNORECASE,
)


def _render_markdown_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    caption: str | None = None,
) -> str:
    column_count = max(len(headers), max((len(row) for row in rows), default=0))
    if column_count == 0:
        return caption or ""

    effective_headers = headers or [f"column_{idx + 1}" for idx in range(column_count)]
    if len(effective_headers) < column_count:
        effective_headers.extend(
            f"column_{idx + 1}" for idx in range(len(effective_headers), column_count)
        )

    lines: list[str] = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    lines.append("| " + " | ".join(effective_headers) + " |")
    lines.append("| " + " | ".join(["---"] * column_count) + " |")
    for row in rows:
        padded = row + [""] * max(0, column_count - len(row))
        lines.append("| " + " | ".join(padded[:column_count]) + " |")
    return "\n".join(lines).strip()


def _render_html_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    caption: str | None = None,
) -> str:
    column_count = max(len(headers), max((len(row) for row in rows), default=0))
    effective_headers = headers or [f"column_{idx + 1}" for idx in range(column_count)]
    if len(effective_headers) < column_count:
        effective_headers.extend(
            f"column_{idx + 1}" for idx in range(len(effective_headers), column_count)
        )

    root = lxml_html.Element("table")
    if caption:
        caption_node = lxml_html.Element("caption")
        caption_node.text = caption
        root.append(caption_node)

    if effective_headers:
        thead = lxml_html.Element("thead")
        header_row = lxml_html.Element("tr")
        for header in effective_headers:
            cell = lxml_html.Element("th")
            cell.text = header
            header_row.append(cell)
        thead.append(header_row)
        root.append(thead)

    tbody = lxml_html.Element("tbody")
    for row in rows:
        row_node = lxml_html.Element("tr")
        padded = row + [""] * max(0, column_count - len(row))
        for value in padded[:column_count]:
            cell = lxml_html.Element("td")
            cell.text = value
            row_node.append(cell)
        tbody.append(row_node)
    root.append(tbody)
    return lxml_html.tostring(root, encoding="unicode")


def _extract_table_rows_from_html(html_value: str) -> tuple[list[str], list[list[str]]]:
    if not html_value.strip():
        return [], []

    try:
        root = lxml_html.fragment_fromstring(html_value, create_parent="div")
    except Exception:  # noqa: BLE001
        return [], []

    table = root.find(".//table")
    if table is None:
        return [], []

    header_cells = table.xpath("./thead/tr[1]/th|./thead/tr[1]/td")
    headers = [_normalize_text(cell.text_content()) for cell in header_cells]

    rows: list[list[str]] = []
    body_rows = table.xpath("./tbody/tr") or table.xpath("./tr[position()>1]")
    if not headers:
        header_candidate = table.xpath("./tr[1]/th|./tr[1]/td")
        headers = [_normalize_text(cell.text_content()) for cell in header_candidate]

    for row_node in body_rows:
        row = [
            _normalize_text(cell.text_content()) for cell in row_node.xpath("./th|./td")
        ]
        if row:
            rows.append(row)

    return headers, rows


def _common_section_path(paths: list[list[str]]) -> list[str]:
    non_empty = [path for path in paths if path]
    if not non_empty:
        return []

    prefix = non_empty[0][:]
    for path in non_empty[1:]:
        limit = min(len(prefix), len(path))
        idx = 0
        while idx < limit and prefix[idx] == path[idx]:
            idx += 1
        prefix = prefix[:idx]
        if not prefix:
            break
    return prefix


def _kind_for_caption_pattern(kind: str) -> str:
    lowered = kind.strip().lower()
    if lowered == "table":
        return "table"
    if lowered in {"chart", "figure", "picture"}:
        return "figure"
    return "generic"


def _caption_pattern_matches(kind: str, text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    pattern_kind = _kind_for_caption_pattern(kind)
    if pattern_kind == "table":
        return bool(_TABLE_CAPTION_PATTERN.match(normalized))
    if pattern_kind == "figure":
        return bool(_FIGURE_CAPTION_PATTERN.match(normalized))
    return bool(
        _TABLE_CAPTION_PATTERN.match(normalized)
        or _FIGURE_CAPTION_PATTERN.match(normalized)
    )


def _caption_candidate_payload(
    *,
    text: str,
    source: str,
    confidence: float,
    page: int | None,
    bbox: list[float] | None,
    distance: float | None,
    matched_pattern: bool,
) -> dict[str, Any]:
    return {
        "text": text,
        "source": source,
        "confidence": round(confidence, 3),
        "page": page,
        "bbox": bbox,
        "distance": round(distance, 2) if distance is not None else None,
        "matched_pattern": matched_pattern,
    }


def _caption_resolution_payload(
    *,
    text: str | None,
    source: str | None,
    confidence: float | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "text": text,
        "source": source,
        "confidence": round(confidence, 3) if confidence is not None else None,
        "candidates": candidates,
    }


def _diagnostic(
    code: str,
    message: str,
    *,
    severity: str = "warning",
    hint: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_details = details or {}
    fingerprint = hashlib.sha1(
        json.dumps(
            {
                "code": code,
                "severity": severity,
                "details": resolved_details,
            },
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    payload = {
        "code": code,
        "fingerprint": fingerprint,
        "severity": severity,
        "message": message,
        "details": resolved_details,
    }
    if hint:
        payload["hint"] = hint
    return payload


def _severity_count(issues: list[dict[str, Any]], severity: str) -> int:
    return sum(1 for issue in issues if issue.get("severity") == severity)


@dataclass(slots=True)
class PdfVisualExtractionResult:
    tables: list[PdfTableRecord]
    figures: list[PdfFigureRecord]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class _PageTextBlock:
    ref: str | None
    text: str
    page: int | None
    bbox: list[float] | None


@dataclass(slots=True)
class _TableSegment:
    segment_index: int
    page: int | None
    page_height: float | None
    bbox: list[float] | None
    caption: str | None
    headers: list[str]
    rows: list[list[str]]
    file_path: str
    width: int | None
    height: int | None
    section_path: list[str]
    caption_source: str | None = None
    caption_confidence: float | None = None
    caption_candidates: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)


class DoclingPdfVisualExtractor:
    """Extract structured tables and figure crops from PDFs using Docling."""

    def __init__(
        self,
        *,
        performance_config: PdfParserPerformanceConfig | None = None,
        images_scale: float = 2.0,
    ) -> None:
        self.performance_config = performance_config or PdfParserPerformanceConfig()
        self.images_scale = max(1.0, images_scale)

    def _pipeline_options_for_config(self, config: PdfParserPerformanceConfig) -> Any:
        try:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_FAILED,
                "Docling visual extraction dependencies are unavailable.",
            ) from exc

        options = PdfPipelineOptions()
        options.accelerator_options = AcceleratorOptions(
            num_threads=config.num_threads,
            device=config.device,
        )
        options.ocr_batch_size = config.ocr_batch_size
        options.layout_batch_size = config.layout_batch_size
        options.table_batch_size = config.table_batch_size
        options.generate_page_images = True
        options.generate_picture_images = True
        options.images_scale = self.images_scale
        return options

    def _build_docling_converter(self) -> Any:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_FAILED,
                "Docling visual extraction dependencies are unavailable.",
            ) from exc

        _configure_docling_runtime_logging()
        options = self._pipeline_options_for_config(self.performance_config)
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )

    def _section_path_for_page(
        self, chunks: list[ChunkRecord], page: int | None
    ) -> list[str]:
        if page is None:
            return []
        for chunk in chunks:
            page_range = chunk.locator.page_range
            if not page_range or len(page_range) != 2:
                continue
            if page_range[0] <= page <= page_range[1]:
                return chunk.section_path
        return []

    def _build_text_by_ref(self, document: Any) -> dict[str, str]:
        text_by_ref: dict[str, str] = {}
        for item, _level in document.iterate_items():
            ref = getattr(item, "self_ref", None)
            text = getattr(item, "text", None)
            if not ref or not isinstance(text, str):
                continue
            normalized = _normalize_text(text)
            if normalized:
                text_by_ref[str(ref)] = normalized
        return text_by_ref

    def _build_page_text_blocks(self, document: Any) -> list[_PageTextBlock]:
        blocks: list[_PageTextBlock] = []
        for item, _level in document.iterate_items():
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                continue
            normalized = _normalize_text(text)
            if not normalized or len(normalized) > 260:
                continue

            provenance = getattr(item, "prov", None) or []
            page: int | None = None
            bbox: list[float] | None = None
            if provenance:
                prov = provenance[0]
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    page = int(page_no)
                bbox_value = getattr(prov, "bbox", None)
                if bbox_value is not None and page is not None:
                    page_item = document.pages.get(page)
                    if page_item is not None:
                        normalized_bbox = bbox_value.to_top_left_origin(
                            page_height=page_item.size.height
                        )
                        bbox = [
                            round(float(normalized_bbox.l), 2),
                            round(float(normalized_bbox.t), 2),
                            round(float(normalized_bbox.r), 2),
                            round(float(normalized_bbox.b), 2),
                        ]

            blocks.append(
                _PageTextBlock(
                    ref=str(getattr(item, "self_ref", "") or "") or None,
                    text=normalized,
                    page=page,
                    bbox=bbox,
                )
            )
        return blocks

    @staticmethod
    def _bbox_vertical_distance(
        visual_bbox: list[float] | None,
        text_bbox: list[float] | None,
    ) -> tuple[float | None, str | None]:
        if visual_bbox is None or text_bbox is None:
            return None, None
        if text_bbox[1] >= visual_bbox[3]:
            return text_bbox[1] - visual_bbox[3], "below"
        if text_bbox[3] <= visual_bbox[1]:
            return visual_bbox[1] - text_bbox[3], "above"
        return 0.0, "overlap"

    def _caption_resolution_from_refs(
        self,
        *,
        item: Any,
        kind: str,
        text_by_ref: dict[str, str],
    ) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        resolved_texts: list[str] = []
        for ref in getattr(item, "captions", []):
            cref = getattr(ref, "cref", None)
            if not cref:
                continue
            text = text_by_ref.get(str(cref))
            if not text:
                continue
            resolved_texts.append(text)
            candidates.append(
                _caption_candidate_payload(
                    text=text,
                    source="docling_caption_ref",
                    confidence=0.98,
                    page=None,
                    bbox=None,
                    distance=None,
                    matched_pattern=_caption_pattern_matches(kind, text),
                )
            )

        if not resolved_texts:
            return None
        joined = _normalize_text(" ".join(resolved_texts))
        return _caption_resolution_payload(
            text=joined or None,
            source="docling_caption_ref",
            confidence=0.98,
            candidates=candidates,
        )

    def _caption_resolution_from_page_text(
        self,
        *,
        kind: str,
        page: int | None,
        bbox: list[float] | None,
        page_text_blocks: list[_PageTextBlock],
    ) -> dict[str, Any] | None:
        if page is None:
            return None

        candidates: list[dict[str, Any]] = []
        for block in page_text_blocks:
            if block.page != page:
                continue
            text = block.text
            if not text or len(text) > 220:
                continue

            matched_pattern = _caption_pattern_matches(kind, text)
            distance, position = self._bbox_vertical_distance(bbox, block.bbox)
            score = 0.0
            if matched_pattern:
                score += 0.42
            if position == "below":
                if distance is not None and distance <= 160:
                    score += max(0.0, 0.34 - distance / 600)
            elif position == "above":
                if distance is not None and distance <= 100:
                    score += max(0.0, 0.2 - distance / 500)
            elif position == "overlap":
                score += 0.08

            if bbox is not None and block.bbox is not None:
                overlap_ratio = self._overlap_ratio(
                    bbox[0],
                    bbox[2],
                    block.bbox[0],
                    block.bbox[2],
                )
                score += overlap_ratio * 0.18
            else:
                overlap_ratio = None

            if len(text) <= 120:
                score += 0.06
            if len(text.split()) <= 18:
                score += 0.04

            if score <= 0:
                continue

            candidate = _caption_candidate_payload(
                text=text,
                source="page_text_heuristic",
                confidence=min(score, 0.92),
                page=block.page,
                bbox=block.bbox,
                distance=distance,
                matched_pattern=matched_pattern,
            )
            candidate["position"] = position
            if overlap_ratio is not None:
                candidate["horizontal_overlap_ratio"] = round(overlap_ratio, 3)
            candidates.append(candidate)

        if not candidates:
            return None

        candidates.sort(
            key=lambda candidate: (
                float(candidate.get("confidence") or 0.0),
                1 if candidate.get("matched_pattern") else 0,
                1 if candidate.get("position") == "below" else 0,
            ),
            reverse=True,
        )
        best = candidates[0]
        if float(best.get("confidence") or 0.0) < 0.5:
            return _caption_resolution_payload(
                text=None,
                source=None,
                confidence=None,
                candidates=candidates[:5],
            )

        source = (
            "page_text_pattern" if best.get("matched_pattern") else "page_text_nearby"
        )
        return _caption_resolution_payload(
            text=str(best["text"]),
            source=source,
            confidence=float(best.get("confidence") or 0.0),
            candidates=candidates[:5],
        )

    def _resolve_caption_resolution(
        self,
        *,
        item: Any,
        kind: str,
        page: int | None,
        bbox: list[float] | None,
        text_by_ref: dict[str, str],
        page_text_blocks: list[_PageTextBlock],
    ) -> dict[str, Any]:
        resolution = self._caption_resolution_from_refs(
            item=item,
            kind=kind,
            text_by_ref=text_by_ref,
        )
        if resolution is not None:
            return resolution

        heuristic = self._caption_resolution_from_page_text(
            kind=kind,
            page=page,
            bbox=bbox,
            page_text_blocks=page_text_blocks,
        )
        if heuristic is not None:
            return heuristic

        return _caption_resolution_payload(
            text=None,
            source=None,
            confidence=None,
            candidates=[],
        )

    def _caption_issues(
        self,
        *,
        entity_kind: str,
        item_id: str | None,
        page: int | None,
        resolution: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prefix = "PDF_TABLE" if entity_kind == "table" else "PDF_FIGURE"
        details_key = "table_id" if entity_kind == "table" else "figure_id"
        details = {details_key: item_id, "page": page}
        source = resolution.get("source")
        confidence = resolution.get("confidence")
        candidates = resolution.get("candidates") or []

        if resolution.get("text"):
            if source == "docling_caption_ref":
                return []
            severity = "info"
            if confidence is not None and confidence < 0.75:
                severity = "warning"
            return [
                _diagnostic(
                    f"{prefix}_CAPTION_HEURISTIC_MATCH",
                    "Caption text was recovered heuristically instead of directly from Docling caption refs.",
                    severity=severity,
                    details={
                        **details,
                        "caption_source": source,
                        "caption_confidence": confidence,
                    },
                    hint="Verify the resolved caption on the source page if downstream automation depends on exact caption wording.",
                )
            ]

        if candidates:
            candidate_preview = [candidate.get("text") for candidate in candidates[:3]]
            return [
                _diagnostic(
                    f"{prefix}_CAPTION_AMBIGUOUS",
                    "No caption could be resolved confidently, but nearby text candidates were found.",
                    details={
                        **details,
                        "candidate_preview": candidate_preview,
                    },
                    hint="Inspect the candidate texts and the source page manually; keep the sample if this ambiguity blocks the reading workflow.",
                )
            ]

        return [
            _diagnostic(
                f"{prefix}_CAPTION_MISSING",
                f"A {entity_kind} was extracted without caption text.",
                details=details,
                hint="Inspect nearby chunk text; if identification stays ambiguous, escalate this sample to a human user.",
            )
        ]

    def _resolve_page_bbox(
        self, item: Any, document: Any
    ) -> tuple[int | None, list[float] | None, float | None]:
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            return None, None, None

        prov = provenance[0]
        page_no = getattr(prov, "page_no", None)
        if page_no is None:
            return None, None, None

        page = document.pages.get(page_no)
        if page is None:
            return int(page_no), None, None

        bbox_value = getattr(prov, "bbox", None)
        if bbox_value is None:
            return int(page_no), None, float(page.size.height)

        bbox = bbox_value.to_top_left_origin(page_height=page.size.height)
        return (
            int(page_no),
            [
                round(float(bbox.l), 2),
                round(float(bbox.t), 2),
                round(float(bbox.r), 2),
                round(float(bbox.b), 2),
            ],
            float(page.size.height),
        )

    def _extract_table_rows(
        self, item: Any, document: Any, html_value: str
    ) -> tuple[list[str], list[list[str]], bool]:
        try:
            frame = item.export_to_dataframe(doc=document)
            headers = [_stringify_cell(value) for value in list(frame.columns)]
            if hasattr(frame, "fillna"):
                values = frame.fillna("").values.tolist()
            else:
                values = frame.values.tolist()
            rows = [[_stringify_cell(cell) for cell in row] for row in values]
            return headers, rows, False
        except Exception:  # noqa: BLE001
            headers, rows = _extract_table_rows_from_html(html_value)
            return headers, rows, True

    def _table_segment_path(
        self, out_dir: Path, *, order_index: int, page: int | None
    ) -> Path:
        page_no = page or 0
        return out_dir / f"table_{order_index:04d}_p{page_no:04d}.png"

    def _figure_path(
        self, out_dir: Path, *, order_index: int, page: int | None
    ) -> Path:
        page_no = page or 0
        return out_dir / f"figure_{order_index:04d}_p{page_no:04d}.png"

    def _extract_table_segments(
        self,
        *,
        document: Any,
        chunks: list[ChunkRecord],
        out_dir: Path,
        text_by_ref: dict[str, str],
        page_text_blocks: list[_PageTextBlock],
    ) -> tuple[list[_TableSegment], list[dict[str, Any]]]:
        segments: list[_TableSegment] = []
        issues: list[dict[str, Any]] = []
        for order_index, item in enumerate(getattr(document, "tables", [])):
            image = item.get_image(document)
            if image is None:
                issues.append(
                    _diagnostic(
                        "PDF_TABLE_IMAGE_UNAVAILABLE",
                        "Docling detected a table item but did not provide an image crop.",
                        details={"segment_index": order_index},
                        hint="Inspect the source PDF manually and keep this sample for parser tuning if the missing crop blocks reading.",
                    )
                )
                continue

            page, bbox, page_height = self._resolve_page_bbox(item, document)
            path = self._table_segment_path(out_dir, order_index=order_index, page=page)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            html_value = item.export_to_html(doc=document, add_caption=False)
            headers, rows, used_html_fallback = self._extract_table_rows(
                item, document, html_value
            )
            caption_resolution = self._resolve_caption_resolution(
                item=item,
                kind="table",
                page=page,
                bbox=bbox,
                text_by_ref=text_by_ref,
                page_text_blocks=page_text_blocks,
            )
            segment_issues: list[dict[str, Any]] = []
            if used_html_fallback:
                segment_issues.append(
                    _diagnostic(
                        "PDF_TABLE_DATAFRAME_FALLBACK_HTML",
                        "Table dataframe export failed, so the extractor fell back to HTML parsing.",
                        details={"segment_index": order_index, "page": page},
                        hint="If the parsed cells look wrong, report this PDF sample for table export tuning.",
                    )
                )
            if bbox is None:
                segment_issues.append(
                    _diagnostic(
                        "PDF_TABLE_BBOX_MISSING",
                        "A table was extracted without page bounding-box provenance.",
                        details={"segment_index": order_index, "page": page},
                        hint="Use the saved table image and nearby text context for inspection; missing bbox should be treated as a parser degradation.",
                    )
                )
            segment_issues.extend(
                self._caption_issues(
                    entity_kind="table",
                    item_id=None,
                    page=page,
                    resolution=caption_resolution,
                )
            )
            if not caption_resolution.get("text"):
                for issue in segment_issues:
                    details = dict(issue.get("details") or {})
                    details.setdefault("segment_index", order_index)
                    issue["details"] = details
            else:
                for issue in segment_issues:
                    details = dict(issue.get("details") or {})
                    details.setdefault("segment_index", order_index)
                    details.setdefault(
                        "caption_source", caption_resolution.get("source")
                    )
                    details.setdefault(
                        "caption_confidence", caption_resolution.get("confidence")
                    )
                    issue["details"] = details
            if not headers:
                segment_issues.append(
                    _diagnostic(
                        "PDF_TABLE_HEADERS_MISSING",
                        "A table was extracted without structured headers.",
                        details={"segment_index": order_index, "page": page},
                        hint="Treat this table as lower-confidence structured output and prefer the saved image for manual verification.",
                    )
                )
            if not rows:
                segment_issues.append(
                    _diagnostic(
                        "PDF_TABLE_ROWS_EMPTY",
                        "A table was extracted but contains no structured rows.",
                        details={"segment_index": order_index, "page": page},
                        hint="The crop may still be useful visually; if the empty rows block task completion, report the sample PDF.",
                    )
                )
            issues.extend(segment_issues)
            segments.append(
                _TableSegment(
                    segment_index=order_index,
                    page=page,
                    page_height=page_height,
                    bbox=bbox,
                    caption=caption_resolution.get("text"),
                    headers=headers,
                    rows=rows,
                    file_path=str(path),
                    width=image.width,
                    height=image.height,
                    section_path=self._section_path_for_page(chunks, page),
                    caption_source=caption_resolution.get("source"),
                    caption_confidence=caption_resolution.get("confidence"),
                    caption_candidates=caption_resolution.get("candidates") or [],
                    issues=segment_issues,
                )
            )
        return segments, issues

    @staticmethod
    def _overlap_ratio(
        start_a: float,
        end_a: float,
        start_b: float,
        end_b: float,
    ) -> float:
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        overlap = max(0.0, overlap_end - overlap_start)
        union = max(end_a, end_b) - min(start_a, start_b)
        if union <= 0:
            return 0.0
        return overlap / union

    def _evaluate_merge_candidate(
        self, current: _TableSegment, following: _TableSegment
    ) -> dict[str, Any]:
        observation = {
            "decision": "rejected",
            "score": None,
            "reasons": [],
            "details": {
                "segment_indices": [current.segment_index, following.segment_index],
                "pages": [current.page, following.page],
                "captions": [current.caption, following.caption],
                "section_paths": [current.section_path, following.section_path],
            },
        }

        if current.page is None or following.page is None:
            observation["reasons"].append("missing_page")
            return observation
        if current.page + 1 != following.page:
            observation["reasons"].append("pages_not_adjacent")
            return observation
        if current.page_height is None or following.page_height is None:
            observation["reasons"].append("missing_page_height")
            return observation
        if current.bbox is None or following.bbox is None:
            observation["reasons"].append("missing_bbox")
            return observation
        if current.section_path and following.section_path:
            if current.section_path != following.section_path:
                observation["reasons"].append("section_path_mismatch")
                return observation

        headers_a = _normalize_headers(current.headers)
        headers_b = _normalize_headers(following.headers)
        observation["details"]["headers_a"] = current.headers
        observation["details"]["headers_b"] = following.headers
        if not headers_a:
            observation["reasons"].append("headers_missing")
            return observation
        if headers_a != headers_b:
            observation["reasons"].append("headers_mismatch")
            return observation

        bottom_ratio = current.bbox[3] / current.page_height
        top_ratio = following.bbox[1] / following.page_height
        observation["details"]["bottom_ratio"] = round(bottom_ratio, 3)
        observation["details"]["top_ratio"] = round(top_ratio, 3)
        if bottom_ratio < 0.72:
            observation["reasons"].append("current_not_near_page_bottom")
            return observation
        if top_ratio > 0.28:
            observation["reasons"].append("following_not_near_page_top")
            return observation

        overlap_ratio = self._overlap_ratio(
            current.bbox[0],
            current.bbox[2],
            following.bbox[0],
            following.bbox[2],
        )
        observation["details"]["horizontal_overlap_ratio"] = round(overlap_ratio, 3)
        if overlap_ratio < 0.7:
            observation["reasons"].append("horizontal_overlap_too_low")
            return observation

        width_a = max(1.0, current.bbox[2] - current.bbox[0])
        width_b = max(1.0, following.bbox[2] - following.bbox[0])
        width_similarity = 1.0 - abs(width_a - width_b) / max(width_a, width_b)
        observation["details"]["width_similarity"] = round(width_similarity, 3)
        if width_similarity < 0.85:
            observation["reasons"].append("width_similarity_too_low")
            return observation

        caption_a = _normalize_text(current.caption or "")
        caption_b = _normalize_text(following.caption or "")
        if caption_a and caption_b and caption_a != caption_b:
            observation["reasons"].append("caption_mismatch")
            return observation

        top_score = 1.0 - min(top_ratio / 0.28, 1.0)
        score = (
            min(bottom_ratio, 1.0) + top_score + overlap_ratio + width_similarity
        ) / 4.0
        observation["decision"] = "merged"
        observation["score"] = round(score, 3)
        observation["reasons"] = [
            "adjacent_pages",
            "matching_headers",
            "page_edge_alignment_ok",
            "horizontal_overlap_ok",
            "width_similarity_ok",
        ]
        return observation

    @staticmethod
    def _merge_images(paths: list[str], target_path: Path) -> tuple[int, int]:
        loaded: list[Image.Image] = []
        try:
            for path in paths:
                image = Image.open(path)
                image.load()
                loaded.append(image)
            max_width = max(image.width for image in loaded)
            total_height = (
                sum(image.height for image in loaded) + max(0, len(loaded) - 1) * 8
            )
            merged = Image.new("RGB", (max_width, total_height), color="white")
            offset = 0
            for image in loaded:
                if image.width != max_width:
                    padded = Image.new("RGB", (max_width, image.height), color="white")
                    padded.paste(image, ((max_width - image.width) // 2, 0))
                    merged.paste(padded, (0, offset))
                else:
                    merged.paste(image, (0, offset))
                offset += image.height + 8
            target_path.parent.mkdir(parents=True, exist_ok=True)
            merged.save(target_path, format="PNG")
            return merged.width, merged.height
        finally:
            for image in loaded:
                image.close()

    def _build_logical_table(
        self,
        *,
        doc_id: str,
        order_index: int,
        out_dir: Path,
        segments: list[_TableSegment],
        merge_scores: list[float],
    ) -> tuple[PdfTableRecord, dict[str, Any]]:
        pages = [segment.page for segment in segments if segment.page is not None]
        page_range = [min(pages), max(pages)] if pages else None
        caption_segment = next(
            (segment for segment in segments if segment.caption),
            None,
        )
        if caption_segment is None:
            caption_segment = next(
                (segment for segment in segments if segment.caption_candidates),
                None,
            )
        caption = caption_segment.caption if caption_segment is not None else None
        section_path = _common_section_path(
            [segment.section_path for segment in segments]
        )
        headers = segments[0].headers[:]
        rows: list[list[str]] = []
        for segment in segments:
            rows.extend(segment.rows)

        if len(segments) == 1:
            file_path = segments[0].file_path
            width = segments[0].width
            height = segments[0].height
            bbox = segments[0].bbox
        else:
            start_page = page_range[0] if page_range else 0
            end_page = page_range[1] if page_range else 0
            target_path = (
                out_dir
                / f"table_{order_index:04d}_merged_p{start_page:04d}_{end_page:04d}.png"
            )
            width, height = self._merge_images(
                [segment.file_path for segment in segments],
                target_path,
            )
            file_path = str(target_path)
            bbox = None

        markdown = _render_markdown_table(headers, rows, caption=caption)
        html_value = _render_html_table(headers, rows, caption=caption)
        table_key = (
            f"{doc_id}:{order_index}:{page_range}:{caption}:{headers}:"
            f"{sum(len(row) for row in rows)}"
        )
        table_id = hashlib.sha1(
            table_key.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:16]

        record = PdfTableRecord(
            table_id=table_id,
            doc_id=doc_id,
            order_index=order_index,
            section_path=section_path,
            page_range=page_range,
            bbox=bbox,
            caption=caption,
            headers=headers,
            rows=rows,
            markdown=markdown,
            html=html_value,
            file_path=file_path,
            width=width,
            height=height,
            merged=len(segments) > 1,
            merge_confidence=round(sum(merge_scores) / len(merge_scores), 3)
            if merge_scores
            else None,
            segments=[
                TableSegmentRecord(
                    page=segment.page,
                    bbox=segment.bbox,
                    caption=segment.caption,
                    file_path=segment.file_path,
                    width=segment.width,
                    height=segment.height,
                )
                for segment in segments
            ],
        )
        table_issues: list[dict[str, Any]] = []
        for segment in segments:
            for issue in segment.issues:
                enriched = {**issue}
                details = dict(issue.get("details") or {})
                details.setdefault("table_id", table_id)
                details.setdefault("segment_index", segment.segment_index)
                details.setdefault("page", segment.page)
                enriched["details"] = details
                table_issues.append(enriched)

        merge_observation: dict[str, Any] | None = None
        if len(segments) > 1:
            merge_observation = {
                "decision": "merged",
                "score": record.merge_confidence,
                "pages": page_range,
                "segment_indices": [segment.segment_index for segment in segments],
                "reason": "adjacent table segments passed all merge checks",
            }
        caption_observation = {
            "text": caption,
            "source": caption_segment.caption_source if caption_segment else None,
            "confidence": caption_segment.caption_confidence
            if caption_segment
            else None,
            "candidates": caption_segment.caption_candidates if caption_segment else [],
            "needs_review": bool(
                caption_segment
                and caption_segment.caption_source != "docling_caption_ref"
            )
            or caption is None,
        }
        return (
            record,
            {
                "issues": table_issues,
                "merge": merge_observation,
                "caption": caption_observation,
            },
        )

    def _merge_table_segments(
        self,
        *,
        doc_id: str,
        out_dir: Path,
        segments: list[_TableSegment],
    ) -> tuple[list[PdfTableRecord], list[dict[str, Any]], dict[str, dict[str, Any]]]:
        if not segments:
            return [], [], {}

        ordered_segments = sorted(
            segments,
            key=lambda segment: (
                segment.page or 0,
                segment.bbox[1] if segment.bbox is not None else 0.0,
                segment.bbox[0] if segment.bbox is not None else 0.0,
            ),
        )

        logical_tables: list[PdfTableRecord] = []
        merge_decisions: list[dict[str, Any]] = []
        table_observations: dict[str, dict[str, Any]] = {}
        idx = 0
        order_index = 0
        while idx < len(ordered_segments):
            cluster = [ordered_segments[idx]]
            merge_scores: list[float] = []
            next_idx = idx + 1
            while next_idx < len(ordered_segments):
                observation = self._evaluate_merge_candidate(
                    cluster[-1], ordered_segments[next_idx]
                )
                merge_decisions.append(observation)
                score = observation.get("score")
                if observation.get("decision") != "merged" or score is None:
                    break
                cluster.append(ordered_segments[next_idx])
                merge_scores.append(score)
                next_idx += 1

            table, table_observation = self._build_logical_table(
                doc_id=doc_id,
                order_index=order_index,
                out_dir=out_dir,
                segments=cluster,
                merge_scores=merge_scores,
            )
            logical_tables.append(table)
            table_observations[table.table_id] = table_observation
            order_index += 1
            idx = next_idx
        return logical_tables, merge_decisions, table_observations

    def _extract_figures(
        self,
        *,
        doc_id: str,
        document: Any,
        chunks: list[ChunkRecord],
        out_dir: Path,
        text_by_ref: dict[str, str],
        page_text_blocks: list[_PageTextBlock],
    ) -> tuple[list[PdfFigureRecord], list[dict[str, Any]], dict[str, dict[str, Any]]]:
        figures: list[PdfFigureRecord] = []
        issues: list[dict[str, Any]] = []
        figure_observations: dict[str, dict[str, Any]] = {}
        for order_index, item in enumerate(getattr(document, "pictures", [])):
            image = item.get_image(document)
            if image is None:
                issues.append(
                    _diagnostic(
                        "PDF_FIGURE_IMAGE_UNAVAILABLE",
                        "Docling detected a figure item but did not provide an image crop.",
                        details={"order_index": order_index},
                        hint="Inspect the source PDF manually and keep the sample if missing figure crops hurt downstream reading.",
                    )
                )
                continue

            page, bbox, _page_height = self._resolve_page_bbox(item, document)
            path = self._figure_path(out_dir, order_index=order_index, page=page)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            label = getattr(item, "label", "picture")
            kind = str(getattr(label, "value", label))
            caption_resolution = self._resolve_caption_resolution(
                item=item,
                kind=kind,
                page=page,
                bbox=bbox,
                text_by_ref=text_by_ref,
                page_text_blocks=page_text_blocks,
            )
            caption = caption_resolution.get("text")
            figure_key = f"{doc_id}:{order_index}:{page}:{bbox}:{kind}"
            figure_id = hashlib.sha1(
                figure_key.encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:16]
            figure_issues: list[dict[str, Any]] = []
            if bbox is None:
                figure_issues.append(
                    _diagnostic(
                        "PDF_FIGURE_BBOX_MISSING",
                        "A figure was extracted without page bounding-box provenance.",
                        details={"figure_id": figure_id, "page": page},
                        hint="Use the saved figure image and nearby chunk context for inspection; missing bbox should be reported if it blocks automation.",
                    )
                )
            figure_issues.extend(
                self._caption_issues(
                    entity_kind="figure",
                    item_id=figure_id,
                    page=page,
                    resolution=caption_resolution,
                )
            )
            for issue in figure_issues:
                details = dict(issue.get("details") or {})
                details.setdefault("kind", kind)
                details.setdefault("caption_source", caption_resolution.get("source"))
                details.setdefault(
                    "caption_confidence", caption_resolution.get("confidence")
                )
                issue["details"] = details
            issues.extend(figure_issues)
            figure = PdfFigureRecord(
                figure_id=figure_id,
                doc_id=doc_id,
                order_index=order_index,
                section_path=self._section_path_for_page(chunks, page),
                page=page,
                bbox=bbox,
                caption=caption,
                kind=kind,
                file_path=str(path),
                width=image.width,
                height=image.height,
            )
            figures.append(figure)
            figure_observations[figure_id] = {
                "issues": figure_issues,
                "caption": {
                    "text": caption,
                    "source": caption_resolution.get("source"),
                    "confidence": caption_resolution.get("confidence"),
                    "candidates": caption_resolution.get("candidates") or [],
                    "needs_review": bool(
                        caption_resolution.get("source") != "docling_caption_ref"
                    )
                    or caption is None,
                },
            }
        return figures, issues, figure_observations

    def extract(
        self,
        *,
        pdf_path: str,
        doc_id: str,
        chunks: list[ChunkRecord],
        tables_dir: Path,
        figures_dir: Path,
    ) -> PdfVisualExtractionResult:
        path = Path(pdf_path).expanduser().resolve()
        if not path.exists():
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                f"Document not found: {pdf_path}",
            )

        try:
            converter = self._build_docling_converter()
            result = converter.convert(str(path))
            document = result.document
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PDF_DOCLING_FAILED,
                f"Docling visual extraction failed for {pdf_path}",
            ) from exc

        text_by_ref = self._build_text_by_ref(document)
        page_text_blocks = self._build_page_text_blocks(document)
        table_segments, table_segment_issues = self._extract_table_segments(
            document=document,
            chunks=chunks,
            out_dir=tables_dir,
            text_by_ref=text_by_ref,
            page_text_blocks=page_text_blocks,
        )
        tables, merge_decisions, table_observations = self._merge_table_segments(
            doc_id=doc_id,
            out_dir=tables_dir,
            segments=table_segments,
        )
        figures, figure_issues, figure_observations = self._extract_figures(
            doc_id=doc_id,
            document=document,
            chunks=chunks,
            out_dir=figures_dir,
            text_by_ref=text_by_ref,
            page_text_blocks=page_text_blocks,
        )
        issues = table_segment_issues + figure_issues
        if not table_segments:
            issues.append(
                _diagnostic(
                    "PDF_TABLES_NOT_DETECTED",
                    "Docling visual extraction found no tables in this PDF.",
                    severity="info",
                    hint="If you expected tables, inspect the PDF manually and report the sample for parser tuning.",
                    details={"doc_id": doc_id},
                )
            )
        if not figures:
            issues.append(
                _diagnostic(
                    "PDF_FIGURES_NOT_DETECTED",
                    "Docling visual extraction found no figures in this PDF.",
                    severity="info",
                    hint="If you expected charts or figures, inspect the PDF manually and report the sample for parser tuning.",
                    details={"doc_id": doc_id},
                )
            )

        diagnostics = {
            "extractor": "docling-visuals",
            "summary": {
                "tables_detected_raw": len(table_segments),
                "tables_returned": len(tables),
                "figures_returned": len(figures),
                "merged_tables_count": sum(1 for table in tables if table.merged),
                "issues_count": len(issues),
                "warning_count": _severity_count(issues, "warning"),
                "error_count": _severity_count(issues, "error"),
                "info_count": _severity_count(issues, "info"),
            },
            "issues": issues,
            "merge_decisions": merge_decisions,
            "tables": table_observations,
            "figures": figure_observations,
        }
        return PdfVisualExtractionResult(
            tables=tables,
            figures=figures,
            diagnostics=diagnostics,
        )
