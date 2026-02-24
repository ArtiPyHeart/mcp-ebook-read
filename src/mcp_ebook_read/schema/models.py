"""Core schema models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    PDF = "pdf"
    EPUB = "epub"


class DocumentStatus(StrEnum):
    DISCOVERED = "discovered"
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"


class Profile(StrEnum):
    BOOK = "book"
    PAPER = "paper"


class OutlineNode(BaseModel):
    id: str
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    spine_ref: str | None = None
    children: list["OutlineNode"] = Field(default_factory=list)


class Locator(BaseModel):
    doc_id: str
    chunk_id: str
    section_path: list[str] = Field(default_factory=list)
    page_range: list[int] | None = None
    epub_locator: dict[str, str] | None = None
    bbox: list[float] | None = None
    method: str
    confidence: float | None = None


class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    order_index: int
    section_path: list[str] = Field(default_factory=list)
    text: str
    search_text: str
    locator: Locator
    method: str
    confidence: float | None = None


class FormulaRecord(BaseModel):
    formula_id: str
    doc_id: str
    chunk_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    bbox: list[float] | None = None
    latex: str
    source: str
    confidence: float | None = None
    status: str = "resolved"


class ExtractedImage(BaseModel):
    image_id: str
    doc_id: str
    order_index: int
    section_path: list[str] = Field(default_factory=list)
    spine_id: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    href: str | None = None
    anchor: str | None = None
    alt: str | None = None
    caption: str | None = None
    media_type: str | None = None
    extension: str = ".bin"
    width: int | None = None
    height: int | None = None
    data: bytes
    source: str = "epub"
    status: str = "ready"


class ImageRecord(BaseModel):
    image_id: str
    doc_id: str
    order_index: int
    section_path: list[str] = Field(default_factory=list)
    spine_id: str | None = None
    page: int | None = None
    bbox: list[float] | None = None
    href: str | None = None
    anchor: str | None = None
    alt: str | None = None
    caption: str | None = None
    media_type: str | None = None
    file_path: str
    width: int | None = None
    height: int | None = None
    source: str = "epub"
    status: str = "ready"


class DocumentRecord(BaseModel):
    doc_id: str
    path: str
    type: DocumentType
    sha256: str
    mtime: float
    title: str | None = None
    status: DocumentStatus = DocumentStatus.DISCOVERED
    profile: Profile = Profile.BOOK
    parser_chain: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    outline: list[OutlineNode] = Field(default_factory=list)
    overall_confidence: float | None = None


class ParsedDocument(BaseModel):
    title: str
    parser_chain: list[str]
    metadata: dict[str, Any]
    outline: list[OutlineNode]
    chunks: list[ChunkRecord]
    formulas: list[FormulaRecord] = Field(default_factory=list)
    images: list[ExtractedImage] = Field(default_factory=list)
    reading_markdown: str
    raw_artifacts: dict[str, str] = Field(default_factory=dict)
    overall_confidence: float | None = None


class ToolEnvelope(BaseModel):
    ok: bool
    data: dict[str, Any] | None
    error: dict[str, Any] | None
    trace_id: str


OutlineNode.model_rebuild()
