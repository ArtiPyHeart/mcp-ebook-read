from __future__ import annotations

from pathlib import Path

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentType,
    Locator,
)
from mcp_ebook_read.service import AppService
from mcp_ebook_read.store.catalog import CatalogStore


class DummyVectorIndex:
    def rebuild_document(
        self, doc_id: str, title: str, chunks: list[ChunkRecord]
    ) -> None:
        return None

    def search(self, query: str, top_k: int = 20, doc_ids: list[str] | None = None):
        return []


class DummyParser:
    def parse(self, pdf_path: str, doc_id: str):
        raise NotImplementedError


class DummyGrobid:
    def parse_fulltext(self, pdf_path: str):
        raise NotImplementedError


def build_service(tmp_path: Path) -> AppService:
    sidecar_dir = tmp_path / ".mcp-ebook-read"
    return AppService(
        data_dir=sidecar_dir,
        catalog=CatalogStore(sidecar_dir / "catalog.db"),
        sidecar_dir_name=".mcp-ebook-read",
        vector_index=DummyVectorIndex(),
        pdf_parser=DummyParser(),
        grobid_client=DummyGrobid(),
        epub_parser=DummyParser(),
    )


def test_read_locator_not_found(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    with pytest.raises(AppError) as exc:
        service.read(
            locator={
                "doc_id": "missing",
                "chunk_id": "missing",
                "method": "docling",
            },
            before=1,
            after=1,
            out_format="markdown",
        )
    assert exc.value.code == ErrorCode.READ_LOCATOR_NOT_FOUND


def test_read_success(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    doc_id = "abcd1234abcd1234"
    service.catalog.upsert_scanned_document(
        DocumentRecord(
            doc_id=doc_id,
            path=str((tmp_path / "x.pdf").resolve()),
            type=DocumentType.PDF,
            sha256="f" * 64,
            mtime=1.0,
        )
    )

    chunks = []
    for idx in range(2):
        locator = Locator(
            doc_id=doc_id,
            chunk_id=f"c{idx}",
            section_path=["Sec"],
            page_range=[1, 1],
            method="docling",
        )
        chunks.append(
            ChunkRecord(
                chunk_id=f"c{idx}",
                doc_id=doc_id,
                order_index=idx,
                section_path=["Sec"],
                text=f"text {idx}",
                search_text=f"text {idx}",
                locator=locator,
                method="docling",
            )
        )
    service.catalog.replace_chunks(doc_id, chunks)

    data = service.read(
        locator=chunks[0].locator.model_dump(), before=0, after=1, out_format="text"
    )
    assert "text 0" in data["content"]
    assert "text 1" in data["content"]
