"""GROBID client for scientific paper metadata and outline extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from lxml import etree

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.network import should_trust_env_proxy
from mcp_ebook_read.schema.models import OutlineNode

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class GrobidResult:
    metadata: dict
    outline: list[OutlineNode]


class GrobidClient:
    """Fail-fast GROBID wrapper."""

    def __init__(self, base_url: str | None, timeout_seconds: float = 20.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._trust_env_proxy = should_trust_env_proxy(self.base_url)

    @classmethod
    def from_env(cls) -> "GrobidClient":
        return cls(
            base_url=os.environ.get("GROBID_URL"),
            timeout_seconds=float(os.environ.get("GROBID_TIMEOUT_SECONDS", "20")),
        )

    def assert_available(self) -> None:
        """Explicit startup preflight check for GROBID availability."""
        self._assert_available()

    def _assert_available(self) -> None:
        if not self.base_url:
            raise AppError(
                ErrorCode.INGEST_PAPER_GROBID_UNAVAILABLE,
                "GROBID_URL is not configured.",
            )

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                trust_env=self._trust_env_proxy,
            ) as client:
                response = client.get(f"{self.base_url}/api/isalive")
                response.raise_for_status()
                if "true" not in response.text.lower():
                    raise AppError(
                        ErrorCode.INGEST_PAPER_GROBID_UNAVAILABLE,
                        "GROBID is not alive.",
                    )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PAPER_GROBID_UNAVAILABLE,
                "Cannot connect to GROBID service.",
                details={"base_url": self.base_url},
            ) from exc

    def parse_fulltext(self, pdf_path: str) -> GrobidResult:
        self._assert_available()

        path = Path(pdf_path)
        if not path.exists():
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                f"Document not found: {pdf_path}",
            )

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                trust_env=self._trust_env_proxy,
            ) as client:
                with path.open("rb") as fh:
                    response = client.post(
                        f"{self.base_url}/api/processFulltextDocument",
                        files={"input": (path.name, fh, "application/pdf")},
                    )
                response.raise_for_status()
                tei_xml = response.text
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PAPER_GROBID_FAILED,
                f"GROBID parsing failed for {pdf_path}",
            ) from exc

        return self._parse_tei(tei_xml)

    def _parse_tei(self, tei_xml: str) -> GrobidResult:
        try:
            root = etree.fromstring(tei_xml.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_PAPER_GROBID_FAILED,
                "Invalid TEI XML returned by GROBID.",
            ) from exc

        title = " ".join(
            root.xpath("//tei:titleStmt/tei:title[1]//text()", namespaces=NS)
        ).strip()
        abstract = " ".join(root.xpath("//tei:abstract//text()", namespaces=NS)).strip()
        doi = str(
            root.xpath(
                "string((//tei:sourceDesc//tei:biblStruct[1]//tei:idno[@type='DOI'])[1])",
                namespaces=NS,
            )
        ).strip()

        sections = root.xpath("//tei:body/tei:div", namespaces=NS)
        outline: list[OutlineNode] = []
        for idx, div in enumerate(sections):
            heading = " ".join(div.xpath("./tei:head//text()", namespaces=NS)).strip()
            if not heading:
                continue
            outline.append(
                OutlineNode(
                    id=f"grobid-{idx}",
                    title=heading,
                    level=1,
                )
            )

        biblio_count = int(
            root.xpath("count(//tei:listBibl/tei:biblStruct)", namespaces=NS)
        )
        references: list[dict] = []
        for idx, bibl in enumerate(
            root.xpath("//tei:listBibl/tei:biblStruct", namespaces=NS),
            start=1,
        ):
            xml_id = (
                bibl.get("{http://www.w3.org/XML/1998/namespace}id")
                or bibl.get("xml:id")
                or f"ref-{idx}"
            )
            ref_title = " ".join(
                bibl.xpath(
                    ".//tei:analytic/tei:title[1]//text() | .//tei:monogr/tei:title[1]//text()",
                    namespaces=NS,
                )
            ).strip()
            authors = []
            for author in bibl.xpath(".//tei:author", namespaces=NS):
                author_name = " ".join(author.xpath(".//text()", namespaces=NS)).strip()
                if author_name:
                    authors.append(author_name)
            date = " ".join(
                bibl.xpath(".//tei:date/@when | .//tei:date/text()", namespaces=NS)
            ).strip()
            ref_doi = " ".join(
                bibl.xpath(".//tei:idno[@type='DOI']/text()", namespaces=NS)
            ).strip()
            raw_text = " ".join(bibl.xpath(".//text()", namespaces=NS)).strip()
            references.append(
                {
                    "reference_id": xml_id,
                    "title": ref_title or None,
                    "authors": authors,
                    "date": date or None,
                    "doi": ref_doi or None,
                    "raw_text": raw_text or None,
                    "source": "grobid",
                }
            )

        citations: list[dict] = []
        for idx, ref in enumerate(
            root.xpath("//tei:body//tei:ref[@type='bibr']", namespaces=NS),
            start=1,
        ):
            target = (ref.get("target") or "").strip()
            text = " ".join(ref.xpath(".//text()", namespaces=NS)).strip()
            citations.append(
                {
                    "citation_id": f"cite-{idx}",
                    "target": target or None,
                    "text": text or target or f"cite-{idx}",
                    "source": "grobid",
                }
            )
        metadata = {
            "paper_title": title or None,
            "abstract": abstract or None,
            "doi": doi or None,
            "bibliography_count": biblio_count,
            "references": references,
            "citations": citations,
        }
        return GrobidResult(metadata=metadata, outline=outline)
