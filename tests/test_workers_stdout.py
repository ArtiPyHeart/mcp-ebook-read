from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

from mcp_ebook_read.schema.models import ParsedDocument
from mcp_ebook_read.workers import pdf_parse, pdf_visuals


class NoisyPdfParser:
    def __init__(self, **kwargs: Any) -> None:
        print("parser init noise")
        self.kwargs = kwargs

    def parse(self, *args: Any, **kwargs: Any) -> ParsedDocument:
        print("parser parse noise")
        return ParsedDocument(
            title="Noisy PDF",
            parser_chain=["noisy"],
            metadata={},
            outline=[],
            chunks=[],
            reading_markdown="",
        )

    def close(self) -> None:
        print("parser close noise")


class NoisyVisualExtractor:
    def __init__(self, **kwargs: Any) -> None:
        print("visual init noise")

    def extract(self, **kwargs: Any):  # noqa: ANN201
        print("visual extract noise")

        class Extracted:
            tables: list[Any] = []
            figures: list[Any] = []
            diagnostics: dict[str, Any] = {}

        return Extracted()


def test_pdf_parse_worker_keeps_stdout_json_only(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf_parse, "DoclingPdfParser", NoisyPdfParser)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pdf_parse",
            "--pdf-path",
            str(tmp_path / "doc.pdf"),
            "--doc-id",
            "doc-noisy",
        ],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))

    assert pdf_parse.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["data"]["title"] == "Noisy PDF"
    assert "parser parse noise" not in captured.out
    assert "parser parse noise" in captured.err


def test_pdf_visuals_worker_keeps_stdout_json_only(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf_visuals, "DoclingPdfVisualExtractor", NoisyVisualExtractor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pdf_visuals",
            "--pdf-path",
            str(tmp_path / "doc.pdf"),
            "--doc-id",
            "doc-noisy",
            "--tables-dir",
            str(tmp_path / "tables"),
            "--figures-dir",
            str(tmp_path / "figures"),
        ],
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))

    assert pdf_visuals.main() == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["data"]["tables"] == []
    assert "visual extract noise" not in captured.out
    assert "visual extract noise" in captured.err
