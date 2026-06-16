from __future__ import annotations

import zipfile
from pathlib import Path

from mcp_ebook_read.benchmark.parser_engines import (
    _collect_paths,
    _docling_document_metrics,
    _parse_engines,
    _query_replay_from_text,
    _read_queries_file,
    _run_epub_zip_lxml,
    run_parser_engine_benchmark,
)


def _write_minimal_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
            """,
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Minimal EPUB</dc:title>
              </metadata>
              <manifest>
                <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="chapter"/>
              </spine>
            </package>
            """,
        )
        archive.writestr(
            "OEBPS/chapter.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml">
              <body>
                <h1>Chapter One</h1>
                <p>Hello parser benchmark.</p>
                <p>Inventory risk matters in a limit order book.</p>
                <img src="cover.png" alt="cover"/>
              </body>
            </html>
            """,
        )


def test_epub_zip_lxml_extracts_fast_baseline_metrics(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _write_minimal_epub(epub_path)

    result = _run_epub_zip_lxml(epub_path, ["inventory risk"])

    assert result["status"] == "ok"
    assert result["metrics"]["title"] == "Minimal EPUB"
    assert result["metrics"]["spine_documents"] == 1
    assert result["metrics"]["headings"] == 1
    assert result["metrics"]["images"] == 1
    assert result["metrics"]["normalized_text_chars"] > 0
    assert result["metrics"]["query_replay"]["queries_with_hit"] == 1


def test_query_replay_reports_best_match_without_full_text_dump() -> None:
    replay = _query_replay_from_text(
        "The alpha signal depends on the micro-price and queue imbalance.",
        ["micro price", "nonexistent phrase"],
    )

    assert replay["queries_total"] == 2
    assert replay["queries_with_hit"] == 1
    assert replay["queries"][0]["hit"] is True
    assert replay["queries"][0]["best_overlap_ratio"] == 1.0
    assert replay["queries"][0]["best_chunk"]["preview"]
    assert replay["queries"][1]["hit"] is False


def test_docling_document_metrics_counts_optional_collections() -> None:
    class FakeDoclingDocument:
        pages = {1: object(), 2: object()}
        tables = [object()]
        pictures = [object(), object(), object()]
        texts = [object(), object()]
        groups = object()

    metrics = _docling_document_metrics(FakeDoclingDocument())

    assert metrics == {
        "pages": 2,
        "tables": 1,
        "figures": 3,
        "images": 3,
        "docling_text_items": 2,
        "docling_groups": 0,
    }


def test_collect_paths_reads_manifest_relative_to_manifest_dir(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _write_minimal_epub(epub_path)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# comment\nbook.epub\n", encoding="utf-8")

    paths = _collect_paths(
        samples_dir=tmp_path,
        preset="smoke",
        manifest=manifest,
        paths=[],
    )

    assert paths == [epub_path.resolve()]


def test_parse_engines_rejects_unknown_engine() -> None:
    try:
        _parse_engines("epub_zip_lxml,unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_read_queries_file_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text(
        "# benchmark queries\n\nmarket making\n queue position \n",
        encoding="utf-8",
    )

    assert _read_queries_file(queries_file) == ["market making", "queue position"]


def test_run_parser_engine_benchmark_uses_subprocess(tmp_path: Path) -> None:
    epub_path = tmp_path / "book.epub"
    _write_minimal_epub(epub_path)

    result = run_parser_engine_benchmark(
        document_paths=[epub_path],
        engines=["epub_zip_lxml"],
        queries=["parser benchmark", "inventory risk"],
        timeout_seconds=30,
    )

    assert result["summary"]["documents_total"] == 1
    assert result["summary"]["queries"] == ["parser benchmark", "inventory risk"]
    assert result["engine_summary"]["epub_zip_lxml"]["ok"] == 1
    assert result["engine_summary"]["epub_zip_lxml"]["query_replay_hit_rate"] == 1.0
    assert result["documents"][0]["engines"][0]["status"] == "ok"
