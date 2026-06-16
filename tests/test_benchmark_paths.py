from pathlib import Path

from mcp_ebook_read.benchmark.paths import DOCUMENT_SUFFIXES, collect_documents


def test_collect_documents_reads_manifest_relative_to_manifest_dir(
    tmp_path: Path,
) -> None:
    book = tmp_path / "book.epub"
    paper = tmp_path / "nested" / "paper.pdf"
    ignored = tmp_path / "notes.txt"
    paper.parent.mkdir()
    book.write_bytes(b"epub")
    paper.write_bytes(b"pdf")
    ignored.write_text("ignore", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "# representative corpus",
                "book.epub",
                "nested/paper.pdf",
                "notes.txt",
                "book.epub",
            ]
        ),
        encoding="utf-8",
    )

    paths = collect_documents(manifest=manifest, suffixes=DOCUMENT_SUFFIXES)

    assert paths == [book.resolve(), paper.resolve()]


def test_collect_documents_combines_directory_and_manifest_with_limit(
    tmp_path: Path,
) -> None:
    directory_doc = tmp_path / "dir.pdf"
    manifest_doc = tmp_path / "manifest.epub"
    directory_doc.write_bytes(b"pdf")
    manifest_doc.write_bytes(b"epub")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("manifest.epub\n", encoding="utf-8")

    paths = collect_documents(
        samples_dir=tmp_path,
        manifest=manifest,
        suffixes=DOCUMENT_SUFFIXES,
        max_documents=1,
    )

    assert paths == [directory_doc.resolve()]
