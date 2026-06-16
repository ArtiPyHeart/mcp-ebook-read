from pathlib import Path
from typing import Any

from mcp_ebook_read.benchmark.ingest import (
    read_profile_manifest,
    run_service_ingest_benchmark,
)


class FakeIngestService:
    def __init__(self) -> None:
        self.default_library_root = Path.cwd()
        self.calls: list[tuple[str, str, str | None, bool]] = []

    def _submit(
        self, operation: str, path: str | None, root: str | None, force: bool
    ) -> dict[str, Any]:
        assert path is not None
        source = Path(path)
        sidecar = Path(root or self.default_library_root) / ".mcp-ebook-read"
        sidecar.mkdir(exist_ok=True)
        (sidecar / "catalog.db").write_text("fake", encoding="utf-8")
        doc_id = source.stem
        self.calls.append((operation, str(source), root, force))
        return {
            "doc_id": doc_id,
            "job_id": f"job-{doc_id}",
            "status": "queued",
            "cached": False,
            "deduplicated": False,
        }

    def document_ingest_epub_book(
        self, *, doc_id: str | None, path: str | None, root: str | None, force: bool
    ) -> dict[str, Any]:
        assert doc_id is None
        return self._submit("document_ingest_epub_book", path, root, force)

    def document_ingest_pdf_book(
        self, *, doc_id: str | None, path: str | None, root: str | None, force: bool
    ) -> dict[str, Any]:
        assert doc_id is None
        return self._submit("document_ingest_pdf_book", path, root, force)

    def document_ingest_pdf_paper(
        self, *, doc_id: str | None, path: str | None, root: str | None, force: bool
    ) -> dict[str, Any]:
        assert doc_id is None
        return self._submit("document_ingest_pdf_paper", path, root, force)

    def document_ingest_status(
        self, doc_id: str, job_id: str | None = None
    ) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "job_id": job_id,
            "status": "succeeded",
            "progress": {"stage": "completed", "pct": 100.0},
            "result": {
                "doc_id": doc_id,
                "chunks_count": 2,
                "formulas_count": 0,
                "images_count": 0,
            },
        }


class FakeLateTerminalService(FakeIngestService):
    def __init__(self) -> None:
        super().__init__()
        self.status_calls = 0

    def document_ingest_status(
        self, doc_id: str, job_id: str | None = None
    ) -> dict[str, Any]:
        self.status_calls += 1
        if self.status_calls == 1:
            return {
                "doc_id": doc_id,
                "job_id": job_id,
                "status": "queued",
                "progress": {"stage": "parsing", "pct": 50.0},
            }
        return {
            "doc_id": doc_id,
            "job_id": job_id,
            "status": "succeeded",
            "progress": {"stage": "completed", "pct": 100.0},
            "result": {"doc_id": doc_id, "chunks_count": 1},
        }


def test_run_service_ingest_benchmark_drives_real_tool_routing(
    tmp_path: Path,
) -> None:
    epub = tmp_path / "books" / "book.epub"
    paper = tmp_path / "papers" / "paper.pdf"
    epub.parent.mkdir()
    paper.parent.mkdir()
    epub.write_bytes(b"epub")
    paper.write_bytes(b"pdf")
    old_sidecar = tmp_path / ".mcp-ebook-read"
    old_sidecar.mkdir()
    (old_sidecar / "catalog.db").write_text("old", encoding="utf-8")

    service = FakeIngestService()
    result = run_service_ingest_benchmark(
        [paper, epub],
        service=service,
        pdf_profile="auto",
        library_root=tmp_path,
        force=True,
        delete_sidecars=True,
        poll_interval_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert result["summary"]["documents_total"] == 2
    assert result["summary"]["documents_ok"] == 2
    assert result["summary"]["documents_failed"] == 0
    assert str(old_sidecar) in result["summary"]["deleted_sidecars"]
    assert {call[0] for call in service.calls} == {
        "document_ingest_epub_book",
        "document_ingest_pdf_paper",
    }
    assert {call[2] for call in service.calls} == {str(tmp_path.resolve())}
    assert all(document["sidecar_bytes"] > 0 for document in result["documents"])


def test_profile_manifest_routes_mixed_pdf_profiles_with_space_paths(
    tmp_path: Path,
) -> None:
    paper = tmp_path / "market making paper.pdf"
    book = tmp_path / "book.pdf"
    epub = tmp_path / "book with spaces.epub"
    paper.write_bytes(b"pdf")
    book.write_bytes(b"pdf")
    epub.write_bytes(b"epub")
    manifest = tmp_path / "profile.manifest"
    manifest.write_text(
        "\n".join(
            [
                "# explicit mixed profiles",
                f"paper {paper.name}",
                f"book {book.name}",
                f"epub {epub.name}",
            ]
        ),
        encoding="utf-8",
    )

    profiled = read_profile_manifest(manifest)
    service = FakeIngestService()
    result = run_service_ingest_benchmark(
        [item.path for item in profiled],
        service=service,
        document_pdf_profiles={
            str(item.path): item.profile
            for item in profiled
            if item.path.suffix.lower() == ".pdf"
        },
        pdf_profile="auto",
        library_root=tmp_path,
        poll_interval_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert [(item.path, item.profile) for item in profiled] == [
        (paper.resolve(), "paper"),
        (book.resolve(), "book"),
        (epub.resolve(), "epub"),
    ]
    assert {call[0] for call in service.calls} == {
        "document_ingest_pdf_paper",
        "document_ingest_pdf_book",
        "document_ingest_epub_book",
    }
    effective_profiles = {
        Path(document["path"]).name: document["effective_profile"]
        for document in result["documents"]
    }
    assert effective_profiles == {
        "market making paper.pdf": "paper",
        "book.pdf": "book",
        "book with spaces.epub": "epub",
    }


def test_run_service_ingest_benchmark_uses_timeout_grace_final_poll(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "papers" / "late.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"pdf")
    service = FakeLateTerminalService()

    result = run_service_ingest_benchmark(
        [pdf],
        service=service,
        pdf_profile="paper",
        library_root=tmp_path,
        poll_interval_seconds=0.01,
        timeout_seconds=0.0,
    )

    assert service.status_calls == 2
    assert result["summary"]["documents_ok"] == 1
    assert result["documents"][0]["status"] == "succeeded"
    assert result["documents"][0]["final"]["status"] == "succeeded"
