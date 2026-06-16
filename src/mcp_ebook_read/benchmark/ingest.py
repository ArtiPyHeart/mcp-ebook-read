"""Service-level benchmark for the real MCP ingest path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any, Protocol

from mcp_ebook_read.benchmark.paths import DOCUMENT_SUFFIXES, collect_documents
from mcp_ebook_read.errors import to_error_payload
from mcp_ebook_read.schema.models import IngestJobStatus

_PDF_PROFILES = {"auto", "book", "paper"}
_PROFILE_MANIFEST_PROFILES = {"book", "paper", "epub"}
_TERMINAL_STATUSES = {
    IngestJobStatus.SUCCEEDED,
    IngestJobStatus.FAILED,
    IngestJobStatus.CANCELED,
    IngestJobStatus.SUCCEEDED.value,
    IngestJobStatus.FAILED.value,
    IngestJobStatus.CANCELED.value,
}
_PAPER_PATH_HINTS = ("paper", "papers", "论文", "article", "arxiv")


@dataclass(frozen=True)
class ProfiledDocument:
    path: Path
    profile: str


class IngestServiceProtocol(Protocol):
    def document_ingest(
        self, *, doc_id: str | None, path: str | None, root: str | None, force: bool
    ) -> dict[str, Any]: ...

    def document_ingest_status(
        self, doc_id: str, job_id: str | None = None
    ) -> dict[str, Any]: ...


def _infer_pdf_profile(path: Path) -> str:
    lowered_parts = [part.lower() for part in path.parts]
    if any(any(hint in part for hint in _PAPER_PATH_HINTS) for part in lowered_parts):
        return "paper"
    return "book"


def read_profile_manifest(manifest: Path) -> list[ProfiledDocument]:
    """Read newline-delimited ``profile path`` entries for mixed benchmark sets."""
    base = manifest.expanduser().resolve().parent
    profiled: list[ProfiledDocument] = []
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            profile, raw_path = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                "Profile manifest lines must be formatted as `paper|book|epub <path>`."
            ) from exc
        if profile not in _PROFILE_MANIFEST_PROFILES:
            raise ValueError(
                f"Unsupported profile '{profile}' at {manifest}:{line_number}. "
                "Expected one of: book, epub, paper."
            )
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base / path
        resolved = path.resolve()
        suffix = resolved.suffix.lower()
        if profile == "epub" and suffix != ".epub":
            raise ValueError(
                f"Profile 'epub' requires an .epub file at {manifest}:{line_number}."
            )
        if profile in {"book", "paper"} and suffix != ".pdf":
            raise ValueError(
                f"Profile '{profile}' requires a .pdf file at {manifest}:{line_number}."
            )
        profiled.append(ProfiledDocument(path=resolved, profile=profile))
    return profiled


def _sidecar_dir_for_root(root: Path, sidecar_dir_name: str) -> Path:
    return root.expanduser().resolve() / sidecar_dir_name


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _delete_root_sidecar(root: Path, *, sidecar_dir_name: str) -> list[str]:
    sidecar_dir = _sidecar_dir_for_root(root, sidecar_dir_name)
    if not sidecar_dir.exists():
        return []
    shutil.rmtree(sidecar_dir)
    return [str(sidecar_dir)]


def _submit_ingest(
    service: IngestServiceProtocol,
    *,
    path: Path,
    root: Path,
    force: bool,
) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix not in {".epub", ".pdf"}:
        raise ValueError(f"Unsupported ingest benchmark document type: {suffix}")
    return (
        "document_ingest",
        service.document_ingest(
            doc_id=None,
            path=str(path),
            root=str(root),
            force=force,
        ),
    )


def _poll_ingest_job(
    service: IngestServiceProtocol,
    *,
    doc_id: str,
    job_id: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    while True:
        status = service.document_ingest_status(doc_id=doc_id, job_id=job_id)
        if status.get("status") in _TERMINAL_STATUSES:
            return status
        if time.perf_counter() - started > timeout_seconds:
            time.sleep(min(5.0, max(0.05, poll_interval_seconds * 5)))
            final_status = service.document_ingest_status(doc_id=doc_id, job_id=job_id)
            if final_status.get("status") in _TERMINAL_STATUSES:
                return final_status
            return {
                **final_status,
                "status": "timeout",
                "error": {
                    "code": "INGEST_BENCHMARK_TIMEOUT",
                    "message": "Ingest benchmark timed out while polling job status.",
                    "details": {
                        "doc_id": doc_id,
                        "job_id": job_id,
                        "timeout_seconds": timeout_seconds,
                    },
                },
            }
        time.sleep(max(0.01, poll_interval_seconds))


def run_service_ingest_benchmark(
    document_paths: list[Path],
    *,
    service: IngestServiceProtocol | None = None,
    pdf_profile: str = "auto",
    document_pdf_profiles: dict[str, str] | None = None,
    library_root: Path | None = None,
    force: bool = True,
    delete_sidecars: bool = False,
    sidecar_dir_name: str = ".mcp-ebook-read",
    poll_interval_seconds: float = 0.2,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Run real service ingest jobs and record sidecar/product-path metrics."""
    if pdf_profile not in _PDF_PROFILES:
        raise ValueError(f"Unsupported pdf_profile: {pdf_profile}")

    resolved_paths = [
        path.expanduser().resolve()
        for path in sorted(document_paths)
        if path.suffix.lower() in DOCUMENT_SUFFIXES
    ]
    owns_service = service is None
    if service is None:
        from mcp_ebook_read.service import AppService

        service = AppService.from_env()

    effective_root = (
        library_root.expanduser().resolve()
        if library_root is not None
        else Path(getattr(service, "default_library_root", Path.cwd())).resolve()
    )
    deleted_sidecars = (
        _delete_root_sidecar(effective_root, sidecar_dir_name=sidecar_dir_name)
        if delete_sidecars
        else []
    )

    documents: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    try:
        for path in resolved_paths:
            doc_started = time.perf_counter()
            sidecar_dir = _sidecar_dir_for_root(effective_root, sidecar_dir_name)
            try:
                path_profile = (
                    document_pdf_profiles.get(str(path))
                    if document_pdf_profiles is not None
                    else None
                )
                operation, submitted = _submit_ingest(
                    service,
                    path=path,
                    root=effective_root,
                    force=force,
                )
                doc_id = str(submitted["doc_id"])
                job_id = str(submitted["job_id"])
                final_status = (
                    submitted
                    if submitted.get("status") in _TERMINAL_STATUSES
                    else _poll_ingest_job(
                        service,
                        doc_id=doc_id,
                        job_id=job_id,
                        poll_interval_seconds=poll_interval_seconds,
                        timeout_seconds=timeout_seconds,
                    )
                )
                elapsed_seconds = time.perf_counter() - doc_started
                effective_profile = (
                    "epub"
                    if path.suffix.lower() == ".epub"
                    else path_profile or pdf_profile
                )
                documents.append(
                    {
                        "path": str(path),
                        "status": final_status.get("status"),
                        "operation": operation,
                        "doc_id": doc_id,
                        "job_id": job_id,
                        "elapsed_seconds": round(elapsed_seconds, 6),
                        "sidecar_path": str(sidecar_dir),
                        "sidecar_bytes": _directory_size_bytes(sidecar_dir),
                        "effective_profile": effective_profile,
                        "cached": bool(submitted.get("cached")),
                        "deduplicated": bool(submitted.get("deduplicated")),
                        "submit": submitted,
                        "final": final_status,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                documents.append(
                    {
                        "path": str(path),
                        "status": "error",
                        "elapsed_seconds": round(time.perf_counter() - doc_started, 6),
                        "sidecar_path": str(sidecar_dir),
                        "sidecar_bytes": _directory_size_bytes(sidecar_dir),
                        "error": to_error_payload(exc),
                    }
                )
    finally:
        if owns_service:
            close = getattr(service, "close", None)
            if callable(close):
                close()

    ok_documents = [
        item
        for item in documents
        if item.get("status")
        in {IngestJobStatus.SUCCEEDED, IngestJobStatus.SUCCEEDED.value}
    ]
    failed_documents = [
        item
        for item in documents
        if item.get("status")
        not in {IngestJobStatus.SUCCEEDED, IngestJobStatus.SUCCEEDED.value}
    ]
    total_seconds = time.perf_counter() - benchmark_started
    return {
        "summary": {
            "documents_total": len(documents),
            "documents_ok": len(ok_documents),
            "documents_failed": len(failed_documents),
            "elapsed_seconds": round(total_seconds, 6),
            "throughput_docs_per_second": round(len(ok_documents) / total_seconds, 6)
            if total_seconds > 0
            else None,
            "sidecar_bytes_total": sum(
                int(item.get("sidecar_bytes") or 0) for item in documents
            ),
            "pdf_profile": pdf_profile,
            "force": force,
            "delete_sidecars": delete_sidecars,
            "deleted_sidecars": deleted_sidecars,
            "library_root": str(effective_root),
        },
        "documents": documents,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-ebook-ingest-benchmark",
        description="Run a service-level benchmark through the real eager MCP ingest path.",
    )
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Newline-delimited EPUB/PDF path manifest. Relative paths resolve from the manifest directory.",
    )
    parser.add_argument(
        "--profile-manifest",
        type=Path,
        help=(
            "Newline-delimited `paper|book|epub <path>` manifest for mixed "
            "benchmark sets. Relative paths resolve from the manifest directory."
        ),
    )
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument(
        "--root",
        type=Path,
        help="Unified library root sidecar for all benchmarked documents. Defaults to the MCP service project root.",
    )
    parser.add_argument("--pdf-profile", choices=sorted(_PDF_PROFILES), default="auto")
    parser.add_argument(
        "--no-force", action="store_true", help="Allow cached READY docs to be reused."
    )
    parser.add_argument(
        "--delete-sidecars",
        action="store_true",
        help="Delete the selected library root sidecar before running cold ingest.",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    document_pdf_profiles: dict[str, str] | None = None
    if args.profile_manifest is not None:
        if args.samples_dir is not None or args.manifest is not None:
            raise SystemExit(
                "--profile-manifest cannot be combined with --samples-dir or --manifest."
            )
        profiled = read_profile_manifest(args.profile_manifest)
        if args.max_documents > 0:
            profiled = profiled[: args.max_documents]
        documents = [item.path for item in profiled]
        document_pdf_profiles = {
            str(item.path): item.profile
            for item in profiled
            if item.path.suffix.lower() == ".pdf"
        }
    else:
        documents = collect_documents(
            samples_dir=args.samples_dir,
            manifest=args.manifest,
            suffixes=DOCUMENT_SUFFIXES,
            max_documents=args.max_documents,
        )
    if not documents:
        raise SystemExit(
            "No EPUB/PDF documents found. Pass --samples-dir, --manifest, or --profile-manifest."
        )
    result = run_service_ingest_benchmark(
        documents,
        pdf_profile=args.pdf_profile,
        document_pdf_profiles=document_pdf_profiles,
        library_root=args.root,
        force=not args.no_force,
        delete_sidecars=args.delete_sidecars,
        poll_interval_seconds=args.poll_interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if result["summary"]["documents_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
