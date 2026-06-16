"""Shared document path collection helpers for benchmark CLIs."""

from __future__ import annotations

from pathlib import Path


DOCUMENT_SUFFIXES = {".epub", ".pdf"}
PDF_SUFFIXES = {".pdf"}


def read_manifest(manifest: Path, *, suffixes: set[str]) -> list[Path]:
    """Read newline-delimited document paths from a manifest file."""
    base = manifest.expanduser().resolve().parent
    paths: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        if not path.is_absolute():
            path = base / path
        if path.suffix.lower() in suffixes:
            paths.append(path.resolve())
    return paths


def collect_documents(
    *,
    samples_dir: Path | None = None,
    manifest: Path | None = None,
    suffixes: set[str] = DOCUMENT_SUFFIXES,
    max_documents: int = 0,
) -> list[Path]:
    """Collect benchmark document paths from a directory and/or manifest."""
    documents: list[Path] = []
    if samples_dir is not None:
        root = samples_dir.expanduser().resolve()
        if root.exists():
            documents.extend(
                path.resolve()
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.lower() in suffixes
            )
    if manifest is not None:
        documents.extend(read_manifest(manifest, suffixes=suffixes))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in documents:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    deduped.sort()
    return deduped[:max_documents] if max_documents > 0 else deduped
