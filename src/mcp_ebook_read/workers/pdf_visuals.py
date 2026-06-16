"""Isolated worker for heavyweight Docling PDF visual extraction."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from mcp_ebook_read.errors import to_error_payload
from mcp_ebook_read.render.pdf_visuals import DoclingPdfVisualExtractor
from mcp_ebook_read.schema.models import ChunkRecord, PdfParserPerformanceConfig


def _load_config() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--tables-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    config = _load_config()
    performance_config_payload = config.get("performance_config")
    performance_config = (
        PdfParserPerformanceConfig.model_validate(performance_config_payload)
        if isinstance(performance_config_payload, dict)
        else PdfParserPerformanceConfig()
    )
    chunks = [
        ChunkRecord.model_validate(chunk)
        for chunk in config.get("chunks", [])
        if isinstance(chunk, dict)
    ]
    extractor = DoclingPdfVisualExtractor(
        performance_config=performance_config,
        images_scale=float(config.get("images_scale") or 2.0),
    )
    try:
        extracted = extractor.extract(
            pdf_path=args.pdf_path,
            doc_id=args.doc_id,
            chunks=chunks,
            tables_dir=Path(args.tables_dir),
            figures_dir=Path(args.figures_dir),
        )
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "data": {
                        "tables": [
                            table.model_dump(mode="json") for table in extracted.tables
                        ],
                        "figures": [
                            figure.model_dump(mode="json")
                            for figure in extracted.figures
                        ],
                        "diagnostics": extracted.diagnostics,
                    },
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": to_error_payload(exc),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
