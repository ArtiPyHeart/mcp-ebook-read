"""Isolated worker for heavyweight Docling PDF parsing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from mcp_ebook_read.errors import to_error_payload
from mcp_ebook_read.parsers.pdf_docling import DoclingPdfParser
from mcp_ebook_read.render.pdf_visuals import DoclingPdfVisualExtractor
from mcp_ebook_read.schema.models import PdfParserPerformanceConfig


def _load_config() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--doc-id", required=True)
    args = parser.parse_args()

    config = _load_config()
    performance_config_payload = config.get("performance_config")
    performance_config = (
        PdfParserPerformanceConfig.model_validate(performance_config_payload)
        if isinstance(performance_config_payload, dict)
        else PdfParserPerformanceConfig()
    )
    visual_config = config.get("visual_extraction")
    visual_extractor = None
    visual_tables_dir = None
    visual_figures_dir = None
    visual_images_scale = 2.0
    if isinstance(visual_config, dict):
        visual_tables_dir = visual_config.get("tables_dir")
        visual_figures_dir = visual_config.get("figures_dir")
        visual_images_scale = float(visual_config.get("images_scale") or 2.0)
        if visual_tables_dir and visual_figures_dir:
            visual_tables_dir = Path(str(visual_tables_dir))
            visual_figures_dir = Path(str(visual_figures_dir))
            visual_extractor = DoclingPdfVisualExtractor(
                performance_config=performance_config,
                images_scale=visual_images_scale,
            )
    pdf_parser = DoclingPdfParser(
        enable_docling_formula_enrichment=bool(
            config.get("enable_docling_formula_enrichment", True)
        ),
        require_formula_engine=bool(config.get("require_formula_engine", True)),
        formula_batch_size=int(config.get("formula_batch_size") or 1),
        performance_config=performance_config,
        enable_visual_images=visual_extractor is not None,
        visual_images_scale=visual_images_scale,
    )
    try:
        parsed = pdf_parser.parse(
            args.pdf_path,
            args.doc_id,
            visual_extractor=visual_extractor,
            visual_tables_dir=visual_tables_dir,
            visual_figures_dir=visual_figures_dir,
        )
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "data": parsed.model_dump(mode="json"),
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
    finally:
        pdf_parser.close()


if __name__ == "__main__":
    raise SystemExit(main())
