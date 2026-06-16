from __future__ import annotations

import json
import logging

from mcp_ebook_read.logging import JsonFormatter


def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.makeLogRecord(
        {
            "name": "mcp_ebook_read.store.catalog",
            "levelno": logging.WARNING,
            "levelname": "WARNING",
            "msg": "local_index_rebuild_delayed",
            "attempt": 2,
            "retrying": True,
            "sidecar_dir": "/tmp/.mcp-ebook-read",
        }
    )

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "local_index_rebuild_delayed"
    assert payload["attempt"] == 2
    assert payload["retrying"] is True
    assert payload["sidecar_dir"] == "/tmp/.mcp-ebook-read"
