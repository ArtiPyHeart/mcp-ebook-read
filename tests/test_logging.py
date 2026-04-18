from __future__ import annotations

import json
import logging

from mcp_ebook_read.logging import JsonFormatter


def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.makeLogRecord(
        {
            "name": "mcp_ebook_read.index.vector",
            "levelno": logging.WARNING,
            "levelname": "WARNING",
            "msg": "fastembed_init_failed",
            "attempt": 2,
            "retrying": True,
            "cache_dir": "/tmp/fastembed",
        }
    )

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "fastembed_init_failed"
    assert payload["attempt"] == 2
    assert payload["retrying"] is True
    assert payload["cache_dir"] == "/tmp/fastembed"
