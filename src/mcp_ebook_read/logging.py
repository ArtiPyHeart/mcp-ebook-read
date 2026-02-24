"""Structured logging utilities (stderr only)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from uuid import uuid4


class JsonFormatter(logging.Formatter):
    """Write logs as json objects to stderr."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            payload["trace_id"] = record.trace_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Initialize root logger once."""
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.setLevel(level)
    root.addHandler(handler)


def make_trace_id() -> str:
    """Generate tool invocation trace id."""
    return uuid4().hex
