from __future__ import annotations

from mcp_ebook_read.errors import AppError, ErrorCode, to_error_payload


def test_to_error_payload_from_app_error() -> None:
    payload = to_error_payload(
        AppError(
            ErrorCode.SCAN_INVALID_ROOT,
            "invalid root",
            details={"root": "/tmp/missing"},
        )
    )

    assert payload == {
        "code": ErrorCode.SCAN_INVALID_ROOT,
        "message": "invalid root",
        "details": {"root": "/tmp/missing"},
    }


def test_to_error_payload_from_internal_error() -> None:
    payload = to_error_payload(RuntimeError("boom"))

    assert payload == {
        "code": "INTERNAL_ERROR",
        "message": "boom",
        "details": None,
    }
