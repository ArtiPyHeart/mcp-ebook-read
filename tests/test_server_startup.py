from __future__ import annotations

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read import server


def test_cli_entry_startup_failure_outputs_structured_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise AppError(
            ErrorCode.STARTUP_DEPENDENCY_NOT_READY,
            "Startup preflight failed",
            details={
                "required_env": {
                    "QDRANT_URL": "http://127.0.0.1:6333",
                    "GROBID_URL": "http://127.0.0.1:8070",
                }
            },
        )

    monkeypatch.setattr(server.AppService, "from_env", _raise)
    monkeypatch.setattr(server.mcp, "run", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc:
        server.cli_entry()

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert "STARTUP_DEPENDENCY_NOT_READY" in stderr
    assert "QDRANT_URL" in stderr
    assert "GROBID_URL" in stderr
