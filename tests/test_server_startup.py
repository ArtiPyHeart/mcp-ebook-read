from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read import server


def test_server_instructions_pin_explore_first_routing() -> None:
    assert "library_explore" in server.mcp.instructions
    assert "document_explore" in server.mcp.instructions
    assert "document_node" in server.mcp.instructions
    assert "No Qdrant" in server.mcp.instructions


def test_cli_entry_startup_failure_outputs_structured_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise AppError(
            ErrorCode.STARTUP_CONFIG_INVALID,
            "Startup configuration failed",
            details={
                "required_env": {},
                "optional_env": {"GROBID_URL": "http://127.0.0.1:8070"},
            },
        )

    monkeypatch.setattr(server.AppService, "from_env", _raise)
    monkeypatch.setattr(server.mcp, "run", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc:
        server.cli_entry()

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert "STARTUP_CONFIG_INVALID" in stderr
    assert "required_env" in stderr
    assert "GROBID_URL" in stderr


def test_cli_entry_starts_without_external_services_or_startup_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    registered: list[object] = []

    for name in (
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "GROBID_URL",
        "GROBID_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server.mcp, "run", lambda *args, **kwargs: calls.append("run"))
    monkeypatch.setattr(server.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(server, "service", None)

    server.cli_entry()

    assert calls == ["run"]
    assert registered
    assert not (tmp_path / ".mcp-ebook-read").exists()
    assert server.service is not None
    server.service.close()
    server.service = None


def test_package_startup_smoke_without_env_or_sidecar(tmp_path: Path) -> None:
    env = os.environ.copy()
    for name in (
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "GROBID_URL",
        "GROBID_TIMEOUT_SECONDS",
    ):
        env.pop(name, None)
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = (
        str(src_dir)
        if not env.get("PYTHONPATH")
        else f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = """
from pathlib import Path
from mcp_ebook_read.service import AppService

service = AppService.from_env()
try:
    print(service.__class__.__name__)
    assert not (Path.cwd() / ".mcp-ebook-read").exists()
finally:
    service.close()
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "AppService" in completed.stdout
    assert not (tmp_path / ".mcp-ebook-read").exists()


def test_cli_entry_registers_shutdown_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    registered: list[object] = []

    class FakeService:
        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(server.AppService, "from_env", lambda: FakeService())
    monkeypatch.setattr(server.mcp, "run", lambda *args, **kwargs: calls.append("run"))
    monkeypatch.setattr(server.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(server, "service", None)

    server.cli_entry()

    assert calls == ["run"]
    assert len(registered) == 1
    registered[0]()
    assert calls == ["run", "close"]
    assert server.service is None
