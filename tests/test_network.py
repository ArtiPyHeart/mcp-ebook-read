from __future__ import annotations

from mcp_ebook_read.network import should_trust_env_proxy


def test_should_not_trust_env_proxy_for_local_targets() -> None:
    local_targets = [
        "http://localhost:6333",
        "http://127.0.0.1:8070",
        "http://[::1]:8070",
        "http://host.docker.internal:6333",
        "http://service.internal",
        "http://box.local",
        "http://192.168.1.10:6333",
        "http://10.0.0.2",
    ]

    for target in local_targets:
        assert should_trust_env_proxy(target) is False


def test_should_trust_env_proxy_for_remote_targets() -> None:
    remote_targets = [
        "https://qdrant.example.com",
        "http://grobid.example.org",
        "https://api.openai.com",
    ]

    for target in remote_targets:
        assert should_trust_env_proxy(target) is True
