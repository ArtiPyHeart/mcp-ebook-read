"""Network helpers for local-vs-remote routing decisions."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

LOCAL_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "host.docker.internal",
}

LOCAL_DOMAIN_SUFFIXES = (
    ".local",
    ".lan",
    ".home.arpa",
    ".localdomain",
    ".internal",
    ".docker.internal",
)


def should_trust_env_proxy(target_url: str | None) -> bool:
    """Return False for local endpoints so requests bypass system proxies."""
    host = _extract_hostname(target_url)
    if not host:
        return True
    return not _is_local_host(host)


def _extract_hostname(target_url: str | None) -> str | None:
    if not target_url:
        return None
    value = target_url.strip()
    if not value:
        return None
    parsed = urlparse(value if "://" in value else f"//{value}", scheme="http")
    return parsed.hostname


def _is_local_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized in LOCAL_HOSTS:
        return True
    if normalized.endswith(LOCAL_DOMAIN_SUFFIXES):
        return True

    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False

    return ip.is_loopback or ip.is_private or ip.is_link_local
