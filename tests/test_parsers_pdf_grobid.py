from __future__ import annotations

from pathlib import Path

import pytest

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.parsers.pdf_grobid import GrobidClient


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeHttpxClient:
    def __init__(
        self,
        *,
        get_response: FakeResponse | None = None,
        post_response: FakeResponse | None = None,
        fail_get: bool = False,
        fail_post: bool = False,
        **_kwargs: object,
    ) -> None:
        self.get_response = get_response or FakeResponse("true")
        self.post_response = post_response or FakeResponse(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"></TEI>'
        )
        self.fail_get = fail_get
        self.fail_post = fail_post

    def __enter__(self) -> "FakeHttpxClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def get(self, _url: str) -> FakeResponse:
        if self.fail_get:
            raise RuntimeError("cannot connect")
        return self.get_response

    def post(
        self, _url: str, files: dict[str, tuple[str, object, str]]
    ) -> FakeResponse:  # noqa: ARG002
        if self.fail_post:
            raise RuntimeError("post failed")
        return self.post_response


def test_assert_available_requires_url() -> None:
    client = GrobidClient(base_url=None)
    with pytest.raises(AppError) as exc:
        client._assert_available()
    assert exc.value.code == ErrorCode.INGEST_PAPER_GROBID_UNAVAILABLE


def test_assert_available_connect_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_grobid.httpx.Client",
        lambda **kwargs: FakeHttpxClient(fail_get=True, **kwargs),
    )
    client = GrobidClient(base_url="http://localhost:8070")

    with pytest.raises(AppError) as exc:
        client._assert_available()

    assert exc.value.code == ErrorCode.INGEST_PAPER_GROBID_UNAVAILABLE


def test_local_base_url_disables_env_proxy() -> None:
    client = GrobidClient(base_url="http://localhost:8070")
    assert client._trust_env_proxy is False


def test_remote_base_url_keeps_env_proxy() -> None:
    client = GrobidClient(base_url="https://grobid.example.com")
    assert client._trust_env_proxy is True


def test_assert_available_passes_trust_env_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def _factory(**kwargs: object) -> FakeHttpxClient:
        captured.append(kwargs)
        return FakeHttpxClient(**kwargs)

    monkeypatch.setattr("mcp_ebook_read.parsers.pdf_grobid.httpx.Client", _factory)
    client = GrobidClient(base_url="http://127.0.0.1:8070")
    client._assert_available()

    assert captured
    assert captured[0].get("trust_env") is False


def test_parse_fulltext_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_grobid.httpx.Client",
        lambda **kwargs: FakeHttpxClient(**kwargs),
    )
    client = GrobidClient(base_url="http://localhost:8070")

    missing = tmp_path / "missing.pdf"
    with pytest.raises(AppError) as exc:
        client.parse_fulltext(str(missing))

    assert exc.value.code == ErrorCode.INGEST_DOC_NOT_FOUND


def test_parse_fulltext_request_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mcp_ebook_read.parsers.pdf_grobid.httpx.Client",
        lambda **kwargs: FakeHttpxClient(fail_post=True, **kwargs),
    )
    client = GrobidClient(base_url="http://localhost:8070")
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"pdf")

    with pytest.raises(AppError) as exc:
        client.parse_fulltext(str(pdf))

    assert exc.value.code == ErrorCode.INGEST_PAPER_GROBID_FAILED


def test_parse_tei_success() -> None:
    client = GrobidClient(base_url="http://localhost:8070")
    tei = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt><title>My Paper</title></titleStmt>
        </fileDesc>
      </teiHeader>
      <profileDesc><abstract>Abstract text.</abstract></profileDesc>
      <sourceDesc><biblStruct><idno type="DOI">10.1000/xyz</idno></biblStruct></sourceDesc>
      <text>
        <body>
          <div><head>Intro</head></div>
          <div><head>Method</head></div>
        </body>
        <back>
          <listBibl>
            <biblStruct />
            <biblStruct />
          </listBibl>
        </back>
      </text>
    </TEI>
    """

    result = client._parse_tei(tei)

    assert result.metadata["paper_title"] == "My Paper"
    assert result.metadata["doi"] == "10.1000/xyz"
    assert result.metadata["bibliography_count"] == 2
    assert [node.title for node in result.outline] == ["Intro", "Method"]


def test_parse_tei_invalid_xml() -> None:
    client = GrobidClient(base_url="http://localhost:8070")

    with pytest.raises(AppError) as exc:
        client._parse_tei("<not-xml")

    assert exc.value.code == ErrorCode.INGEST_PAPER_GROBID_FAILED
