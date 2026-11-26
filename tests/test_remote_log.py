"""Remote log retrieval, size accounting, and decoding."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
from aiointercept import aiointercept
from multidict import CIMultiDict
import pytest

from logdetective.artifacts import fetch_suite_artifacts
from logdetective.config import SERVER_CONFIG
from logdetective.constants import DEFAULT_MAXIMUM_ARTIFACT_MIB
from logdetective.exceptions import (
    RemoteLogAccessError,
    RemoteLogHeaderError,
    RemoteLogRequestError,
    RemoteLogTooLargeError,
)
from logdetective.models import TestingFarmSuiteRequest as SuiteRequest
from logdetective.remote_log import RemoteLog


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, mock_header, mock_body, limit_bytes, exc_type, exc_match",
    [
        ("http://example.com/build.log", {"Content-Length": "3"}, "123", None, None, None),
        (
            "http://example.com/build.log",
            {"Content-Length": f"{(DEFAULT_MAXIMUM_ARTIFACT_MIB) * 1024**2 + 1}"},
            "123",
            None,
            RemoteLogTooLargeError,
            "Content-Length is over the limit",
        ),
        ("not-a-valid-url", {}, "123", None, RemoteLogRequestError, "Invalid log URL"),
        # No Content-Length: small body succeeds
        ("http://example.com/build.log", {}, "123", None, None, None),
        # No Content-Length: body over limit is rejected while reading
        (
            "http://example.com/build.log",
            {},
            "x" * 20,
            10,
            RemoteLogTooLargeError,
            "exceeds the limit",
        ),
        # Lying Content-Length: declared within limit, actual body over limit
        (
            "http://example.com/build.log",
            {"Content-Length": "3"},
            "x" * 20,
            10,
            RemoteLogTooLargeError,
            "exceeds the limit",
        ),
    ],
    indirect=False,
)
# pylint: disable=too-many-arguments,too-many-positional-arguments
async def test_get_url_content(
    url,
    mock_header,
    mock_body,
    limit_bytes,
    exc_type,
    exc_match
):
    """Test various URL requests and correct Exceptions during RemoteLog access."""
    async with aiointercept(mock_external_urls=True) as mock:
        # Using invalid URL with aiointercept raises ValueError.
        # The exception is raised before the request is made,
        # so we don't need the full mock.
        if exc_match != "Invalid log URL":
            mock.head(url, status=200, headers=mock_header)
            mock.get(url, status=200, body=mock_body)
        async with aiohttp.ClientSession() as http:
            kwargs = {"limit_bytes": limit_bytes} if limit_bytes is not None else {}
            if exc_type:
                with pytest.raises(exc_type, match=exc_match):
                    remote_log = RemoteLog(url, http, **kwargs)
                    await remote_log.get_url_content()
            else:
                remote_log = RemoteLog(url, http, **kwargs)
                url_output = await remote_log.get_url_content()
                assert url_output == mock_body


@pytest.mark.asyncio
async def test_get_url_content_invalid_content_length():
    """An invalid header reaches the size check, even though aiohttp cannot serve it."""
    url = "http://example.com/build.log"
    http = Mock()
    http.head = AsyncMock(
        return_value=SimpleNamespace(headers=CIMultiDict({"Content-Length": "test"}))
    )

    with pytest.raises(RemoteLogHeaderError, match="Content-Length header is invalid"):
        await RemoteLog(url, http).get_url_content()

    http.head.assert_awaited_once_with(url, raise_for_status=True)
    http.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "head_status, get_status",
    [(404, 200), (500, 200), (503, 200), (200, 404), (200, 500), (200, 503)],
    indirect=False,
)
async def test_get_url_content_connection_fails(head_status, get_status):
    """Test HEAD/GET failures during RemoteLog access. All should raise RemoteLogAccessError."""
    url = "http://example.com/build.log"
    mock_head_response = {"Content-Length": "11"} if head_status <= 399 else None
    mock_get_response = "Lorem Ipsum"
    async with aiointercept(mock_external_urls=True) as mock:
        mock.head(url, status=head_status, headers=mock_head_response)
        mock.get(url, status=get_status, body=mock_get_response)
        async with aiohttp.ClientSession() as http:
            remote_log = RemoteLog(url, http)
            with pytest.raises(RemoteLogAccessError):
                await remote_log.get_url_content()


def response_body(charset, body):
    """Supply streamed response bytes without replacing the real decoder."""
    async def chunks(_size):
        yield body

    return SimpleNamespace(charset=charset, content=SimpleNamespace(iter_chunked=chunks))


@pytest.mark.asyncio
@pytest.mark.parametrize(("charset", "cause"), [
    ("x-invalid", LookupError), ("base64", LookupError), ("idna", UnicodeError),
])
async def test_unsupported_charset_retains_byte_count(charset, cause):
    """Unsupported codecs or error handlers become access errors after accounting."""
    remote = RemoteLog("https://example.org/log", MagicMock())
    with pytest.raises(RemoteLogAccessError, match="Unsupported remote log charset") as error:
        await remote._read_with_size_limit(response_body(charset, b"fail"))
    assert isinstance(error.value.__cause__, cause)
    assert remote.remote_log_size == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("charset,body,expected", [
    (None, b"caf\xc3\xa9", "café"),
    ("iso-8859-1", b"caf\xe9", "café"),
    ("utf-8", b"bad\xff", "bad\ufffd"),
])
async def test_supported_charset_decoding_is_unchanged(charset, body, expected):
    """Declared encodings, UTF-8 fallback, and replacement decoding still work."""
    remote = RemoteLog("https://example.org/log", MagicMock())
    assert await remote._read_with_size_limit(response_body(charset, body)) == expected
    assert remote.remote_log_size == len(body)


@pytest.mark.asyncio
async def test_suite_charset_failure_is_cached_scoped_and_charged(monkeypatch):
    """Bad decoding warns per reference, charges once, and preserves sibling logs."""
    downloads = []

    async def download(remote):
        downloads.append((remote.url, remote._limit_bytes))
        charset = "idna" if remote.url.endswith("bad") else "utf-8"
        return await remote._read_with_size_limit(response_body(charset, b"fail"))

    monkeypatch.setattr(RemoteLog, "get_url_content", download)
    monkeypatch.setattr(SERVER_CONFIG.general, "max_artifact_size", 200)
    payload = SuiteRequest(suite={"plan": "/plan"}, cases=[
        {"case_id": case_id, "name": "/test", "result": "failed", "artifacts": [
            {"name": "log", "url": f"https://example.org/{source}"},
        ]}
        for case_id, source in [("bad1", "bad"), ("bad2", "bad"), ("good", "good")]
    ])
    result = await fetch_suite_artifacts(payload, 100)
    assert downloads == [("https://example.org/bad", 100), ("https://example.org/good", 96)]
    assert result.artifacts == {"case/good/log": "fail"}
    assert [case.case_id for case in result.eligible] == ["good"]
    assert set(result.errors) == {"bad1", "bad2"}
    assert all(outcome.error.code == "artifact_error" for outcome in result.errors.values())
    assert [warning.source_file for warning in result.warnings] == [
        "case/bad1/log", "case/bad2/log",
    ]
    assert all(warning.code == "artifact_unavailable" for warning in result.warnings)
