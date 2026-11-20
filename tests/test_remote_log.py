"""Remote decoding failures retain accounting and stay local to suite artifacts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from logdetective.artifacts import fetch_suite_artifacts
from logdetective.config import SERVER_CONFIG
from logdetective.exceptions import RemoteLogAccessError
from logdetective.models import TestingFarmSuiteRequest as SuiteRequest
from logdetective.remote_log import RemoteLog


def response_body(charset, body):
    """Supply streamed response bytes without replacing the real decoder."""
    async def chunks(_size):
        yield body

    return SimpleNamespace(charset=charset, content=SimpleNamespace(iter_chunked=chunks))


@pytest.mark.asyncio
@pytest.mark.parametrize("charset", ["x-invalid", "base64"])
async def test_unsupported_charset_retains_byte_count(charset):
    """Unknown and non-text codecs become remote access errors after accounting."""
    remote = RemoteLog("https://example.org/log", MagicMock())
    with pytest.raises(RemoteLogAccessError, match="Unsupported remote log charset") as error:
        await remote._read_with_size_limit(response_body(charset, b"fail"))
    assert isinstance(error.value.__cause__, LookupError)
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
        charset = "x-invalid" if remote.url.endswith("bad") else "utf-8"
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
