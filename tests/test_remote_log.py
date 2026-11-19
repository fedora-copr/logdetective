"""Remote decoding failures retain accounting and stay local to suite artifacts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from logdetective.exceptions import RemoteLogAccessError
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
