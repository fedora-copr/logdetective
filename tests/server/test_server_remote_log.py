import pytest
import aiohttp
import aioresponses

from logdetective.remote_log import RemoteLog
from logdetective.server.compressors import RemoteLogCompressor


@pytest.mark.asyncio
async def test_server_remote_log():
    http_session = aiohttp.ClientSession()

    mock_response = "Warning: Permanently added"
    with aioresponses.aioresponses() as mock:
        mock.get(
            "https://example.com/logs/123", status=200, body=mock_response, repeat=True
        )

        remote_log = RemoteLog("https://example.com/logs/123", http_session)
        compressor = RemoteLogCompressor(remote_log)
        content = await remote_log.content
        assert content
        assert mock_response in content
        zip_data = await compressor.zip_content()
        url_text = compressor.unzip(zip_data)
        assert url_text
        assert mock_response in url_text

    await http_session.close()
