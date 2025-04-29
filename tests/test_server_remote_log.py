import pytest
import aiohttp
import aioresponses

from logdetective.server.remote_log import RemoteLog


@pytest.mark.asyncio
async def test_server_remote_log():
    http_session = aiohttp.ClientSession()

    mock_response = "Warning: Permanently added"
    with aioresponses.aioresponses() as mock:
        mock.get(
            "https://example.com/logs/123", status=200, body=mock_response, repeat=True
        )

        remote_log = RemoteLog("https://example.com/logs/123", http_session)
        content = await remote_log.content
        assert content
        assert mock_response in content
        zip_data = await remote_log.zip_content
        url_text = remote_log.unzip(zip_data)
        assert url_text
        assert mock_response in url_text
