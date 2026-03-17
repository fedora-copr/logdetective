import pytest
import pytest_asyncio

import aiohttp
import aioresponses
from httpx import ASGITransport, AsyncClient

from logdetective.server.server import app, get_http_session
from logdetective.server.utils import validate_request_size, SERVER_CONFIG
from logdetective.utils import mib_to_bytes

from tests.server.test_helpers import mock_AnalyzeRequestMetrics


@pytest_asyncio.fixture
async def test_client(mock_AnalyzeRequestMetrics):
    """Mocking AsyncClient for sending and checking requests."""

    async def override_get_http_session():
        async with aiohttp.ClientSession() as session:
            yield session

    app.dependency_overrides[validate_request_size] = lambda: None
    app.dependency_overrides[get_http_session] = override_get_http_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost:8080"
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_body, head_status, mock_headers, final_status",
    [
        ({"usb": "http://example.com/build.log"}, None, None, 422),
        ({"url": "not-a-valid-url-for-testing"}, None, None, 400),
        ({"url": "http://example.com/build.log"}, 404, None, 502),
        ({"url": "http://example.com/build.log"}, 500, None, 502),
        ({"url": "http://example.com/build.log"}, 503, None, 502),
        ({"url": "http://example.com/build.log"}, 200, {}, 411),
        (
            {"url": "http://example.com/build.log"},
            200,
            {"Content-Length": f"{mib_to_bytes(SERVER_CONFIG.general.max_artifact_size) + 1}"},
            413
        ),
    ],
    indirect=False
)
async def test_analyze_invalid_errors(
    test_client,
    request_body,
    head_status,
    mock_headers,
    final_status
):
    """Test various cases when submitting invalid request with log's URL fails."""
    if not head_status:
        response = await test_client.post("/analyze", json=request_body)
    else:
        with aioresponses.aioresponses() as mock:
            mock.head(request_body["url"], status=head_status, headers=mock_headers)
            response = await test_client.post("/analyze", json=request_body)

    assert response.status_code == final_status
