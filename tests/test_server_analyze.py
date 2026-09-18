import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from logdetective.server import app, validate_request_size


@pytest_asyncio.fixture
async def test_client():
    """Mocking AsyncClient for sending and checking requests."""

    async def no_request_size_limit():
        return None

    app.dependency_overrides[validate_request_size] = no_request_size_limit

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost:8080"
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_body",
    [
        {"usb": "http://example.com/build.log"},
        {"url": "not-a-valid-url-for-testing"},
        {"url": "http://example.com/build.log"},
    ],
)
async def test_analyze_rejects_obsolete_request_shapes(test_client, request_body):
    """The async API accepts only the documented artifact-list request model."""
    response = await test_client.post("/analyze", json=request_body)

    assert response.status_code == 422
