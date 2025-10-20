import datetime

import pytest
import aiohttp
import aioresponses

from flexmock import flexmock

from logdetective.server.models import Explanation
from logdetective.server.metric import track_request, EndpointType

from tests.server.test_helpers import build_log, mock_AnalyzeRequestMetrics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        flexmock(
            response_certainty=37.7, explanation=Explanation(text="abc", logprobs=[])
        ),
        flexmock(),  # mimic StreamResponse
    ],
)
async def test_track_request_async(build_log, mock_AnalyzeRequestMetrics, response):
    @track_request()
    async def analyze_log(build_log, http_session):
        return response

    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.get("https://example.com/logs/123", status=200, body=mock_response)
        await analyze_log(**build_log, http_session=aiohttp.ClientSession())

    mock_create = mock_AnalyzeRequestMetrics["mock_create"]
    mock_update = mock_AnalyzeRequestMetrics["mock_update"]

    create_kwargs = mock_create.await_args.kwargs
    update_kwargs = mock_update.await_args.kwargs

    # Verify that time stamp is set
    assert isinstance(create_kwargs["request_received_at"], datetime.datetime)

    # Verify that endpoint is set to `EndpointType.ANALYZE`
    assert create_kwargs["endpoint"] == EndpointType.ANALYZE

    # Verify presence, type and contents of compressed log
    assert "compressed_log" in create_kwargs
    assert isinstance(create_kwargs["compressed_log"], bytes)
    assert len(create_kwargs["compressed_log"]) > 0

    # value of _id used in calling `update` method must match
    # value returned by `create` method
    assert update_kwargs["id_"] == 1

    # Verify type of time stamp
    assert isinstance(update_kwargs["response_sent_at"], datetime.datetime)

    # Verify value of 'response_certainty'
    assert update_kwargs["response_certainty"] == getattr(
        response, "response_certainty", None
    )

    # Verify value of response length
    if explanation := getattr(response, "explanation", None):
        assert update_kwargs["response_length"] == len(explanation.text)
