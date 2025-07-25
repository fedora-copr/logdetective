import pytest
import aiohttp
import aioresponses

from flexmock import flexmock
from sqlalchemy.orm import session

from logdetective.server.models import Explanation
from logdetective.server.database.models import AnalyzeRequestMetrics
from logdetective.server.metric import track_request

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
