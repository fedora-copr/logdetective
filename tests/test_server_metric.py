import pytest
import aiohttp
import aioresponses

from flexmock import flexmock
from sqlalchemy.orm import session

from logdetective.server.database.models import AnalyzeRequestMetrics
from logdetective.server.metric import track_request


@pytest.fixture
def build_log():
    return {"build_log": flexmock(url="https://example.com/logs/123")}


@pytest.fixture
def mock_AnalyzeRequestMetrics():
    update_response_metrics = flexmock()
    all_metrics = (
        flexmock().should_receive("first").and_return(update_response_metrics).mock()
    )
    query = flexmock().should_receive("filter_by").and_return(all_metrics).mock()
    flexmock(session.Session).should_receive("query").and_return(query)
    flexmock(session.Session).should_receive("add").and_return()
    flexmock(AnalyzeRequestMetrics).should_receive("create").once().and_return(1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        flexmock(response_certainty=37.7, explanation=[{"text": "abc"}]),
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
