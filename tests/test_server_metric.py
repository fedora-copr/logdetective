import datetime
from unittest.mock import AsyncMock

import pytest
import aiohttp
import aioresponses
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from flexmock import flexmock

from logdetective.database.models import EndpointType, TimePeriod
from logdetective.config import SERVER_CONFIG
from logdetective.models import Explanation, MetricsData
from logdetective.metric import (
    track_request,
    requests_statistics,
)
from logdetective.server import app

from tests.test_helpers import (
    build_log_request,
    build_log_url,
    build_log_one_file,
    build_log_two_files,
    mock_AnalyzeRequestMetrics,
    PopulateDatabase,
)


@pytest.mark.parametrize(
    "build_log_request",
    ["build_log_url", "build_log_one_file", "build_log_two_files"],
    indirect=True,
)
@pytest.mark.parametrize(
    "response",
    [
        flexmock(
            explanation=Explanation(text="abc"),
            model_dump_json=lambda: "{explanation: 'abc'}",
        ),
        flexmock(),  # mimic StreamResponse
    ],
)
@pytest.mark.asyncio
async def test_track_request_async(
    build_log_request, mock_AnalyzeRequestMetrics, response
):
    """Test the @track_request decorator for a mock analyze log function call."""

    @track_request()
    async def analyze(payload, http_session, request=None):
        return response

    mock_header = {"Content-Length": "3"}
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.head("https://example.com/logs/123", status=200, headers=mock_header)
        mock.get("https://example.com/logs/123", status=200, body=mock_response)
        async with aiohttp.ClientSession() as session:
            request = Request({"type": "http"})
            request.state.api_token_name = "packit"
            await analyze(**build_log_request, http_session=session, request=request)
    mock_create = mock_AnalyzeRequestMetrics["mock_create"]
    mock_update = mock_AnalyzeRequestMetrics["mock_update"]

    create_kwargs = mock_create.await_args.kwargs
    update_kwargs = mock_update.await_args.kwargs

    # Verify that endpoint is set to `EndpointType.ANALYZE`
    assert create_kwargs["endpoint"] == EndpointType.ANALYZE
    assert create_kwargs["api_token_name"] == "packit"

    # value of _id used in calling `update` method must match
    # value returned by `create` method
    assert update_kwargs["id_"] == 1

    # Verify type of time stamp
    assert isinstance(update_kwargs["response_sent_at"], datetime.datetime)

    # Verify value of response length
    if getattr(response, "explanation", None):
        assert update_kwargs["response_length"] == len(response.model_dump_json())
    else:
        assert update_kwargs["response_length"] is None


def test_track_request_rejects_synchronous_functions():
    """Metric tracking only supports the async endpoints it was designed for."""
    def analyze():
        return None

    with pytest.raises(NotImplementedError, match="async coroutine"):
        track_request()(analyze)


@pytest.mark.asyncio
async def test_requests_statistics_defaults_to_current_time(mocker):
    """Omitting end_time queries through now and returns an empty shaped model."""
    get_stats = mocker.patch(
        "logdetective.metric.AnalyzeRequestMetrics.get_requests_stats_for_period",
        new_callable=AsyncMock,
        return_value=[[], [], [], [], []],
    )
    start_time = datetime.datetime(2077, 1, 1, tzinfo=datetime.timezone.utc)
    before = datetime.datetime.now(datetime.timezone.utc)

    stats = await requests_statistics(start_time=start_time)

    after = datetime.datetime.now(datetime.timezone.utc)
    assert stats == MetricsData(
        endpoint=EndpointType.ANALYZE.value,
        period_start=[],
        total_count=[],
        average_response_time=[],
        average_response_len=[],
        average_completion_time=[],
    )
    call_kwargs = get_stats.await_args.kwargs
    assert before <= call_kwargs["end_time"] <= after
    assert call_kwargs == {
        "start_time": start_time,
        "end_time": call_kwargs["end_time"],
        "time_period": TimePeriod.DAY,
        "endpoint": EndpointType.ANALYZE,
        "api_token_name": None,
    }


@pytest.mark.parametrize(
    "route, endpoint",
    [
        ("analyze", EndpointType.ANALYZE),
        ("analyze-gitlab", EndpointType.ANALYZE_GITLAB_JOB),
    ],
)
@pytest.mark.asyncio
async def test_metrics_endpoint_returns_columnar_statistics(
    route, endpoint, mocker, monkeypatch
):
    """The HTTP endpoint parses filters and exposes every metrics column."""
    monkeypatch.setattr(SERVER_CONFIG.gitlab, "instances", {"configured": object()})
    period_start = datetime.datetime(2077, 1, 1, tzinfo=datetime.timezone.utc)
    end_time = period_start + datetime.timedelta(days=1)
    data = MetricsData(
        endpoint=endpoint.value,
        period_start=[period_start],
        total_count=[2],
        average_response_time=[3.5],
        average_response_len=[200.0],
        average_completion_time=[2.5],
    )
    get_stats = mocker.patch(
        "logdetective.server.requests_statistics",
        new_callable=AsyncMock,
        return_value=data,
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/metrics/{route}/",
            params={
                "start_time": period_start.isoformat(),
                "end_time": end_time.isoformat(),
                "time_period": TimePeriod.HOUR.value,
                "api_token_name": "packit",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "metrics": [
            {
                "endpoint": endpoint.value,
                "period_start": ["2077-01-01T00:00:00Z"],
                "total_count": [2],
                "average_response_time": [3.5],
                "average_response_len": [200.0],
                "average_completion_time": [2.5],
            }
        ]
    }
    get_stats.assert_awaited_once_with(
        start_time=period_start,
        end_time=end_time,
        time_period=TimePeriod.HOUR,
        endpoint=endpoint,
        api_token_name="packit",
    )


@pytest.mark.asyncio
async def test_gitlab_metrics_endpoint_rejects_missing_configuration(
    mocker, monkeypatch
):
    """GitLab metrics are unavailable when no GitLab instance is configured."""
    monkeypatch.setattr(SERVER_CONFIG.gitlab, "instances", {})
    get_stats = mocker.patch(
        "logdetective.server.requests_statistics", new_callable=AsyncMock
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/metrics/analyze-gitlab/",
            params={
                "start_time": "2077-01-01T00:00:00Z",
                "time_period": "day",
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No gitlab instance configured, skipping metrics collection."
    }
    get_stats.assert_not_awaited()


@pytest.mark.parametrize(
    "route, params",
    [
        ("invalid", {"start_time": "2077-01-01T00:00:00Z", "time_period": "day"}),
        ("analyze", {"time_period": "day"}),
        ("analyze", {"start_time": "not-a-date", "time_period": "day"}),
        ("analyze", {"start_time": "2077-01-01T00:00:00Z", "time_period": "week"}),
    ],
)
@pytest.mark.asyncio
async def test_metrics_endpoint_rejects_invalid_parameters(route, params):
    """Route, timestamp, and aggregation period inputs are validated by FastAPI."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/metrics/{route}/", params=params)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "endpoint",
    [
        pytest.param(
            EndpointType.ANALYZE,
            id="Requests stats for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE_KOJI_TASK,
            id="Average stats for ANALYZE_KOJI_TASK endpoint",
        ),
    ],
)
@pytest.mark.parametrize(
    "time_period, records",
    [
        pytest.param(
            TimePeriod.HOUR,
            [
                (datetime.timedelta(minutes=10), 1.0),
                (datetime.timedelta(minutes=40), 2.0),
                (datetime.timedelta(hours=4), 3.0),
                (datetime.timedelta(hours=8, minutes=5), 3.5),
                (datetime.timedelta(hours=8, minutes=50), 4.5),
                (datetime.timedelta(hours=12, minutes=59), 1.0),
                (datetime.timedelta(hours=16, minutes=1), 1.0),
                (datetime.timedelta(hours=23), 5.0),  # ignored
            ],
            id="hourly",
        ),
        pytest.param(
            TimePeriod.DAY,
            [
                (datetime.timedelta(days=0, hours=2), 1.0),
                (datetime.timedelta(days=3, hours=1), 2.5),
                (datetime.timedelta(days=3, hours=6), 2.0),
                (datetime.timedelta(days=3, hours=14), 1.5),
                (datetime.timedelta(days=5, hours=23), 2.5),
                (datetime.timedelta(days=8, hours=12), 4.0),
                (datetime.timedelta(days=10, hours=2), 3.0),  # ignored
            ],
            id="daily",
        ),
        pytest.param(
            TimePeriod.MONTH,
            [
                (datetime.timedelta(days=1, hours=5), 1.0),
                (datetime.timedelta(days=1, hours=20), 3.0),
                (datetime.timedelta(days=15), 2.0),
                (datetime.timedelta(days=28, hours=12), 1.5),
                (datetime.timedelta(days=42, hours=3), 2.5),
                (datetime.timedelta(days=42, hours=18), 4.0),
                (datetime.timedelta(days=65, hours=1), 1.0),
                (datetime.timedelta(days=80, hours=16), 2.5),  # ignored
            ],
            id="monthly",
        ),
    ],
)
@pytest.mark.asyncio
async def test_request_stats(
    endpoint: EndpointType,
    time_period: TimePeriod,
    records: list[tuple[datetime.timedelta, float]],
):
    """
    Populate DB with a some preset mock transaction metadata (`records`) and check that
    they are selected (only the selected `period`) and aggregated (`stats_function`) properly.
    """
    # `anchor` refers to the last full-hour (X:00:00), or midnight,
    # for more deterministic bucket testing.
    anchor = datetime.datetime(
        year=2077,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        tzinfo=datetime.timezone.utc,
    )
    if time_period in [TimePeriod.DAY, TimePeriod.MONTH]:
        anchor = anchor.replace(hour=0)

    start_time = anchor - (records[-1][0])
    assert start_time < anchor, start_time

    async with PopulateDatabase.populate_db_with_analysis_records(
        time_anchor=anchor, records=records, endpoint=endpoint
    ) as _:
        stats = await requests_statistics(
            start_time=start_time,
            end_time=anchor,
            endpoint=endpoint,
            time_period=time_period,
        )

    assert len(stats.period_start) == len(stats.average_response_time) > 0
    assert len(stats.average_response_len) == len(stats.average_response_time)
    assert len(stats.total_count) == len(stats.average_response_len)

    # We only use .0, .5, and .25 in the mock data
    # so that we can do exact comparisons with floats
    if time_period == TimePeriod.HOUR:
        assert stats.total_count == [1, 1, 2, 1, 2]
        assert stats.average_response_time == [1.0, 1.0, 4.0, 3.0, 1.5]
    elif time_period == TimePeriod.DAY:
        assert stats.total_count == [1, 1, 3, 1]
        assert stats.average_response_time == [4.0, 2.5, 2.0, 1.0]
    elif time_period == TimePeriod.MONTH:
        assert stats.total_count == [1, 2, 4]
        assert stats.average_response_time == [1.0, 3.25, 1.875]
    else:
        msg = (
            "Did not test any of the expected checks, "
            f"period={time_period}, endpoint={endpoint}, stats={stats}"
        )
        assert False, msg
