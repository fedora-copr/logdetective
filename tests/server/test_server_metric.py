import datetime
from typing import Callable

import pytest
import aiohttp
import aioresponses

from flexmock import flexmock

from logdetective.server.database.models import AnalyzeRequestMetrics, EndpointType
from logdetective.server.models import Explanation, TimePeriod, MetricTimeSeries
from logdetective.server.metric import (
    track_request,
    create_time_series_arrays,
    requests_per_time,
    average_time_per_responses,
    emojis_per_time,
    TimeDefinition
)

from tests.server.test_helpers import build_log, mock_AnalyzeRequestMetrics, PopulateDatabase


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


def test_week_Definition():
    time_def = TimeDefinition(TimePeriod(weeks=3))
    assert time_def.days_diff == 21


def test_day_Definition():
    time_def = TimeDefinition(TimePeriod(days=3))
    assert time_def.days_diff == 3


def test_hour_Definition():
    time_def = TimeDefinition(TimePeriod(hours=3))
    assert time_def.days_diff == 0


@pytest.mark.parametrize(
    "endpoint",
    [EndpointType.ANALYZE, EndpointType.ANALYZE_STAGED],
)
@pytest.mark.asyncio
async def test_create_time_series_arrays(endpoint):
    duration = datetime.timedelta(hours=15)
    async with PopulateDatabase.populate_db(
        duration=duration,
        endpoint=endpoint,
    ) as _:
        period = TimePeriod(hours=22)
        time_def = TimeDefinition(period)
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = period.get_period_start_time(end_time)
        counts_dict = await AnalyzeRequestMetrics.get_requests_in_period(
            start_time, end_time, time_def.time_format, endpoint
        )
        timestamps, counts = create_time_series_arrays(
            counts_dict,
            time_def,
            start_time,
            end_time,
        )
        assert len(timestamps) == len(counts) == 22 + 1
        assert (
            sum(counts) < 22 * 4
        )  # since we have added requests just for the last 15 hours


@pytest.mark.parametrize(
    "end_time",
    [None, datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)]
)
@pytest.mark.asyncio
async def test_get_period_start_time(end_time):
    """Test that start time retrieval works with and without set `end_time`."""
    period = TimePeriod(hours=22)
    start_time = period.get_period_start_time(end_time)

    if end_time:
        assert start_time == end_time - period.get_time_period()
    else:
        assert start_time <= datetime.datetime.now(datetime.timezone.utc) - period.get_time_period()


async def _test_stats(
    duration: datetime.timedelta,
    period: TimePeriod,
    stats_function: Callable,
    endpoint: EndpointType,
) -> list[MetricTimeSeries]:
    if stats_function is emojis_per_time:
        async with PopulateDatabase.populate_db_with_emojis(
            duration=duration,
        ) as _:
            stats = await stats_function(period)

    else:
        async with PopulateDatabase.populate_db(
            duration=duration,
            endpoint=endpoint,
        ) as _:
            stats = [await stats_function(period, endpoint)]
    assert isinstance(stats, list)
    for e in stats:
        assert isinstance(e, MetricTimeSeries)

    return stats


@pytest.mark.parametrize(
    "endpoint,stats_function",
    [
        pytest.param(
            EndpointType.ANALYZE,
            requests_per_time,
            id="Requests per time for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE_STAGED,
            requests_per_time,
            id="Requests per time for ANALYZE_STAGED endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE,
            average_time_per_responses,
            id="average time and length for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE_STAGED,
            average_time_per_responses,
            id="average time and length for ANALYZE_STAGED endpoint",
        ),
        pytest.param(
            None,
            emojis_per_time,
            id="emoji feedback",
        ),
    ],
)
@pytest.mark.asyncio
async def test_hourly_stats(endpoint: EndpointType, stats_function: Callable):
    duration = datetime.timedelta(hours=14)
    period = TimePeriod(hours=22)
    await _test_stats(duration, period, stats_function, endpoint)


@pytest.mark.parametrize(
    "endpoint,stats_function",
    [
        pytest.param(
            EndpointType.ANALYZE,
            requests_per_time,
            id="Requests per time for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE_STAGED,
            requests_per_time,
            id="Requests per time for ANALYZE_STAGED endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE,
            average_time_per_responses,
            id="average time and length for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE_STAGED,
            average_time_per_responses,
            id="average time and length for ANALYZE_STAGED endpoint",
        ),
        pytest.param(
            None,
            emojis_per_time,
            id="emoji feedback",
        ),
    ],
)
@pytest.mark.asyncio
async def test_daily_stats(endpoint: EndpointType, stats_function: Callable):
    duration = datetime.timedelta(days=9)
    period = TimePeriod(days=15)
    await _test_stats(duration, period, stats_function, endpoint)


@pytest.mark.parametrize(
    "endpoint,stats_function",
    [
        pytest.param(
            EndpointType.ANALYZE,
            requests_per_time,
            id="Requests per time for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE_STAGED,
            requests_per_time,
            id="Requests per time for ANALYZE_STAGED endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE,
            average_time_per_responses,
            id="average time and length for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE_STAGED,
            average_time_per_responses,
            id="average time and length for ANALYZE_STAGED endpoint",
        ),
        pytest.param(
            None,
            emojis_per_time,
            id="emoji feedback",
        ),
    ],
)
@pytest.mark.asyncio
async def test_weekly_stats(endpoint: EndpointType, stats_function: Callable):
    duration = datetime.timedelta(weeks=3)
    period = TimePeriod(weeks=5)
    await _test_stats(duration, period, stats_function, endpoint)
