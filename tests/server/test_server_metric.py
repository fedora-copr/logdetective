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

from tests.server.test_helpers import (
    build_log_request,
    build_log_url,
    build_log_one_file,
    build_log_two_files,
    mock_AnalyzeRequestMetrics,
    PopulateDatabase,
)


@pytest.mark.parametrize(
    "build_log_request",
    [
        "build_log_url", "build_log_one_file", "build_log_two_files"
    ],
    indirect=True
)
@pytest.mark.parametrize(
    "response",
    [
        flexmock(
            response_certainty=37.7, explanation=Explanation(text="abc", logprobs=[])
        ),
        flexmock(),  # mimic StreamResponse
    ],
)
@pytest.mark.asyncio
async def test_track_request_async(build_log_request, mock_AnalyzeRequestMetrics, response):
    """Test the @track_request decorator for a mock analyze log function call."""
    @track_request()
    async def analyze_log(payload, http_session):
        return response

    mock_header = {"Content-Length": "3"}
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.head("https://example.com/logs/123", status=200, headers=mock_header)
        mock.get("https://example.com/logs/123", status=200, body=mock_response)
        async with aiohttp.ClientSession() as session:
            await analyze_log(**build_log_request, http_session=session)
    mock_create = mock_AnalyzeRequestMetrics["mock_create"]
    mock_update = mock_AnalyzeRequestMetrics["mock_update"]

    create_kwargs = mock_create.await_args.kwargs
    update_kwargs = mock_update.await_args.kwargs

    # Verify that endpoint is set to `EndpointType.ANALYZE`
    assert create_kwargs["endpoint"] == EndpointType.ANALYZE

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
    [EndpointType.ANALYZE],
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
        )
        assert len(timestamps) == len(counts)
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


@pytest.mark.parametrize(
    "endpoint, stats_function",
    [
        pytest.param(
            EndpointType.ANALYZE,
            requests_per_time,
            id="Requests per time for ANALYZE endpoint",
        ),
        pytest.param(
            EndpointType.ANALYZE,
            average_time_per_responses,
            id="Average response times for ANALYZE endpoint",
        ),
    ],
)
@pytest.mark.parametrize(
    "period, records",
    [
        pytest.param(
            TimePeriod(hours=14),
            [
                (datetime.timedelta(minutes=10), 1.0),
                (datetime.timedelta(minutes=40), 2.0),
                (datetime.timedelta(hours=4), 3.0),
                (datetime.timedelta(hours=8, minutes=5), 3.5),
                (datetime.timedelta(hours=8, minutes=50), 4.5),
                (datetime.timedelta(hours=12, minutes=59), 1.0),
                (datetime.timedelta(hours=16, minutes=1), 1.0),  # ignored
                (datetime.timedelta(hours=23), 5.0),  # ignored
            ],
            id="hourly",
        ),
        pytest.param(
            TimePeriod(days=9),
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
            TimePeriod(weeks=3),
            [
                (datetime.timedelta(days=1, hours=5), 1.0),
                (datetime.timedelta(days=1, hours=20), 3.0),
                (datetime.timedelta(days=5), 2.0),
                (datetime.timedelta(days=8, hours=12), 1.5),
                (datetime.timedelta(days=14, hours=3), 2.5),
                (datetime.timedelta(days=14, hours=18), 4.0),
                (datetime.timedelta(days=20, hours=1), 1.0),
                (datetime.timedelta(days=23, hours=16), 2.5),  # ignored
            ],
            id="weekly",
        ),
    ]
)
@pytest.mark.asyncio
async def test_request_stats(
    endpoint: EndpointType,
    stats_function: Callable,
    period: TimePeriod,
    records: list[tuple[datetime.timedelta, float]],
):
    """
    Populate DB with a some preset mock transaction metadata (`records`) and check that
    they are selected (only the selected `period`) and aggregated (`stats_function`) properly.
    """
    # `anchor` refers to the last full-hour (X:00:00), or midnight,
    # for more deterministic bucket testing.
    anchor = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0)
    if period.days or period.weeks:
        anchor = anchor.replace(hour=0)

    async with PopulateDatabase.populate_db_with_analysis_records(anchor, records, endpoint) as _:
        stats = await stats_function(period, endpoint, end_time=anchor)

    assert isinstance(stats, MetricTimeSeries)
    assert len(stats.timestamps) > 0
    assert len(stats.values) > 0

    # We only use .0, .5, and .25 in the mock data
    # so that we can do exact comparisons with floats
    if period.hours and stats.metric == "requests":
        assert stats.values == [1.0, 2.0, 1.0, 2.0]
    elif period.hours and stats.metric == "avg_response_time":
        assert stats.values == [1.0, 4.0, 3.0, 1.5]
    elif period.days and stats.metric == "requests":
        assert stats.values == [1.0, 1.0, 3.0, 1.0]
    elif period.days and stats.metric == "avg_response_time":
        assert stats.values == [4.0, 2.5, 2.0, 1.0]
    # Weekly stats are actually in daily buckets, just over a multi-week period.
    elif period.weeks and stats.metric == "requests":
        assert stats.values == [1.0, 2.0, 1.0, 1.0, 2.0]
    elif period.weeks and stats.metric == "avg_response_time":
        assert stats.values == [1.0, 3.25, 1.5, 2.0, 2.0]
    else:
        msg = (
            "Did not test any of the expected checks, "
            f"period={period}, endpoint={endpoint}, metric={stats.metric}"
        )
        assert False, msg


@pytest.mark.parametrize(
    "period, records",
    [
        pytest.param(
            TimePeriod(hours=14),
            [
                (datetime.timedelta(hours=1), {"thumbsup": 3, "thumbsdown": 1}),
                (datetime.timedelta(hours=1, minutes=30), {"thumbsup": 2, "confused": 4}),
                (datetime.timedelta(hours=5), {"laughing": 5}),
                (datetime.timedelta(hours=10), {"thumbsup": 1, "heart": 2}),
                (datetime.timedelta(hours=12, minutes=50), {"thumbsdown": 3}),
                (datetime.timedelta(hours=14, minutes=1), {"heart": 2}),  # ignored
            ],
            id="hourly"
        ),
        pytest.param(
            TimePeriod(days=9),
            [
                (datetime.timedelta(days=1, hours=3), {"thumbsup": 5, "thumbsdown": 2}),
                (datetime.timedelta(days=1, hours=18), {"thumbsup": 3}),
                (datetime.timedelta(days=4), {"laughing": 7, "confused": 1}),
                (datetime.timedelta(days=6, hours=12), {"heart": 4, "thumbsup": 2}),
                (datetime.timedelta(days=7, hours=23), {"thumbsdown": 6}),
                (datetime.timedelta(days=10, hours=2), {"thumbsdown": 1, "heart": 1}),  # ignored
            ],
            id="daily"
        ),
        pytest.param(
            TimePeriod(weeks=3),
            [
                (datetime.timedelta(days=2), {"thumbsup": 4, "confused": 2}),
                (datetime.timedelta(days=6), {"laughing": 3}),
                (datetime.timedelta(days=9, hours=6), {"thumbsup": 1, "heart": 5}),
                (datetime.timedelta(days=15), {"thumbsdown": 8, "laughing": 2}),
                (datetime.timedelta(days=20, hours=12), {"thumbsup": 6, "confused": 3}),
                (datetime.timedelta(days=22, hours=6), {"heart": 1, "laughing": 1}),  # ignored
            ],
            id="weekly"
        ),
    ]
)
@pytest.mark.asyncio
async def test_emoji_stats(
    period: TimePeriod,
    records: list[tuple[datetime.timedelta, dict[str, int]]],
):
    """
    Populate DB with a some preset mock emoji metadata (`records`) and check that
    they are selected (only the selected `period`) properly.
    """
    anchor = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0)
    if period.days or period.weeks:
        anchor = anchor.replace(hour=0)

    async with PopulateDatabase.populate_db_with_emoji_records(
        anchor,
        records,
    ) as _:
        stats = await emojis_per_time(period, end_time=anchor)

    emoji_sums = {  # (hourly, daily, weekly) -> counts ignore the entries outside `period`
        "thumbsup": (6, 10, 11),
        "thumbsdown": (4, 8, 8),
        "laughing": (5, 7, 5),
        "heart": (2, 4, 5),
        "confused": (4, 1, 5),
    }
    assert isinstance(stats, list)
    for e in stats:
        assert isinstance(e, MetricTimeSeries)
        hourly_sum, daily_sum, weekly_sum = emoji_sums[e.metric.removeprefix("emoji_")]
        if period.hours:
            assert float(hourly_sum) == sum(e.values)
        elif period.days:
            assert float(daily_sum) == sum(e.values)
        elif period.weeks:
            assert float(weekly_sum) == sum(e.values)
        else:
            assert False, f"Did not test any of the expected checks for {e.metric}"
