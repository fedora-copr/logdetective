import datetime

import pytest
from sqlalchemy import select

from tests.test_helpers import (
    DatabaseFactory,
    PopulateDatabase,
)

from logdetective.database.models import (
    AnalyzeRequestMetrics,
    EndpointType,
)

from logdetective.database.models.metrics import TimePeriod


@pytest.mark.asyncio
async def test_create_and_update_AnalyzeRequestMetrics():
    async with DatabaseFactory().make_new_db() as session_factory:
        metrics_id = await AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE,
            api_token_name="packit",
        )
        assert metrics_id == 1
        await AnalyzeRequestMetrics.update(
            id_=metrics_id,
            response_sent_at=datetime.datetime.now(datetime.timezone.utc),
            response_length=0,
        )

        query = select(AnalyzeRequestMetrics).filter(
            AnalyzeRequestMetrics.id == metrics_id
        )
        async with session_factory() as session:
            query_result = await session.execute(query)
            metrics = query_result.scalars().first()

        assert metrics is not None
        assert metrics.response_length == 0
        assert metrics.api_token_name == "packit"


@pytest.mark.asyncio
async def test_filter_request_metrics_by_api_token_name():
    now = datetime.datetime.now(datetime.timezone.utc)
    async with DatabaseFactory().make_new_db():
        await AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE,
            request_received_at=now,
            api_token_name="packit",
        )
        await AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE,
            request_received_at=now,
            api_token_name="monitoring",
        )

        metrics = await AnalyzeRequestMetrics.get_requests_stats_for_period(
            now - datetime.timedelta(minutes=1),
            now + datetime.timedelta(minutes=1),
            endpoint=EndpointType.ANALYZE,
            api_token_name="packit",
        )

    assert sum(metrics[1]) == 1


@pytest.mark.parametrize(
    "endpoint",
    [EndpointType.ANALYZE],
)
@pytest.mark.asyncio
async def test_AnalyzeRequestMetrics_get_requests_stats_for_period(endpoint):
    duration = datetime.timedelta(hours=13)
    end_time = datetime.datetime(year=2077, month=1, day=1, tzinfo=datetime.UTC)
    async with PopulateDatabase.populate_db(
        duration=duration,
        endpoint=endpoint,
        end_time=end_time,
    ) as _:
        start_time = end_time - datetime.timedelta(hours=10)
        stats = await AnalyzeRequestMetrics.get_requests_stats_for_period(
            start_time=start_time,
            end_time=end_time,
            time_period=TimePeriod.HOUR,
            endpoint=endpoint,
        )

        # Basic checks on the returned data structure
        assert len(stats) == 5
        for e in stats:
            assert isinstance(e, list)

        response_times = stats[2]
        assert len(response_times) == 10
        # responses times always increase in the same way inside the hour
        assert response_times[2] == pytest.approx(response_times[3], abs=1e-3)
        assert response_times[4] == pytest.approx(response_times[5], abs=1e-3)
