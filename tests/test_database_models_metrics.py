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
from logdetective.metric import update_metrics
from logdetective.models import APIResponse


def test_endpoint_enum_values_match_lowercase_database_labels() -> None:
    """Persist endpoint values instead of SQLAlchemy's uppercase member names."""
    assert AnalyzeRequestMetrics.__table__.c.endpoint.type.enums == [
        endpoint.value for endpoint in EndpointType
    ]


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
        assert metrics.response_sent_at is not None
        assert metrics.response_length == 0
        assert metrics.api_token_name == "packit"


@pytest.mark.asyncio
async def test_worker_update_preserves_admitted_webhook_response_time():
    """Inference output cannot replace the webhook acknowledgement timestamp."""
    received_at = datetime.datetime(2077, 1, 1, tzinfo=datetime.timezone.utc)
    admitted_at = received_at + datetime.timedelta(seconds=2)
    worker_at = received_at + datetime.timedelta(minutes=5)
    response = APIResponse(explanation="Analysis complete")
    async with DatabaseFactory().make_new_db():
        metrics_id = await AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE_GITLAB_JOB,
            request_received_at=received_at,
        )
        await AnalyzeRequestMetrics.update(
            id_=metrics_id,
            response_sent_at=admitted_at,
        )

        await update_metrics(metrics_id, response, sent_at=worker_at)
        metrics = await AnalyzeRequestMetrics.get_metric_by_id(metrics_id)
        statistics = await AnalyzeRequestMetrics.get_requests_stats_for_period(
            start_time=received_at,
            end_time=received_at + datetime.timedelta(hours=1),
            endpoint=EndpointType.ANALYZE_GITLAB_JOB,
            time_period=TimePeriod.HOUR,
        )

    assert metrics.response_sent_at == admitted_at
    assert metrics.response_length == len(response.model_dump_json())
    assert statistics[2] == [2.0]


@pytest.mark.asyncio
async def test_get_metric_by_id_and_missing_metric_errors():
    """Records can be retrieved, while updates and lookups reject unknown IDs."""
    async with DatabaseFactory().make_new_db():
        metrics_id = await AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE,
        )

        metrics = await AnalyzeRequestMetrics.get_metric_by_id(metrics_id)

        assert metrics.id == metrics_id
        with pytest.raises(ValueError, match="table is empty"):
            await AnalyzeRequestMetrics.get_metric_by_id(metrics_id + 1)
        with pytest.raises(ValueError, match="table is empty"):
            await AnalyzeRequestMetrics.update(
                id_=metrics_id + 1,
                response_sent_at=datetime.datetime.now(datetime.timezone.utc),
            )


@pytest.mark.asyncio
async def test_get_requests_stats_for_empty_period():
    """An empty query preserves the five-column metrics response shape."""
    now = datetime.datetime.now(datetime.timezone.utc)
    async with DatabaseFactory().make_new_db():
        metrics = await AnalyzeRequestMetrics.get_requests_stats_for_period(
            start_time=now - datetime.timedelta(days=1),
            end_time=now,
        )

    assert metrics == [[], [], [], [], []]


@pytest.mark.asyncio
async def test_get_requests_stats_aggregates_every_metric_column():
    """Counts and all averages are aggregated into aligned time buckets."""
    period_start = datetime.datetime(2077, 1, 1, tzinfo=datetime.timezone.utc)
    first_request = period_start + datetime.timedelta(minutes=10)
    second_request = period_start + datetime.timedelta(minutes=20)
    first_response = first_request + datetime.timedelta(seconds=2)
    first_completion = first_request + datetime.timedelta(seconds=1)
    second_response = second_request + datetime.timedelta(seconds=4)
    second_completion = second_request + datetime.timedelta(seconds=3)
    async with DatabaseFactory().make_new_db() as session_factory:
        async with session_factory() as session:
            session.add_all(
                [
                    AnalyzeRequestMetrics(
                        endpoint=EndpointType.ANALYZE,
                        request_received_at=first_request,
                        response_sent_at=first_response,
                        analysis_completed_at=first_completion,
                        response_length=100,
                    ),
                    AnalyzeRequestMetrics(
                        endpoint=EndpointType.ANALYZE,
                        request_received_at=second_request,
                        response_sent_at=second_response,
                        analysis_completed_at=second_completion,
                        response_length=300,
                    ),
                ]
            )
            await session.commit()

        metrics = await AnalyzeRequestMetrics.get_requests_stats_for_period(
            start_time=period_start,
            end_time=period_start + datetime.timedelta(hours=1),
            time_period=TimePeriod.HOUR,
        )

    assert metrics == [[period_start], [2], [3.0], [200.0], [2.0]]


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
