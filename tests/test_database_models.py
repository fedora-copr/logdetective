import datetime

from flexmock import flexmock

from test_helpers import (
    DatabaseFactory,
    populate_db_with_analyze_request_every_15_minutes_for_15_hours,
)

from logdetective.server.database.models import AnalyzeRequestMetrics, EndpointType


def test_create_and_update_AnalyzeRequestMetrics():
    with DatabaseFactory().make_new_db() as session_factory:
        metrics_id = AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE,
            log_url="https://example.com/logs/123",
        )
        assert metrics_id == 1
        AnalyzeRequestMetrics.update(
            id_=metrics_id,
            response_sent_at=datetime.datetime.now(datetime.timezone.utc),
            response_length=0,
            response_certainty=37.7,
        )

        metrics = (
            session_factory()
            .query(AnalyzeRequestMetrics)
            .filter_by(id=metrics_id)
            .first()
        )

        assert metrics is not None
        assert metrics.log_url == "https://example.com/logs/123"
        assert metrics.response_length == 0
        assert metrics.response_certainty == 37.7


def test_AnalyzeRequestMetrics_ger_request_in_period(
    populate_db_with_analyze_request_every_15_minutes_for_15_hours,
):
    with populate_db_with_analyze_request_every_15_minutes_for_15_hours as _:
        flexmock(AnalyzeRequestMetrics).should_receive(
            "_get_requests_by_time_for_postgres"
        ).replace_with(AnalyzeRequestMetrics._get_requests_by_time_for_sqllite)
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(hours=10)
        time_format = "%Y-%m-%d %H"
        counts_dict = AnalyzeRequestMetrics.get_requests_in_period(
            start_time, end_time, time_format
        )
        assert len(counts_dict) == 10 or len(counts_dict) == 11
