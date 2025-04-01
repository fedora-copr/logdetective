import datetime
import tempfile

from flexmock import flexmock

from test_helpers import populate_db_with_analyze_request_every_15_minutes_for_15_hours
from test_helpers import populate_db_with_analyze_request_every_15_minutes_for_9_days
from test_helpers import populate_db_with_analyze_request_every_15_minutes_for_3_weeks

from logdetective.server import plot, models
from logdetective.server.database.models import AnalyzeRequestMetrics
from logdetective.server.plot import create_time_series_arrays, requests_per_time


def test_week_Definition():
    details = plot.Definition(models.TimePeriod(weeks=3))
    assert details.time_unit == "week"
    assert details.freq == "W"
    assert details.days_diff == 21


def test_day_Definition():
    details = plot.Definition(models.TimePeriod(days=3))
    assert details.time_unit == "day"
    assert details.freq == "D"
    assert details.days_diff == 3


def test_create_time_series_arrays(
    populate_db_with_analyze_request_every_15_minutes_for_15_hours,
):
    with populate_db_with_analyze_request_every_15_minutes_for_15_hours as _:
        flexmock(AnalyzeRequestMetrics).should_receive(
            "_get_requests_by_time_for_postgres"
        ).replace_with(AnalyzeRequestMetrics._get_requests_by_time_for_sqllite)
        period = models.TimePeriod(hours=22)
        plot_details = plot.Definition(period)
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = period.get_period_start_time(end_time)
        counts_dict = AnalyzeRequestMetrics.get_requests_in_period(
            start_time, end_time, plot_details.time_format
        )
        timestamps, counts = create_time_series_arrays(
            counts_dict,
            start_time,
            end_time,
            plot_details.time_delta,
            plot_details.time_format,
        )
        assert len(timestamps) == len(counts) == 22 + 1
        assert (
            sum(counts) < 22 * 4
        )  # since the fixture add requests just for the last 15 hours


def _save_fig(fig):
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        temp_filename = tmp_file.name
    fig.savefig(temp_filename, bbox_inches="tight")  # for inspecting it later


def test_requests_per_time_hourly(
    populate_db_with_analyze_request_every_15_minutes_for_15_hours,
):
    with populate_db_with_analyze_request_every_15_minutes_for_15_hours as _:
        flexmock(AnalyzeRequestMetrics).should_receive(
            "_get_requests_by_time_for_postgres"
        ).replace_with(AnalyzeRequestMetrics._get_requests_by_time_for_sqllite)
        period = models.TimePeriod(hours=22)
        fig = requests_per_time(period)
        assert fig
        _save_fig(fig)


def test_requests_per_time_daily(
    populate_db_with_analyze_request_every_15_minutes_for_9_days,
):
    with populate_db_with_analyze_request_every_15_minutes_for_9_days as _:
        flexmock(AnalyzeRequestMetrics).should_receive(
            "_get_requests_by_time_for_postgres"
        ).replace_with(AnalyzeRequestMetrics._get_requests_by_time_for_sqllite)
        period = models.TimePeriod(days=15)
        fig = requests_per_time(period)
        assert fig
        _save_fig(fig)


def test_requests_per_time_weekly(
    populate_db_with_analyze_request_every_15_minutes_for_3_weeks,
):
    with populate_db_with_analyze_request_every_15_minutes_for_3_weeks as _:
        flexmock(AnalyzeRequestMetrics).should_receive(
            "_get_requests_by_time_for_postgres"
        ).replace_with(AnalyzeRequestMetrics._get_requests_by_time_for_sqllite)
        period = models.TimePeriod(weeks=5)
        fig = requests_per_time(period)
        assert fig
        _save_fig(fig)
