import datetime
import tempfile

import pytest

from test_helpers import PopulateDatabase

from logdetective.server import plot, models
from logdetective.server.database.models import AnalyzeRequestMetrics, EndpointType
from logdetective.server.plot import (
    create_time_series_arrays,
    requests_per_time,
    average_time_per_responses,
    emojis_per_time,
)


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


@pytest.mark.parametrize(
    "endpoint",
    [EndpointType.ANALYZE, EndpointType.ANALYZE_STAGED],
)
def test_create_time_series_arrays(endpoint):
    duration = datetime.timedelta(hours=15)
    with PopulateDatabase.populate_db(
        duration=duration,
        endpoint=endpoint,
    ) as _:
        period = models.TimePeriod(hours=22)
        plot_details = plot.Definition(period)
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = period.get_period_start_time(end_time)
        counts_dict = AnalyzeRequestMetrics.get_requests_in_period(
            start_time, end_time, plot_details.time_format, endpoint
        )
        timestamps, counts = create_time_series_arrays(
            counts_dict,
            plot_details,
            start_time,
            end_time,
        )
        assert len(timestamps) == len(counts) == 22 + 1
        assert (
            sum(counts) < 22 * 4
        )  # since we have added requests just for the last 15 hours


def _save_fig(fig):
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        temp_filename = tmp_file.name
    fig.savefig(temp_filename, bbox_inches="tight")  # for inspecting it later


def _test_plot(
    duration: datetime.timedelta,
    period: models.TimePeriod,
    plot: callable,
    endpoint: EndpointType = None,
) -> None:
    if plot is emojis_per_time:
        with PopulateDatabase.populate_db_with_emojis(
            duration=duration,
        ) as _:
            fig = plot(period)
    else:
        with PopulateDatabase.populate_db(
            duration=duration,
            endpoint=endpoint,
        ) as _:
            fig = plot(period, endpoint)
    assert fig
    _save_fig(fig)


@pytest.mark.parametrize(
    "endpoint,plot",
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
def test_hourly_plots(endpoint, plot):
    duration = datetime.timedelta(hours=14)
    period = models.TimePeriod(hours=22)
    _test_plot(duration, period, plot, endpoint)


@pytest.mark.parametrize(
    "endpoint,plot",
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
def test_daily_plots(endpoint, plot):
    duration = datetime.timedelta(days=9)
    period = models.TimePeriod(days=15)
    _test_plot(duration, period, plot, endpoint)


@pytest.mark.parametrize(
    "endpoint,plot",
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
def test_weekly_plots(endpoint, plot):
    duration = datetime.timedelta(weeks=3)
    period = models.TimePeriod(weeks=5)
    _test_plot(duration, period, plot, endpoint)
