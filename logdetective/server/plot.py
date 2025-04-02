import datetime
from typing import Optional

import numpy
import matplotlib
import matplotlib.figure
import matplotlib.pyplot

from logdetective.server import models
from logdetective.server.database.models import AnalyzeRequestMetrics, EndpointType


class Definition:
    """Define plot details, given a time period."""

    def __init__(self, time_period: models.TimePeriod):
        self.time_period = time_period
        self.days_diff = time_period.get_time_period().days
        if self.time_period.hours:
            self._freq = "H"
            self._time_format = "%Y-%m-%d %H"
            self._locator = matplotlib.dates.HourLocator(interval=2)
            self._time_unit = "hour"
            self._time_delta = datetime.timedelta(hours=1)
        elif self.time_period.days:
            self._freq = "D"
            self._time_format = "%Y-%m-%d"
            self._locator = matplotlib.dates.DayLocator(interval=1)
            self._time_unit = "day"
            self._time_delta = datetime.timedelta(days=1)
        elif self.time_period.weeks:
            self._freq = "W"
            self._time_format = "%Y-%m-%d"
            self._locator = matplotlib.dates.WeekdayLocator(interval=1)
            self._time_unit = "week"
            self._time_delta = datetime.timedelta(weeks=1)

    @property
    def freq(self):
        # pylint: disable=missing-function-docstring
        return self._freq

    @property
    def time_format(self):
        # pylint: disable=missing-function-docstring
        return self._time_format

    @property
    def locator(self):
        # pylint: disable=missing-function-docstring
        return self._locator

    @property
    def time_unit(self):
        # pylint: disable=missing-function-docstring
        return self._time_unit

    @property
    def time_delta(self):
        # pylint: disable=missing-function-docstring
        return self._time_delta


def create_time_series_arrays(
    counts_dict: dict[datetime.datetime, int],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    time_delta: datetime.timedelta,
    time_format: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Create time series arrays from a dictionary of counts.

    This function generates two aligned numpy arrays:
    1. An array of timestamps from start_time to end_time
    2. A corresponding array of counts for each timestamp

    The timestamps are truncated to the precision specified by time_format.
    If a timestamp in counts_dict matches a generated timestamp, its count is used;
    otherwise, the count defaults to zero.

    Args:
        counts_dict: Dictionary mapping timestamps to their respective counts
        start_time: The starting timestamp of the time series
        end_time: The ending timestamp of the time series
        time_delta: The time interval between consecutive timestamps
        time_format: String format for datetime truncation (e.g., '%Y-%m-%d %H:%M')

    Returns:
        A tuple containing:
            - numpy.ndarray: Array of timestamps
            - numpy.ndarray: Array of corresponding counts
    """
    num_intervals = int((end_time - start_time) / time_delta) + 1

    timestamps = numpy.array(
        [
            datetime.datetime.strptime(
                (start_time + i * time_delta).strftime(format=time_format), time_format
            )
            for i in range(num_intervals)
        ]
    )
    counts = numpy.zeros(num_intervals, dtype=int)

    timestamp_to_index = {timestamp: i for i, timestamp in enumerate(timestamps)}

    for timestamp, count in counts_dict.items():
        if timestamp in timestamp_to_index:
            counts[timestamp_to_index[timestamp]] = count

    return timestamps, counts


def _add_bar_chart_for_requests_count(
    ax1: matplotlib.figure.Axes,
    plot_def: Definition,
    timestamps: numpy.array,
    counts: numpy.array,
) -> None:
    """Add a bar chart for requests count (axes 1)"""
    bar_width = (
        0.8 * plot_def.time_delta.total_seconds() / 86400
    )  # Convert to days for matplotlib
    ax1.bar(
        timestamps,
        counts,
        width=bar_width,
        alpha=0.7,
        color="skyblue",
        label="Requests",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Requests", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(plot_def.time_format))
    ax1.xaxis.set_major_locator(plot_def.locator)

    matplotlib.pyplot.xticks(rotation=45)

    ax1.grid(True, alpha=0.3)


def _add_cumulative_line_for_requests_count(
    ax2: matplotlib.figure.Axes, timestamps: numpy.array, counts: numpy.array
) -> None:
    """Add cumulative line on secondary y-axis"""
    cumulative = numpy.cumsum(counts)
    ax2.plot(timestamps, cumulative, "r-", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Requests", color="red")
    ax2.tick_params(axis="y", labelcolor="red")


def requests_per_time(
    period_of_time: models.TimePeriod,
    endpoint: EndpointType = EndpointType.ANALYZE,
    end_time: Optional[datetime.datetime] = None,
) -> matplotlib.figure.Figure:
    """
    Generate a visualization of request counts over a specified time period.

    This function creates a dual-axis plot showing:
    1. A bar chart of request counts per time interval
    2. A line chart showing the cumulative request count

    The time intervals are determined by the provided TimePeriod object, which defines
    the granularity and formatting of the time axis.

    Args:
        period_of_time: A TimePeriod object that defines the time period and interval
                        for the analysis (e.g., hourly, daily, weekly)
        endpoint: One of the API endpoints
        end_time: The end time for the analysis period. If None, defaults to the current
                  UTC time

    Returns:
        A matplotlib Figure object containing the generated visualization
    """
    end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
    start_time = period_of_time.get_period_start_time(end_time)
    plot_def = Definition(period_of_time)
    requests_counts = AnalyzeRequestMetrics.get_requests_in_period(
        start_time, end_time, plot_def.time_format, endpoint
    )
    timestamps, counts = create_time_series_arrays(
        requests_counts, start_time, end_time, plot_def.time_delta, plot_def.time_format
    )

    fig, ax1 = matplotlib.pyplot.subplots(figsize=(12, 6))
    _add_bar_chart_for_requests_count(ax1, plot_def, timestamps, counts)

    ax2 = ax1.twinx()
    _add_cumulative_line_for_requests_count(ax2, timestamps, counts)

    matplotlib.pyplot.title(
        f"Requests received for API {endpoint} ({start_time.strftime(plot_def.time_format)} "
        f"to {end_time.strftime(plot_def.time_format)})"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center")

    matplotlib.pyplot.tight_layout()

    return fig
