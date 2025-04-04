import datetime
from typing import Optional, Union

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
    values_dict: dict[datetime.datetime, int],
    plot_def: Definition,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    value_type: Optional[Union[int, float]] = int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Create time series arrays from a dictionary of values.

    This function generates two aligned numpy arrays:
    1. An array of timestamps from start_time to end_time
    2. A corresponding array of valuesfor each timestamp

    The timestamps are truncated to the precision specified by time_format.
    If a timestamp in values_dict matches a generated timestamp, its values is used;
    otherwise, the value defaults to zero.

    Args:
        values_dict: Dictionary mapping timestamps to their respective values
        start_time: The starting timestamp of the time series
        end_time: The ending timestamp of the time series
        time_delta: The time interval between consecutive timestamps
        time_format: String format for datetime truncation (e.g., '%Y-%m-%d %H:%M')

    Returns:
        A tuple containing:
            - numpy.ndarray: Array of timestamps
            - numpy.ndarray: Array of corresponding values
    """
    num_intervals = int((end_time - start_time) / plot_def.time_delta) + 1

    timestamps = numpy.array(
        [
            datetime.datetime.strptime(
                (start_time + i * plot_def.time_delta).strftime(
                    format=plot_def.time_format
                ),
                plot_def.time_format,
            )
            for i in range(num_intervals)
        ]
    )
    values = numpy.zeros(num_intervals, dtype=value_type)

    timestamp_to_index = {timestamp: i for i, timestamp in enumerate(timestamps)}

    for timestamp, count in values_dict.items():
        if timestamp in timestamp_to_index:
            values[timestamp_to_index[timestamp]] = count

    return timestamps, values


def _add_bar_chart(
    ax: matplotlib.figure.Axes,
    plot_def: Definition,
    timestamps: numpy.array,
    values: numpy.array,
    label: str,
) -> None:
    """Add a blue bar chart"""
    bar_width = (
        0.8 * plot_def.time_delta.total_seconds() / 86400
    )  # Convert to days for matplotlib
    ax.bar(
        timestamps,
        values,
        width=bar_width,
        alpha=0.7,
        color="skyblue",
        label=label,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel(label, color="blue")
    ax.tick_params(axis="y", labelcolor="blue")

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(plot_def.time_format))
    ax.xaxis.set_major_locator(plot_def.locator)

    matplotlib.pyplot.xticks(rotation=45)

    ax.grid(True, alpha=0.3)


def _add_line_chart(
    ax: matplotlib.figure.Axes, timestamps: numpy.array, values: numpy.array, label: str
) -> None:
    """Add a red line chart"""
    ax.plot(timestamps, values, "r-", linewidth=2, label=label)
    ax.set_ylabel(label, color="red")
    ax.tick_params(axis="y", labelcolor="red")


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
        requests_counts, plot_def, start_time, end_time
    )

    fig, ax1 = matplotlib.pyplot.subplots(figsize=(12, 6))
    _add_bar_chart(ax1, plot_def, timestamps, counts, "Requests")

    ax2 = ax1.twinx()
    _add_line_chart(ax2, timestamps, numpy.cumsum(counts), "Cumulative Requests")

    matplotlib.pyplot.title(
        f"Requests received for API {endpoint} ({start_time.strftime(plot_def.time_format)} "
        f"to {end_time.strftime(plot_def.time_format)})"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center")

    matplotlib.pyplot.tight_layout()

    return fig


def average_time_per_responses(  # pylint: disable=too-many-locals
    period_of_time: models.TimePeriod,
    endpoint: EndpointType = EndpointType.ANALYZE,
    end_time: Optional[datetime.datetime] = None,
) -> matplotlib.figure.Figure:
    """
    Generate a visualization of average response time and length over a specified time period.

    This function creates a dual-axis plot showing:
    1. A bar chart of average response time per time interval
    1. A line chart of average response length per time interval

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
    responses_average_time = AnalyzeRequestMetrics.get_responses_average_time_in_period(
        start_time, end_time, plot_def.time_format, endpoint
    )
    timestamps, average_time = create_time_series_arrays(
        responses_average_time,
        plot_def,
        start_time,
        end_time,
        float,
    )

    fig, ax1 = matplotlib.pyplot.subplots(figsize=(12, 6))
    _add_bar_chart(ax1, plot_def, timestamps, average_time, "average response time (seconds)")

    responses_average_length = (
        AnalyzeRequestMetrics.get_responses_average_length_in_period(
            start_time, end_time, plot_def.time_format, endpoint
        )
    )
    timestamps, average_length = create_time_series_arrays(
        responses_average_length,
        plot_def,
        start_time,
        end_time,
        float,
    )

    ax2 = ax1.twinx()
    _add_line_chart(ax2, timestamps, average_length, "average response length (chars)")

    matplotlib.pyplot.title(
        f"average response time for API {endpoint} ({start_time.strftime(plot_def.time_format)} "
        f"to {end_time.strftime(plot_def.time_format)})"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center")

    matplotlib.pyplot.tight_layout()

    return fig
