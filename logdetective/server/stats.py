import datetime
from typing import Optional, Union, Dict

import numpy

from logdetective.server.models import TimePeriod
from logdetective.server.database.models import (
    AnalyzeRequestMetrics,
    EndpointType,
    Reactions,
)


class TimeDefinition:
    """Define time format details, given a time period."""

    def __init__(self, time_period: TimePeriod):
        self.time_period = time_period
        self.days_diff = time_period.get_time_period().days
        if self.time_period.hours:
            self._time_format = "%Y-%m-%d %H"
            self._time_delta = datetime.timedelta(hours=1)
        elif self.time_period.days:
            self._time_format = "%Y-%m-%d"
            self._time_delta = datetime.timedelta(days=1)
        elif self.time_period.weeks:
            self._time_format = "%Y-%m-%d"
            self._time_delta = datetime.timedelta(weeks=1)

    @property
    def time_format(self):
        # pylint: disable=missing-function-docstring
        return self._time_format

    @property
    def time_delta(self):
        # pylint: disable=missing-function-docstring
        return self._time_delta


def create_time_series_arrays(
    values_dict: dict[datetime.datetime, int],
    time_def: TimeDefinition,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    value_type: Optional[Union[type[int], type[float]]] = int,
) -> tuple[list, list]:
    """Create time series arrays from a dictionary of values.

    This function generates two aligned numpy arrays:
    1. An array of timestamps from start_time to end_time
    2. A corresponding array of values for each timestamp

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
            - list: Array of timestamps
            - list: Array of corresponding values
    """
    num_intervals = int((end_time - start_time) / time_def.time_delta) + 1

    timestamps = numpy.array(
        [
            datetime.datetime.strptime(
                (start_time + i * time_def.time_delta).strftime(
                    format=time_def.time_format
                ),
                time_def.time_format,
            )
            for i in range(num_intervals)
        ]
    )
    values = numpy.zeros(num_intervals, dtype=value_type)

    timestamp_to_index = {timestamp: i for i, timestamp in enumerate(timestamps)}

    for timestamp, count in values_dict.items():
        if timestamp in timestamp_to_index:
            values[timestamp_to_index[timestamp]] = count

    return timestamps.tolist(), values.tolist()


async def requests_per_time(
    period_of_time: TimePeriod,
    endpoint: EndpointType = EndpointType.ANALYZE,
    end_time: Optional[datetime.datetime] = None,
) -> dict[str, list]:
    """
    Get request counts over a specified time period.

    The time intervals are determined by the provided TimePeriod object, which defines
    the granularity.

    Args:
        period_of_time: A TimePeriod object that defines the time period and interval
                        for the analysis (e.g., hourly, daily, weekly)
        endpoint: One of the API endpoints
        end_time: The end time for the analysis period. If None, defaults to the current
                  UTC time

    Returns:
        A dictionary with timestamps and associated number of requests
    """
    end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
    start_time = period_of_time.get_period_start_time(end_time)
    time_def = TimeDefinition(period_of_time)
    requests_counts = await AnalyzeRequestMetrics.get_requests_in_period(
        start_time, end_time, time_def.time_format, endpoint
    )
    timestamps, counts = create_time_series_arrays(
        requests_counts, time_def, start_time, end_time
    )

    return {"timestamp": timestamps, "count": counts}


async def average_time_per_responses(
    period_of_time: TimePeriod,
    endpoint: EndpointType = EndpointType.ANALYZE,
    end_time: Optional[datetime.datetime] = None,
) -> Dict[str, list]:
    """
    Get average response time and length over a specified time period.

    The time intervals are determined by the provided TimePeriod object, which defines
    the granularity.

    Args:
        period_of_time: A TimePeriod object that defines the time period and interval
                        for the analysis (e.g., hourly, daily, weekly)
        endpoint: One of the API endpoints
        end_time: The end time for the analysis period. If None, defaults to the current
                  UTC time

    Returns:
        A dictionary of timestamps and average response times
    """
    end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
    start_time = period_of_time.get_period_start_time(end_time)
    time_def = TimeDefinition(period_of_time)
    responses_average_time = (
        await AnalyzeRequestMetrics.get_responses_average_time_in_period(
            start_time, end_time, time_def.time_format, endpoint
        )
    )
    timestamps, average_time = create_time_series_arrays(
        responses_average_time,
        time_def,
        start_time,
        end_time,
        float,
    )

    return {"timestamp": timestamps, "avg_time": average_time}


async def _collect_emoji_data(
    start_time: datetime.datetime, time_def: TimeDefinition
) -> Dict[str, Dict[datetime.datetime, int]]:
    """Collect and organize emoji feedback data

    Counts all emojis given to logdetective comments created since start_time.
    Collect counts in time accordingly to the time definition.
    """
    reactions = await Reactions.get_since(start_time)
    reactions_values_dict: Dict[str, Dict] = {}
    for comment_created_at, reaction in reactions:
        comment_created_at_formatted = comment_created_at.strptime(
            comment_created_at.strftime(time_def.time_format), time_def.time_format
        )
        if reaction.reaction_type in reactions_values_dict:
            reaction_values_dict = reactions_values_dict[reaction.reaction_type]
            if comment_created_at_formatted in reaction_values_dict:
                reaction_values_dict[comment_created_at_formatted] += reaction.count
            else:
                reaction_values_dict[comment_created_at_formatted] = reaction.count
        else:
            reaction_values_dict = {comment_created_at_formatted: reaction.count}
            reactions_values_dict.update({reaction.reaction_type: reaction_values_dict})

    return reactions_values_dict


async def emojis_per_time(
    period_of_time: TimePeriod,
    end_time: Optional[datetime.datetime] = None,
) -> Dict[str, Dict[datetime.datetime, int]]:
    """
    Retrieve data of emoji feedback over time.

    The time intervals are determined by the provided TimePeriod object, which defines
    the granularity.

    Args:
        period_of_time: A TimePeriod object that defines the time period and interval
                        for the analysis (e.g., hourly, daily, weekly)
        end_time: The end time for the analysis period. If None, defaults to the current
                  UTC time

    Returns:
        A dictionary containing retrieved statistics
    """
    time_def = TimeDefinition(period_of_time)
    end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
    start_time = period_of_time.get_period_start_time(end_time)
    reactions_values_dict = await _collect_emoji_data(start_time, time_def)

    return reactions_values_dict
