import inspect
from collections import defaultdict
import datetime
from typing import Optional, Union, Dict
from functools import wraps

import aiohttp
import numpy
from starlette.responses import StreamingResponse

from logdetective.remote_log import RemoteLog
from logdetective.server.config import LOG
from logdetective.server.compressors import LLMResponseCompressor, RemoteLogCompressor
from logdetective.server.models import (
    TimePeriod,
    MetricTimeSeries,
    StagedResponse,
    Response,
    Explanation,
)
from logdetective.server.database.models import EndpointType, AnalyzeRequestMetrics, Reactions
from logdetective.server.exceptions import LogDetectiveMetricsError


async def add_new_metrics(
    api_name: EndpointType,
    url: Optional[str] = None,
    http_session: Optional[aiohttp.ClientSession] = None,
    received_at: Optional[datetime.datetime] = None,
    compressed_log_content: Optional[bytes] = None,
) -> int:
    """Add a new database entry for a received request.

    This will store the time when this function is called,
    the endpoint from where the request was received,
    and the log (in a zip format) for which analysis is requested.
    """
    if not compressed_log_content:
        if not (url and http_session):
            raise LogDetectiveMetricsError(
                f"""Remote log can not be retrieved without URL and http session.
                URL: {url}, http session:{http_session}""")
        remote_log = RemoteLog(url, http_session)
        compressed_log_content = await RemoteLogCompressor(remote_log).zip_content()

    return await AnalyzeRequestMetrics.create(
        endpoint=EndpointType(api_name),
        compressed_log=compressed_log_content,
        request_received_at=received_at
        if received_at
        else datetime.datetime.now(datetime.timezone.utc),
    )


async def update_metrics(
    metrics_id: int,
    response: Union[Response, StagedResponse, StreamingResponse],
    sent_at: Optional[datetime.datetime] = None,
) -> None:
    """Update a database metric entry for a received request,
    filling data for the given response.

    This will add to the database entry the time when the response was sent,
    the length of the created response and the certainty for it.
    """
    try:
        compressed_response = LLMResponseCompressor(response).zip_response()
    except AttributeError as e:
        compressed_response = None
        LOG.warning(
            "Given response can not be serialized "
            "and saved in db (probably a StreamingResponse): %s.",
            e,
        )

    response_sent_at = (
        sent_at if sent_at else datetime.datetime.now(datetime.timezone.utc)
    )
    response_length = None
    if hasattr(response, "explanation") and isinstance(
        response.explanation, Explanation
    ):
        response_length = len(response.explanation.text)
    response_certainty = (
        response.response_certainty if hasattr(response, "response_certainty") else None
    )
    await AnalyzeRequestMetrics.update(
        id_=metrics_id,
        response_sent_at=response_sent_at,
        response_length=response_length,
        response_certainty=response_certainty,
        compressed_response=compressed_response,
    )


def track_request(name=None):
    """
    Decorator to track requests/responses metrics

    On entering the decorated function, it registers the time for the request
    and saves the passed log content.
    On exiting the decorated function, it registers the time for the response
    and saves the generated response.

    Use it to decorate server endpoints that generate a llm response
    as in the following example:

    >>> @app.post("/analyze", response_model=Response)
    >>> @track_request()
    >>> async def analyze_log(build_log)
    >>>     pass

    Warning: the decorators' order is important!
    The function returned by the *track_request* decorator is the
    server API function we want to be called by FastAPI.
    """

    def decorator(f):
        @wraps(f)
        async def async_decorated_function(*args, **kwargs):
            log_url = kwargs["build_log"].url
            metrics_id = await add_new_metrics(
                api_name=EndpointType(name if name else f.__name__),
                url=log_url, http_session=kwargs["http_session"]
            )
            response = await f(*args, **kwargs)
            await update_metrics(metrics_id, response)
            return response

        if inspect.iscoroutinefunction(f):
            return async_decorated_function
        raise NotImplementedError("An async coroutine is needed")

    return decorator


# TODO: Refactor aggregation to use database operations, instead of timestamp formatting  # pylint: disable=fixme
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

    return timestamps.tolist(), numpy.nan_to_num(values).tolist()


async def requests_per_time(
    period_of_time: TimePeriod,
    endpoint: EndpointType = EndpointType.ANALYZE,
    end_time: Optional[datetime.datetime] = None,
) -> MetricTimeSeries:
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

    return MetricTimeSeries(metric="requests", timestamps=timestamps, values=counts)


async def average_time_per_responses(
    period_of_time: TimePeriod,
    endpoint: EndpointType = EndpointType.ANALYZE,
    end_time: Optional[datetime.datetime] = None,
) -> MetricTimeSeries:
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

    return MetricTimeSeries(metric="avg_response_time", timestamps=timestamps, values=average_time)


async def _collect_emoji_data(
    start_time: datetime.datetime, time_def: TimeDefinition
) -> Dict[str, Dict[str, list]]:
    """Collect and organize emoji feedback data

    For each reaction type, a dictionary is created with time stamps
    as keys, and aggregate counts as values.
    """
    reactions = await Reactions.get_since(start_time)
    reaction_values: defaultdict[str, Dict] = defaultdict(lambda: defaultdict(int))

    for comment_timestamp, reaction in reactions:
        formatted_timestamp = comment_timestamp.strptime(
            comment_timestamp.strftime(time_def.time_format), time_def.time_format
        )

        reaction_values[reaction.reaction_type][formatted_timestamp] += reaction.count

    reaction_time_series = {
        reaction_type: {
            "timestamps": reaction_data.keys(),
            "values": reaction_data.values(),
        }
        for reaction_type, reaction_data in reaction_values.items()
    }

    return reaction_time_series


async def emojis_per_time(
    period_of_time: TimePeriod,
    end_time: Optional[datetime.datetime] = None,
) -> list[MetricTimeSeries]:
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
        A list of `MetricTimeSeries` objects
    """
    time_def = TimeDefinition(period_of_time)
    end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
    start_time = period_of_time.get_period_start_time(end_time)
    reactions_values_dict = await _collect_emoji_data(start_time, time_def)

    reaction_values: list[MetricTimeSeries] = []
    for reaction, time_series in reactions_values_dict.items():
        reaction_values.append(
            MetricTimeSeries(
                metric=f"emoji_{reaction}",
                timestamps=time_series["timestamps"],
                values=time_series["values"]))
    return reaction_values
