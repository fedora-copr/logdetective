import datetime
from typing import Optional

from logdetective.models import (
    MetricsData,
    APIResponse,
    Explanation,
)
from logdetective.database.models import EndpointType, AnalyzeRequestMetrics, TimePeriod


async def add_new_metrics(
    api_name: EndpointType,
    received_at: Optional[datetime.datetime] = None,
    api_token_name: str | None = None,
) -> int:
    """Add a new database entry for a received request.

    This will store the time when this function is called and
    the endpoint from where the request was received.
    """

    # gitlab and koji always fall through here
    return await AnalyzeRequestMetrics.create(
        endpoint=EndpointType(api_name),
        request_received_at=received_at,
        api_token_name=api_token_name,
    )


async def update_metrics(
    metrics_id: int,
    response: APIResponse,
    sent_at: Optional[datetime.datetime] = None,
) -> None:
    """Update a database metric entry for a received request,
    filling data for the given response.

    This fills the response timestamp only if admission did not record it,
    and updates the length of the created response.
    """

    response_sent_at = (
        sent_at if sent_at else datetime.datetime.now(datetime.timezone.utc)
    )
    response_length = None
    if hasattr(response, "explanation") and isinstance(
        response.explanation, Explanation
    ):
        response_length = len(response.model_dump_json())
    await AnalyzeRequestMetrics.update(
        id_=metrics_id,
        response_sent_at=response_sent_at,
        response_length=response_length,
    )


async def requests_statistics(
    start_time: datetime.datetime,
    end_time: Optional[datetime.datetime] = None,
    endpoint: EndpointType = EndpointType.ANALYZE,
    time_period: TimePeriod = TimePeriod.DAY,
    api_token_name: Optional[str] = None,
) -> MetricsData:
    """
    Get request counts and average response times over a specified time period.

    The time intervals are determined by the provided TimePeriod object, which defines
    the granularity.

    Args:
        start_time: The start_time time for the analysis period.
        end_time: The end time for the analysis period. If None, defaults to the current
                UTC time
        endpoint: One of the API endpoints
        api_token_name: If set, include only requests made with this named token

    Returns:
        A `MetricsData` columnar representation of the gathered data
    """
    end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
    statistics = await AnalyzeRequestMetrics.get_requests_stats_for_period(
        start_time=start_time,
        end_time=end_time,
        time_period=time_period,
        endpoint=endpoint,
        api_token_name=api_token_name,
    )

    return MetricsData(
        endpoint=endpoint.value,
        period_start=statistics[0],
        total_count=statistics[1],
        average_response_time=statistics[2],
        average_response_len=statistics[3],
        average_completion_time=statistics[4],
    )
