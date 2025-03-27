import datetime
import inspect
from typing import Union
from functools import wraps

from starlette.responses import StreamingResponse
from logdetective.server.database.models import EndpointType, AnalyzeRequestMetrics
from logdetective.server import models


def add_new_metrics(
    api_name: str, build_log: models.BuildLog, received_at: datetime.datetime = None
) -> int:
    """Add a new database entry for a received request.

    This will store the time when this function is called,
    the endpoint from where the request was received,
    and the log for which analysis is requested.
    """
    return AnalyzeRequestMetrics.create(
        endpoint=EndpointType(api_name),
        log_url=build_log.url,
        request_received_at=received_at
        if received_at
        else datetime.datetime.now(datetime.timezone.utc),
    )


def update_metrics(
    metrics_id: int,
    response: Union[models.Response, models.StagedResponse, StreamingResponse],
    sent_at: datetime.datetime = None,
) -> None:
    """Update a database metric entry for a received request,
    filling data for the given response.

    This will add to the database entry the time when the response was sent,
    the length of the created response and the certainty for it.
    """
    response_sent_at = (
        sent_at if sent_at else datetime.datetime.now(datetime.timezone.utc)
    )
    response_length = None
    if hasattr(response, "explanation") and "choices" in response.explanation:
        response_length = sum(
            len(choice["text"])
            for choice in response.explanation["choices"]
            if "text" in choice
        )
    response_certainty = (
        response.response_certainty if hasattr(response, "response_certainty") else None
    )
    AnalyzeRequestMetrics.update(
        metrics_id, response_sent_at, response_length, response_certainty
    )


def track_request():
    """
    Decorator to track requests metrics
    """

    def decorator(f):
        @wraps(f)
        async def async_decorated_function(*args, **kwargs):
            metrics_id = add_new_metrics(f.__name__, kwargs["build_log"])
            response = await f(*args, **kwargs)
            update_metrics(metrics_id, response)
            return response

        @wraps(f)
        def sync_decorated_function(*args, **kwargs):
            metrics_id = add_new_metrics(f.__name__, kwargs["build_log"])
            response = f(*args, **kwargs)
            update_metrics(metrics_id, response)
            return response

        if inspect.iscoroutinefunction(f):
            return async_decorated_function
        return sync_decorated_function

    return decorator
