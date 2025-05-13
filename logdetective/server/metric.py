import io
import inspect
import datetime

from typing import Union
from functools import wraps

import aiohttp

from starlette.responses import StreamingResponse
from logdetective.server import models
from logdetective.remote_log import RemoteLog
from logdetective.server.config import LOG
from logdetective.server.compressors import LLMResponseCompressor, RemoteLogCompressor
from logdetective.server.database.models import EndpointType, AnalyzeRequestMetrics


async def add_new_metrics(
    api_name: str,
    url: str,
    http_session: aiohttp.ClientSession,
    received_at: datetime.datetime = None,
    compressed_log_content: io.BytesIO = None,
) -> int:
    """Add a new database entry for a received request.

    This will store the time when this function is called,
    the endpoint from where the request was received,
    and the log (in a zip format) for which analysis is requested.
    """
    remote_log = RemoteLog(url, http_session)
    compressed_log_content = (
        compressed_log_content or await RemoteLogCompressor(remote_log).zip_content()
    )
    return AnalyzeRequestMetrics.create(
        endpoint=EndpointType(api_name),
        compressed_log=compressed_log_content,
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
        response.explanation, models.Explanation
    ):
        response_length = len(response.explanation.text)
    response_certainty = (
        response.response_certainty if hasattr(response, "response_certainty") else None
    )
    AnalyzeRequestMetrics.update(
        metrics_id,
        response_sent_at,
        response_length,
        response_certainty,
        compressed_response,
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
                name if name else f.__name__, log_url, kwargs["http_session"]
            )
            response = await f(*args, **kwargs)
            update_metrics(metrics_id, response)
            return response

        if inspect.iscoroutinefunction(f):
            return async_decorated_function
        raise NotImplementedError("An async coroutine is needed")

    return decorator
