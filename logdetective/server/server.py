import os
import asyncio
import datetime
from enum import Enum
from contextlib import asynccontextmanager
from typing import Annotated
from io import BytesIO

import matplotlib
import matplotlib.figure
import matplotlib.pyplot
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Depends,
    Header,
    Path,
    Request,
)

from fastapi.responses import StreamingResponse
from fastapi.responses import Response as BasicResponse
import aiohttp
import sentry_sdk

from logdetective.server.exceptions import KojiInvalidTaskID

from logdetective.server.database.models.koji import KojiTaskAnalysis
from logdetective.server.database.models.exceptions import (
    KojiTaskAnalysisTimeoutError,
    KojiTaskNotAnalyzedError,
    KojiTaskNotFoundError,
)

import logdetective.server.database.base

from logdetective.server.config import SERVER_CONFIG, LOG
from logdetective.server.koji import (
    get_failed_log_from_task as get_failed_log_from_koji_task,
)
from logdetective.remote_log import RemoteLog
from logdetective.server.llm import (
    perform_staged_analysis,
    perfrom_analysis,
    perform_analyis_stream,
)
from logdetective.server.gitlab import process_gitlab_job_event
from logdetective.server.metric import track_request, add_new_metrics, update_metrics
from logdetective.server.models import (
    BuildLog,
    EmojiHook,
    JobHook,
    KojiInstanceConfig,
    KojiStagedResponse,
    Response,
    StagedResponse,
    TimePeriod,
)
from logdetective.server import plot as plot_engine
from logdetective.server.database.models import (
    EndpointType,
    Forge,
)
from logdetective.server.emoji import (
    collect_emojis,
    collect_emojis_for_mr,
)
from logdetective.server.compressors import RemoteLogCompressor


LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)
API_TOKEN = os.environ.get("LOGDETECTIVE_TOKEN", None)


if sentry_dsn := SERVER_CONFIG.general.sentry_dsn:
    sentry_sdk.init(dsn=str(sentry_dsn), traces_sample_rate=1.0)


@asynccontextmanager
async def lifespan(fapp: FastAPI):
    """
    Establish one HTTP session
    """
    fapp.http = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=int(LOG_SOURCE_REQUEST_TIMEOUT), connect=3.07
        )
    )

    # Ensure that the database is initialized.
    logdetective.server.database.base.init()

    # Start the background task scheduler for collecting emojis
    asyncio.create_task(schedule_collect_emojis_task())

    yield

    await fapp.http.close()


async def get_http_session(request: Request) -> aiohttp.ClientSession:
    """
    Return the single aiohttp ClientSession for this app
    """
    return request.app.http


def requires_token_when_set(authorization: Annotated[str | None, Header()] = None):
    """
    FastAPI Depend function that expects a header named Authorization

    If LOGDETECTIVE_TOKEN env var is set, validate the client-supplied token
    otherwise ignore it
    """
    if not API_TOKEN:
        LOG.info("LOGDETECTIVE_TOKEN env var not set, authorization disabled")
        # no token required, means local dev environment
        return
    if authorization:
        try:
            token = authorization.split(" ", 1)[1]
        except (ValueError, IndexError) as ex:
            LOG.warning(
                "Authorization header has invalid structure '%s', it should be 'Bearer TOKEN'",
                authorization,
            )
            # eat the exception and raise 401 below
            raise HTTPException(
                status_code=401,
                detail=f"Invalid authorization, HEADER '{authorization}' not valid.",
            ) from ex
        if token == API_TOKEN:
            return
        LOG.info("Provided token '%s' does not match expected value.", token)
        raise HTTPException(status_code=401, detail=f"Token '{token}' not valid.")
    LOG.error("No authorization header provided but LOGDETECTIVE_TOKEN env var is set")
    raise HTTPException(status_code=401, detail="No token provided.")


app = FastAPI(dependencies=[Depends(requires_token_when_set)], lifespan=lifespan)


@app.post("/analyze", response_model=Response)
@track_request()
async def analyze_log(
    build_log: BuildLog, http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    remote_log = RemoteLog(build_log.url, http_session)
    log_text = await remote_log.process_url()

    return await perfrom_analysis(log_text)


@app.post("/analyze/staged", response_model=StagedResponse)
@track_request()
async def analyze_log_staged(
    build_log: BuildLog, http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    remote_log = RemoteLog(build_log.url, http_session)
    log_text = await remote_log.process_url()

    return await perform_staged_analysis(log_text)


@app.get(
    "/analyze/rpmbuild/koji/{koji_instance}/{task_id}",
    response_model=KojiStagedResponse,
)
async def get_koji_task_analysis(
    koji_instance: Annotated[str, Path(title="The Koji instance to use")],
    task_id: Annotated[int, Path(title="The task ID to analyze")],
    x_koji_token: Annotated[str, Header()] = "",
):
    """Provide endpoint for retrieving log file analysis of a Koji task"""

    try:
        koji_instance_config = SERVER_CONFIG.koji.instances[koji_instance]
    except KeyError:
        # This Koji instance is not configured, so we will return a 404.
        return BasicResponse(status_code=404, content="Unknown Koji instance.")

    # This should always be available in a production environment.
    # In a testing environment, the tokens list may be empty, in which case
    # it will just proceed.
    if koji_instance_config.tokens and x_koji_token not in koji_instance_config.tokens:
        # (Unauthorized) error.
        return BasicResponse(x_koji_token, status_code=401)

    # Check if we have a response for this task
    try:
        return KojiTaskAnalysis.get_response_by_task_id(task_id)

    except (KojiInvalidTaskID, KojiTaskNotFoundError):
        # This task ID is malformed, out of range, or not found, so we will
        # return a 404.
        return BasicResponse(status_code=404)

    except KojiTaskAnalysisTimeoutError:
        # Task analysis has timed out, so we assume that the request was lost
        # and that we need to start another analysis.
        # There isn't a fully-appropriate error code for this, so we'll use
        # 503 (Service Unavailable) as our best option.
        return BasicResponse(
            status_code=503, content="Task analysis timed out, please retry."
        )

    except KojiTaskNotAnalyzedError:
        # Its still running, so we need to return a 202
        # (Accepted) code to let the client know to keep waiting.
        return BasicResponse(
            status_code=202, content=f"Analysis still in progress for task {task_id}"
        )


@app.post(
    "/analyze/rpmbuild/koji/{koji_instance}/{task_id}",
    response_model=KojiStagedResponse,
)
async def analyze_rpmbuild_koji(
    koji_instance: Annotated[str, Path(title="The Koji instance to use")],
    task_id: Annotated[int, Path(title="The task ID to analyze")],
    x_koji_token: Annotated[str, Header()] = "",
    x_koji_callback: Annotated[str, Header()] = "",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Provide endpoint for retrieving log file analysis of a Koji task"""

    try:
        koji_instance_config = SERVER_CONFIG.koji.instances[koji_instance]
    except KeyError:
        # This Koji instance is not configured, so we will return a 404.
        return BasicResponse(status_code=404, content="Unknown Koji instance.")

    # This should always be available in a production environment.
    # In a testing environment, the tokens list may be empty, in which case
    # it will just proceed.
    if koji_instance_config.tokens and x_koji_token not in koji_instance_config.tokens:
        # (Unauthorized) error.
        return BasicResponse(x_koji_token, status_code=401)

    # Check if we already have a response for this task
    try:
        response = KojiTaskAnalysis.get_response_by_task_id(task_id)

    except KojiInvalidTaskID:
        # This task ID is malformed or out of range, so we will return a 400.
        response = BasicResponse(status_code=404, content="Invalid or unknown task ID.")

    except (KojiTaskNotFoundError, KojiTaskAnalysisTimeoutError):
        # Task not yet analyzed or it timed out, so we need to start the
        # analysis in the background and return a 202 (Accepted) error.

        background_tasks.add_task(
            analyze_koji_task,
            task_id,
            koji_instance_config,
        )

        # If a callback URL is provided, we need to add it to the callbacks
        # table so that we can notify it when the analysis is complete.
        if x_koji_callback:
            koji_instance_config.register_callback(task_id, x_koji_callback)

        response = BasicResponse(
            status_code=202, content=f"Beginning analysis of task {task_id}"
        )

    except KojiTaskNotAnalyzedError:
        # Its still running, so we need to return a 202
        # (Accepted) error.
        response = BasicResponse(
            status_code=202, content=f"Analysis still in progress for task {task_id}"
        )

    return response


async def analyze_koji_task(task_id: int, koji_instance_config: KojiInstanceConfig):
    """Analyze a koji task and return the response"""

    # Get the log text from the koji task
    koji_conn = koji_instance_config.get_connection()
    log_file_name, log_text = await get_failed_log_from_koji_task(
        koji_conn, task_id, max_size=SERVER_CONFIG.koji.max_artifact_size
    )

    # We need to handle the metric tracking manually here, because we need
    # to retrieve the metric ID to associate it with the koji task analysis.

    metrics_id = await add_new_metrics(
        "analyze_koji_task",
        log_text,
        received_at=datetime.datetime.now(datetime.timezone.utc),
        compressed_log_content=RemoteLogCompressor.zip_text(log_text),
    )

    # We need to associate the metric ID with the koji task analysis.
    # This will create the new row without a response, which we will use as
    # an indicator that the analysis is in progress.
    KojiTaskAnalysis.create_or_restart(
        koji_instance=koji_instance_config.xmlrpc_url,
        task_id=task_id,
        log_file_name=log_file_name,
    )
    response = await perform_staged_analysis(log_text)

    # Now that we have the response, we can update the metrics and mark the
    # koji task analysis as completed.
    update_metrics(metrics_id, response)
    KojiTaskAnalysis.add_response(task_id, metrics_id)

    # Notify any callbacks that the analysis is complete.
    for callback in koji_instance_config.get_callbacks(task_id):
        LOG.info("Notifying callback %s of task %d completion", callback, task_id)
        asyncio.create_task(send_koji_callback(callback, task_id))

    # Now that it's sent, we can clear the callbacks for this task.
    koji_instance_config.clear_callbacks(task_id)

    return response


async def send_koji_callback(callback: str, task_id: int):
    """Send a callback to the specified URL with the task ID and log file name."""
    async with aiohttp.ClientSession() as session:
        async with session.post(callback, json={"task_id": task_id}):
            pass


@app.get("/queue/print")
async def queue_print(msg: str):
    """Debug endpoint to test the LLM request queue"""
    LOG.info("Will print %s", msg)

    result = await async_log(msg)

    LOG.info("Printed %s and returned it", result)


async def async_log(msg):
    """Debug function to test the LLM request queue"""
    async with SERVER_CONFIG.inference.get_limiter():
        LOG.critical(msg)
    return msg


@app.post("/analyze/stream", response_class=StreamingResponse)
@track_request()
async def analyze_log_stream(
    build_log: BuildLog, http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Stream response endpoint for Logdetective.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    remote_log = RemoteLog(build_log.url, http_session)
    log_text = await remote_log.process_url()
    try:
        stream = perform_analyis_stream(log_text)
    except aiohttp.ClientResponseError as ex:
        raise HTTPException(
            status_code=400,
            detail="HTTP Error while getting response from inference server "
            f"[{ex.status}] {ex.message}",
        ) from ex

    return StreamingResponse(stream)


def is_valid_webhook_secret(forge, x_gitlab_token):
    """Check whether the provided x_gitlab_token matches the webhook secret
    specified in the configuration"""

    gitlab_cfg = SERVER_CONFIG.gitlab.instances[forge.value]

    if not gitlab_cfg.webhook_secrets:
        # No secrets specified, so don't bother validating.
        # This is mostly to be used for development.
        return True

    if x_gitlab_token in gitlab_cfg.webhook_secrets:
        return True

    return False


@app.post("/webhook/gitlab/job_events")
async def receive_gitlab_job_event_webhook(
    job_hook: JobHook,
    background_tasks: BackgroundTasks,
    x_gitlab_instance: Annotated[str | None, Header()],
    x_gitlab_token: Annotated[str | None, Header()] = None,
):
    """Webhook endpoint for receiving job_events notifications from GitLab
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    lists the full specification for the messages sent for job events."""

    try:
        forge = Forge(x_gitlab_instance)
    except ValueError:
        LOG.critical("%s is not a recognized forge. Ignoring.", x_gitlab_instance)
        return BasicResponse(status_code=400)

    if not is_valid_webhook_secret(forge, x_gitlab_token):
        # This request could not be validated, so return a 401
        # (Unauthorized) error.
        return BasicResponse(status_code=401)

    # Handle the message in the background so we can return 204 immediately
    gitlab_cfg = SERVER_CONFIG.gitlab.instances[forge.value]
    background_tasks.add_task(
        process_gitlab_job_event,
        gitlab_cfg,
        forge,
        job_hook,
    )

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)


# A lookup table for whether we are currently processing a given merge request
# The key is the tuple (Forge, ProjectID, MRID) and the value is a boolean
# indicating whether we need to re-trigger the lookup immediately after
# completion due to another request coming in during processing.
# For example: {("https://gitlab.example.com", 23, 2): False}
emoji_lookup = {}


@app.post("/webhook/gitlab/emoji_events")
async def receive_gitlab_emoji_event_webhook(
    x_gitlab_instance: Annotated[str | None, Header()],
    x_gitlab_token: Annotated[str | None, Header()],
    emoji_hook: EmojiHook,
    background_tasks: BackgroundTasks,
):
    """Webhook endpoint for receiving emoji event notifications from Gitlab
    https://docs.gitlab.com/user/project/integrations/webhook_events/#emoji-events
    lists the full specification for the messages sent for emoji events"""

    try:
        forge = Forge(x_gitlab_instance)
    except ValueError:
        LOG.critical("%s is not a recognized forge. Ignoring.", x_gitlab_instance)
        return BasicResponse(status_code=400)

    if not is_valid_webhook_secret(forge, x_gitlab_token):
        # This request could not be validated, so return a 401
        # (Unauthorized) error.
        return BasicResponse(status_code=401)

    if not emoji_hook.merge_request:
        # This is not a merge request event. It is probably an emoji applied
        # to some other "awardable" entity. Just ignore it and return.
        LOG.debug("Emoji event is not related to a merge request. Ignoring.")
        return BasicResponse(status_code=204)

    # We will re-process all the emojis on this merge request, to ensure that
    # we haven't missed any messages, since webhooks do not provide delivery
    # guarantees.

    # Check whether this request is already in progress.
    # We are single-threaded, so we can guarantee that the table won't change
    # between here and when we schedule the lookup.
    key = (
        forge,
        emoji_hook.merge_request.target_project_id,
        emoji_hook.merge_request.iid,
    )
    if key in emoji_lookup:
        # It's already in progress, so we do not want to start another pass
        # concurrently. We'll set the value to True to indicate that we should
        # re-enqueue this lookup after the currently-running one concludes. It
        # is always safe to set this to True, even if it's already True. If
        # multiple requests come in during processing, we only need to re-run
        # it a single time, since it will pick up all the ongoing changes. The
        # worst-case situation is the one where we receive new requests just
        # after processing starts, which will cause the cycle to repeat again.
        # This should be very infrequent, as emoji events are computationally
        # rare and very quick to process.
        emoji_lookup[key] = True
        LOG.info("MR Emojis already being processed for %s. Rescheduling.", key)
        return BasicResponse(status_code=204)

    # Inform the lookup table that we are processing this emoji
    emoji_lookup[key] = False

    # Create a background task to process the emojis on this Merge Request.
    background_tasks.add_task(
        schedule_emoji_collection_for_mr,
        forge,
        emoji_hook.merge_request.target_project_id,
        emoji_hook.merge_request.iid,
        background_tasks,
    )

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)


async def schedule_emoji_collection_for_mr(
    forge: Forge, project_id: int, mr_iid: int, background_tasks: BackgroundTasks
):
    """Background task to update the database on emoji reactions"""

    key = (forge, project_id, mr_iid)

    # FIXME: Look up the connection from the Forge  # pylint: disable=fixme
    gitlab_conn = SERVER_CONFIG.gitlab.instances[forge.value]

    LOG.debug("Looking up emojis for %s, %d, %d", forge, project_id, mr_iid)
    await collect_emojis_for_mr(project_id, mr_iid, gitlab_conn)

    # Check whether we've been asked to re-schedule this lookup because
    # another request came in while it was processing.
    if emoji_lookup[key]:
        # The value is Truthy, which tells us to re-schedule
        # Reset the boolean value to indicate that we're underway again.
        emoji_lookup[key] = False
        background_tasks.add_task(
            schedule_emoji_collection_for_mr,
            forge,
            project_id,
            mr_iid,
            background_tasks,
        )
        return

    # We're all done, so clear this entry out of the lookup
    del emoji_lookup[key]


def _svg_figure_response(fig: matplotlib.figure.Figure):
    """Create a response with the given svg figure."""
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    matplotlib.pyplot.close(fig)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/svg+xml",
        headers={"Content-Disposition": "inline; filename=plot.svg"},
    )


def _multiple_svg_figures_response(figures: list[matplotlib.figure.Figure]):
    """Create a response with multiple svg figures."""
    svg_contents = []
    for i, fig in enumerate(figures):
        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        matplotlib.pyplot.close(fig)
        buf.seek(0)
        svg_contents.append(buf.read().decode("utf-8"))

    html_content = "<html><body>\n"
    for i, svg in enumerate(svg_contents):
        html_content += f"<div id='figure-{i}'>\n{svg}\n</div>\n"
    html_content += "</body></html>"

    return BasicResponse(content=html_content, media_type="text/html")


class MetricRoute(str, Enum):
    """Routes for metrics"""

    ANALYZE = "analyze"
    ANALYZE_STAGED = "analyze-staged"
    ANALYZE_GITLAB_JOB = "analyze-gitlab"


class Plot(str, Enum):
    """Type of served plots"""

    REQUESTS = "requests"
    RESPONSES = "responses"
    EMOJIS = "emojis"
    BOTH = ""


ROUTE_TO_ENDPOINT_TYPES = {
    MetricRoute.ANALYZE: EndpointType.ANALYZE,
    MetricRoute.ANALYZE_STAGED: EndpointType.ANALYZE_STAGED,
    MetricRoute.ANALYZE_GITLAB_JOB: EndpointType.ANALYZE_GITLAB_JOB,
}


@app.get("/metrics/{route}/", response_class=StreamingResponse)
@app.get("/metrics/{route}/{plot}", response_class=StreamingResponse)
async def get_metrics(
    route: MetricRoute,
    plot: Plot = Plot.BOTH,
    period_since_now: TimePeriod = Depends(TimePeriod),
):
    """Get an handler for visualize statistics for the specified endpoint and plot."""
    endpoint_type = ROUTE_TO_ENDPOINT_TYPES[route]

    async def handler():
        """Show statistics for the specified endpoint and plot."""
        if plot == Plot.REQUESTS:
            fig = plot_engine.requests_per_time(period_since_now, endpoint_type)
            return _svg_figure_response(fig)
        if plot == Plot.RESPONSES:
            fig = plot_engine.average_time_per_responses(
                period_since_now, endpoint_type
            )
            return _svg_figure_response(fig)
        if plot == Plot.EMOJIS:
            fig = plot_engine.emojis_per_time(period_since_now)
            return _svg_figure_response(fig)
        # BOTH
        fig_requests = plot_engine.requests_per_time(period_since_now, endpoint_type)
        fig_responses = plot_engine.average_time_per_responses(
            period_since_now, endpoint_type
        )
        fig_emojis = plot_engine.emojis_per_time(period_since_now)
        return _multiple_svg_figures_response([fig_requests, fig_responses, fig_emojis])

    descriptions = {
        Plot.REQUESTS: (
            "Show statistics for the requests received in the given period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
        Plot.RESPONSES: (
            "Show statistics for responses given in the specified period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
        Plot.EMOJIS: (
            "Show statistics for emoji feedback in the specified period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
        Plot.BOTH: (
            "Show statistics for requests and responses in the given period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
    }
    handler.__doc__ = descriptions[plot]

    return await handler()


async def collect_emoji_task():
    """Collect emoji feedback.
    Query only comments created in the last year.
    """

    for instance in SERVER_CONFIG.gitlab.instances.values():
        LOG.info(
            "Collect emoji feedback for %s started at %s",
            instance.url,
            datetime.datetime.now(datetime.timezone.utc),
        )
        await collect_emojis(instance.get_connection(), TimePeriod(weeks=54))
        LOG.info(
            "Collect emoji feedback finished at %s",
            datetime.datetime.now(datetime.timezone.utc),
        )


async def schedule_collect_emojis_task():
    """Schedule the collect_emojis_task to run on a configured interval"""
    while True:
        seconds_until_run = SERVER_CONFIG.general.collect_emojis_interval
        LOG.info("Collect emojis in %d seconds", seconds_until_run)
        await asyncio.sleep(seconds_until_run)

        try:
            await collect_emoji_task()
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOG.exception("Error in collect_emoji_task: %s", e)
