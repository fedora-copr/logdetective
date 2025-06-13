import os
import asyncio
import datetime
from enum import Enum
from contextlib import asynccontextmanager
from typing import Annotated
from io import BytesIO

import matplotlib
import matplotlib.pyplot
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request

from fastapi.responses import StreamingResponse
from fastapi.responses import Response as BasicResponse
import aiohttp
import sentry_sdk

import logdetective.server.database.base

from logdetective.utils import (
    compute_certainty,
    format_snippets,
    prompt_to_messages,
)

from logdetective.server.config import SERVER_CONFIG, PROMPT_CONFIG, LOG
from logdetective.remote_log import RemoteLog
from logdetective.server.llm import (
    mine_logs,
    perform_staged_analysis,
    submit_text,
)
from logdetective.server.gitlab import process_gitlab_job_event
from logdetective.server.metric import track_request
from logdetective.server.models import (
    BuildLog,
    EmojiHook,
    JobHook,
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


def requires_token_when_set(authentication: Annotated[str | None, Header()] = None):
    """
    FastAPI Depend function that expects a header named Authentication

    If LOGDETECTIVE_TOKEN env var is set, validate the client-supplied token
    otherwise ignore it
    """
    if not API_TOKEN:
        LOG.info("LOGDETECTIVE_TOKEN env var not set, authentication disabled")
        # no token required, means local dev environment
        return
    token = None
    if authentication:
        try:
            token = authentication.split(" ", 1)[1]
        except (ValueError, IndexError):
            LOG.warning(
                "Authentication header has invalid structure (%s), it should be 'Bearer TOKEN'",
                authentication,
            )
            # eat the exception and raise 401 below
            token = None
        if token == API_TOKEN:
            return
    LOG.info(
        "LOGDETECTIVE_TOKEN env var is set (%s), clien token = %s", API_TOKEN, token
    )
    raise HTTPException(status_code=401, detail=f"Token {token} not valid.")


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
    log_summary = mine_logs(log_text)
    log_summary = format_snippets(log_summary)
    messages = prompt_to_messages(
        PROMPT_CONFIG.prompt_template.format(log_summary),
        PROMPT_CONFIG.default_system_prompt,
        SERVER_CONFIG.inference.system_role,
        SERVER_CONFIG.inference.user_role,
    )
    response = await submit_text(
        messages,
        inference_cfg=SERVER_CONFIG.inference,
    )
    certainty = 0

    if response.logprobs is not None:
        try:
            certainty = compute_certainty(response.logprobs)
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n{response.logprobs}",
            ) from ex

    return Response(explanation=response, response_certainty=certainty)


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
    log_summary = mine_logs(log_text)
    log_summary = format_snippets(log_summary)
    messages = prompt_to_messages(
        PROMPT_CONFIG.prompt_template.format(log_summary),
        PROMPT_CONFIG.default_system_prompt,
        SERVER_CONFIG.inference.system_role,
        SERVER_CONFIG.inference.user_role,
    )
    try:
        stream = submit_text(
            messages,
            inference_cfg=SERVER_CONFIG.inference,
            stream=True,
        )
    except aiohttp.ClientResponseError as ex:
        raise HTTPException(
            status_code=400,
            detail="HTTP Error while getting response from inference server "
            f"[{ex.status}] {ex.message}",
        ) from ex

    # we need to figure out a better response here, this is how it looks rn:
    # b'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],
    #   "created":1744818071,"id":"chatcmpl-c9geTxNcQO7M9wR...
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
        await collect_emojis(instance.get_connection(), TimePeriod(weeks="54"))
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
