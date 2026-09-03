import os
import asyncio
import datetime
from enum import Enum
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated

from koji import ClientSession
from gitlab import Gitlab
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Depends,
    Header,
    Path,
    Request,
)
from fastapi.responses import Response as BasicResponse
import aiohttp
import sentry_sdk
from beeai_framework.backend import ChatModel

from logdetective.exceptions import (
    KojiInvalidTaskID,
    LogDetectiveInferenceError,
    RemoteLogError,
)
from logdetective.remote_log import RemoteLog
from logdetective.utils import (
    ContentSizeCheck,
    check_content_size,
    sanitize_artifact,
    get_version,
    SSRFProtectedResolver,
)

from logdetective.database.models.koji import KojiTaskAnalysis
from logdetective.database.models.exceptions import (
    KojiTaskAnalysisTimeoutError,
    KojiTaskNotAnalyzedError,
    KojiTaskNotFoundError,
)

from logdetective.agent.agent import analyze_artifacts
import logdetective.database.base

from logdetective.config import SERVER_CONFIG, LOG, get_chat_model
from logdetective.routes_gitlab import gitlab_router
from logdetective.koji import (
    get_failed_log_from_task as get_failed_log_from_koji_task,
)
from logdetective.metric import (
    track_request,
    add_new_metrics,
    update_metrics,
    requests_per_time,
    average_time_per_responses,
)
from logdetective.models import (
    ArtifactFile,
    RemoteArtifactFile,
    AnalysisRequest,
    Config,
    KojiInstanceConfig,
    KojiResponse,
    APIResponse,
    TimePeriod,
    MetricResponse,
)
from logdetective.database.models import EndpointType


LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)
API_TOKEN = os.environ.get("LOGDETECTIVE_TOKEN", None)


if sentry_dsn := SERVER_CONFIG.general.sentry_dsn:
    sentry_sdk.init(dsn=str(sentry_dsn), traces_sample_rate=1.0)


class ConnectionManager:
    """
    Manager for all connections and sesssions.
    """

    koji_connections: dict[str, ClientSession] = {}
    gitlab_connections: dict[str, Gitlab] = {}
    gitlab_http_sessions: dict[str, aiohttp.ClientSession] = {}

    async def initialize(self, service_config: Config):
        """Initialize all managed objects"""

        for connection, config in service_config.gitlab.instances.items():
            self.gitlab_connections[connection] = Gitlab(
                url=config.url,
                private_token=config.api_token,
                timeout=config.timeout,
            )
            self.gitlab_http_sessions[connection] = aiohttp.ClientSession(
                base_url=config.url,
                headers={"Authorization": f"Bearer {config.api_token}"},
                timeout=aiohttp.ClientTimeout(
                    total=config.timeout,
                    connect=3.07,
                ),
            )
        for connection, config in service_config.koji.instances.items():
            self.koji_connections[connection] = ClientSession(baseurl=config.xmlrpc_url)

    async def close(self):
        """Close all managed http sessions"""
        for session in self.gitlab_http_sessions.values():
            await session.close()


class KojiCallbackManager:
    """Manages callbacks used by Koji, with callbacks referenced by task id.

    Multiple callbacks can be assigned to a single task."""

    _callbacks: defaultdict[int, set[str]]

    def __init__(self) -> None:
        self._callbacks = defaultdict(set)

    def register_callback(self, task_id: int, callback: str):
        """Register a callback for a task"""
        self._callbacks[task_id].add(callback)

    def clear_callbacks(self, task_id: int):
        """Unregister a callback for a task"""
        try:
            del self._callbacks[task_id]
        except KeyError:
            pass

    def get_callbacks(self, task_id: int) -> set[str]:
        """Get the callbacks for a task"""
        return self._callbacks[task_id]


@asynccontextmanager
async def lifespan(fapp: FastAPI):
    """
    Establish one HTTP session
    """
    connector = None
    # Custom resolver covering Server-Side Request Forgery
    if SERVER_CONFIG.general.block_localhost_urls:
        connector = aiohttp.TCPConnector(
            resolver=SSRFProtectedResolver(),
        )

    fapp.http = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(
            total=int(LOG_SOURCE_REQUEST_TIMEOUT), connect=3.07
        ),
    )

    # Manager for connections and sessions
    fapp.state.connection_manager = ConnectionManager()

    await fapp.state.connection_manager.initialize(service_config=SERVER_CONFIG)

    # Koji callbacks
    fapp.state.koji_callback_manager = KojiCallbackManager()

    # Chat model for agent
    fapp.state.chat_model = get_chat_model(
        inference_config=SERVER_CONFIG.inference
    )

    # Ensure that the database is initialized.
    await logdetective.database.base.check()

    yield

    await fapp.state.connection_manager.close()
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
        LOG.info("Provided token does not match expected value.")
        raise HTTPException(status_code=401, detail=f"Token '{token}' not valid.")
    LOG.error("No authorization header provided but LOGDETECTIVE_TOKEN env var is set")
    raise HTTPException(status_code=401, detail="No token provided.")


app = FastAPI(
    title="Log Detective",
    contact={
        "name": "Log Detective developers",
        "url": "https://github.com/fedora-copr/logdetective",
        "email": "copr-devel@lists.fedorahosted.org",
    },
    license_info={
        "name": "Apache-2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    version=get_version(),
    dependencies=[Depends(requires_token_when_set)],
    lifespan=lifespan,
    swagger_ui_parameters={"operationsSorter": "alpha"},
)

if SERVER_CONFIG.gitlab.instances:
    app.include_router(gitlab_router)


async def get_artifacts_from_payload(
    payload: AnalysisRequest,
    http_session: aiohttp.ClientSession,
    request_size: int,
) -> dict[str, str | RemoteLog]:
    """Retrieve artifact contents based on the type of artifact.
    Raise ValueError on unsupported element types."""
    build_artifacts: dict[str, str | RemoteLog] = {}

    total_payload_size: int = request_size

    for artifact in payload.files:
        if isinstance(artifact, RemoteArtifactFile):
            remaining_limit = (
                SERVER_CONFIG.general.max_artifact_size - total_payload_size
            )
            if remaining_limit <= 0:
                raise HTTPException(
                    413, detail="Total size of submitted request is over the limit."
                )
            remote_log = RemoteLog(
                str(artifact.url), http_session, limit_bytes=remaining_limit
            )
            if SERVER_CONFIG.general.delay_artifact_download:
                LOG.info(
                    "Delaying download of artifact %s from %s until requested",
                    artifact.name,
                    artifact.url
                )
                # Size is enforced per-file via limit_bytes when
                # get_url_content() runs; total across deferred
                # artifacts is accepted optimistically.
                build_artifacts[artifact.name] = remote_log
            else:
                LOG.info("Downloading artifact %s from %s", artifact.name, artifact.url)
                try:
                    log_text = await remote_log.get_url_content()
                except RemoteLogError as ex:
                    raise HTTPException(
                        status_code=ex.status_code, detail=f"{ex}"
                    ) from ex
                build_artifacts[artifact.name] = log_text
                total_payload_size += remote_log.remote_log_size

        elif isinstance(artifact, ArtifactFile):
            LOG.info("Handling artifact %s as raw string", artifact.name)
            build_artifacts[artifact.name] = sanitize_artifact(artifact.content)
        else:
            raise ValueError(f"Invalid element type {type(artifact)}")

    total_payload_len = sum(
        len(content)
        for _, content in build_artifacts.items()
        if isinstance(content, str)
    )
    LOG.info(
        "Total artifact size from the obtained payload (in chars): %d "
        "Total payload size (in bytes): %d",
        total_payload_len,
        total_payload_size,
    )
    return build_artifacts


def validate_request_size(request: Request) -> int:
    """
    FastAPI Depend function checking request's Content-Length before loading body into memory.

    Note:
        In the case of URL requests, we limit the URL's content to 50 MiB.
        With the direct files raw log content, we limit the whole request size to 50 Mib,
        so this fails if all provided logs are under the limit, but exceed it together.

    Returns:
        Size of request in bytes

    Raises:
        HTTPException(411): If Content-Length header is missing or invalid
        HTTPException(413): If Content-Length exceeds maximum allowed size
    """
    size_check: ContentSizeCheck = check_content_size(
        request.headers, SERVER_CONFIG.general.max_artifact_size
    )
    if size_check.size_in_bytes is None:
        raise HTTPException(
            status_code=411, detail="Content-Length is missing or invalid."
        )
    if not size_check.proceed:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Content-Length is too large: "
                f"{size_check.size_in_bytes} B "
                f"({size_check.size_in_bytes / (1024 * 1024):.2f} MiB) > "
                f"{SERVER_CONFIG.general.max_artifact_size} B "
                f"({SERVER_CONFIG.general.max_artifact_size / (1024 * 1024):.2f} MiB)"
            ),
        )

    return size_check.size_in_bytes


@app.post("/analyze", response_model=APIResponse)
@track_request()
async def analyze(
    payload: AnalysisRequest,
    request: Request,
    http_session: aiohttp.ClientSession = Depends(get_http_session),
    request_size: int = Depends(validate_request_size),
):
    """
    Provide endpoint for analysis of artifacts. Artifacts can be submitted directly,
    or using URL. URL must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    artifacts = await get_artifacts_from_payload(
        payload, http_session, request_size=request_size
    )

    try:
        response = await analyze_artifacts(
            artifacts=artifacts,
            chat_model=request.app.state.chat_model,
            build_metadata=payload.build_metadata
        )
    except LogDetectiveInferenceError as exc:
        raise HTTPException(
            status_code=exc.http_status_code,
            detail=f"{type(exc).__doc__}: {exc}",
        ) from exc
    return response


@app.get(
    "/analyze/rpmbuild/koji/{koji_instance}/{task_id}",
    response_model=KojiResponse,
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
    response_model=KojiResponse,
)
async def analyze_rpmbuild_koji(
    koji_instance: Annotated[str, Path(title="The Koji instance to use")],
    task_id: Annotated[int, Path(title="The task ID to analyze")],
    request: Request,
    x_koji_token: Annotated[str, Header()] = "",
    x_koji_callback: Annotated[str, Header()] = "",
    background_tasks: BackgroundTasks = BackgroundTasks(),
):  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
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
        return BasicResponse(status_code=401)

    # Check if we already have a response for this task
    try:
        response = KojiTaskAnalysis.get_response_by_task_id(task_id)

    except KojiInvalidTaskID:
        # This task ID is malformed or out of range, so we will return a 400.
        response = BasicResponse(status_code=404, content="Invalid or unknown task ID.")

    except (KojiTaskNotFoundError, KojiTaskAnalysisTimeoutError):
        # Task not yet analyzed or it timed out, so we need to start the
        # analysis in the background and return a 202 (Accepted) error.

        koji_connection = request.app.state.connection_manager.koji_connections[
            koji_instance
        ]
        background_tasks.add_task(
            analyze_koji_task,
            task_id,
            koji_instance_config,
            koji_connection,
            request.app.state.koji_callback_manager,
            request.app.state.chat_model,
        )

        # If a callback URL is provided, we need to add it to the callbacks
        # table so that we can notify it when the analysis is complete.
        if x_koji_callback:
            request.app.state.koji_callback_manager.register_callback(
                task_id, x_koji_callback
            )

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


async def analyze_koji_task(
    task_id: int,
    koji_instance_config: KojiInstanceConfig,
    koji_connection: ClientSession,
    koji_callback_manager: KojiCallbackManager,
    chat_model: ChatModel,
):  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
    """Analyze a koji task and return the response"""

    # Get the log text from the koji task
    log_file_name, log_text = await get_failed_log_from_koji_task(
        koji_connection, task_id, max_size=SERVER_CONFIG.koji.max_artifact_size
    )
    log_text = sanitize_artifact(log_text)

    # We need to handle the metric tracking manually here, because we need
    # to retrieve the metric ID to associate it with the koji task analysis.

    metrics_id = await add_new_metrics(
        EndpointType.ANALYZE_KOJI_TASK,
        received_at=datetime.datetime.now(datetime.timezone.utc),
    )
    # We need to associate the metric ID with the koji task analysis.
    # This will create the new row without a response, which we will use as
    # an indicator that the analysis is in progress.
    await KojiTaskAnalysis.create_or_restart(
        koji_instance=koji_instance_config.xmlrpc_url,
        task_id=task_id,
        log_file_name=log_file_name,
    )
    try:
        response = await analyze_artifacts(
            {log_file_name: log_text}, chat_model=chat_model
        )
    except LogDetectiveInferenceError as exc:
        # The empty task will sit with null response_id until analysis_timeout elapses,
        # then callers get KojiTaskAnalysisTimeoutError -> handled as 503
        LOG.error("Not processing Koji task %d: %s: %s", task_id, type(exc).__name__, exc)
        return

    # Now that we have the response, we can update the metrics and mark the
    # koji task analysis as completed.
    await update_metrics(metrics_id, response)
    await KojiTaskAnalysis.add_response(task_id, metrics_id)

    # Notify any callbacks that the analysis is complete.
    for callback in koji_callback_manager.get_callbacks(task_id):
        LOG.info("Notifying callback %s of task %d completion", callback, task_id)
        asyncio.create_task(send_koji_callback(callback, task_id))

    # Now that it's sent, we can clear the callbacks for this task.
    koji_callback_manager.clear_callbacks(task_id)

    return response


async def send_koji_callback(callback: str, task_id: int):
    """Send a callback to the specified URL with the task ID and log file name."""
    connector = None
    if SERVER_CONFIG.general.block_localhost_urls:
        connector = aiohttp.TCPConnector(
            resolver=SSRFProtectedResolver()
        )
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(callback, json={"task_id": task_id}):
            pass


@app.get("/version", response_class=BasicResponse)
async def get_version_wrapper():
    """Get the version of logdetective"""
    return BasicResponse(content=get_version())


class MetricRoute(str, Enum):
    """Routes for metrics"""

    ANALYZE = "analyze"
    ANALYZE_GITLAB_JOB = "analyze-gitlab"


class MetricType(str, Enum):
    """Type of metric retrieved"""

    REQUESTS = "requests"
    RESPONSES = "responses"
    ALL = "all"


ROUTE_TO_ENDPOINT_TYPES = {
    MetricRoute.ANALYZE: EndpointType.ANALYZE,
    MetricRoute.ANALYZE_GITLAB_JOB: EndpointType.ANALYZE_GITLAB_JOB,
}


@app.get("/metrics/{route}/", response_model=MetricResponse)
@app.get("/metrics/{route}/{metric_type}", response_model=MetricResponse)
async def get_metrics(
    route: MetricRoute,
    metric_type: MetricType = MetricType.ALL,
    period_since_now: TimePeriod = Depends(TimePeriod),
):
    """Get an handler returning statistics for the specified endpoint and metric_type."""
    endpoint_type = ROUTE_TO_ENDPOINT_TYPES[route]

    async def handler() -> MetricResponse:
        """Return statistics for the specified endpoint and metric type."""
        statistics = []
        if metric_type == MetricType.ALL:
            statistics.append(await requests_per_time(period_since_now, endpoint_type))
            statistics.append(
                await average_time_per_responses(period_since_now, endpoint_type)
            )
            return MetricResponse(time_series=statistics)
        if metric_type == MetricType.REQUESTS:
            statistics.append(await requests_per_time(period_since_now, endpoint_type))
        elif metric_type == MetricType.RESPONSES:
            statistics.append(
                await average_time_per_responses(period_since_now, endpoint_type)
            )
        return MetricResponse(time_series=statistics)

    if endpoint_type == EndpointType.ANALYZE_GITLAB_JOB and not SERVER_CONFIG.gitlab.instances:
        raise HTTPException(
            status_code=404,
            detail="No gitlab instance configured, skipping metrics collection."
        )

    descriptions = {
        MetricType.REQUESTS: (
            "Get statistics for the requests received in the given period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
        MetricType.RESPONSES: (
            "Get statistics for responses given in the specified period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
        MetricType.ALL: (
            "Get statistics for requests and responses in the given period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
    }
    handler.__doc__ = descriptions[metric_type]

    return await handler()
