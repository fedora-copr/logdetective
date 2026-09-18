"""FastAPI surface for durable asynchronous Log Detective operations."""

from __future__ import annotations

import os
import secrets
from enum import Enum
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated, Optional
from uuid import UUID, uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    Request,
    Response,
)
from fastapi.responses import Response as BasicResponse
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, ValidationError
from logdetective.compressors import LLMResponseCompressor

from logdetective.remote_log import RemoteLog
from logdetective.utils import (
    ContentSizeCheck,
    check_content_size,
    get_version,
    init_sentry,
    load_api_tokens,
)

from logdetective.database.models.tasks import (
    TaskAnalysis,
    TaskType,
    ACTIVE_STATES,
    AnalysisState,
)
from logdetective.database.models.exceptions import (
    AnalysisTaskNotFoundError,
    TaskConflictError,
    TaskTerminalError,
)

import logdetective.database.base

from logdetective.config import SERVER_CONFIG
from logdetective.routes_gitlab import gitlab_router
from logdetective.metric import requests_statistics
from logdetective.models import (
    RemoteArtifactFile,
    AnalysisRequest,
    KojiResponse,
    APIResponse,
    MetricResponse,
    KojiAnalysisRequest,
    KojiTaskMetadata,
    TaskError,
    TaskResponse,
)
from logdetective.database.models import EndpointType, TimePeriod
from logdetective.tasks import analyze_generic, analyze_koji
from logdetective.tasks import app as task_app


API_TOKENS_PATH = os.environ.get("LOGDETECTIVE_TOKENS_FILE")
API_TOKENS = load_api_tokens(API_TOKENS_PATH)
BEARER_SCHEME = HTTPBearer(auto_error=False)


init_sentry()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Open database-backed API resources; inference belongs to workers."""
    await logdetective.database.base.check()
    await task_app.open_async()
    try:
        yield
    finally:
        await task_app.close_async()


async def authenticate_api_token(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(BEARER_SCHEME)
    ],
) -> str | None:
    """Authenticate a bearer token and attach its non-secret name to the request."""
    request.state.api_token_name = None
    if API_TOKENS is None:
        return None

    if credentials is not None:
        supplied_token = credentials.credentials.encode("utf-8")
        for name, expected_token in API_TOKENS.root.items():
            expected_value = expected_token.get_secret_value().encode("utf-8")
            if secrets.compare_digest(supplied_token, expected_value):
                request.state.api_token_name = name
                return name

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


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
    dependencies=[Depends(authenticate_api_token)],
    lifespan=lifespan,
    swagger_ui_parameters={"operationsSorter": "alpha"},
)

if SERVER_CONFIG.gitlab.instances:
    app.include_router(gitlab_router)


def validate_request_size(request: Request) -> int:
    """Reject requests without a bounded, valid Content-Length."""
    size_check: ContentSizeCheck = check_content_size(
        request.headers, SERVER_CONFIG.general.max_artifact_size
    )
    if size_check.size_in_bytes is None:
        raise HTTPException(411, detail="Content-Length is missing or invalid.")
    if not size_check.proceed:
        raise HTTPException(413, detail="Content-Length is too large.")
    return size_check.size_in_bytes


def task_representation(task: TaskAnalysis) -> TaskResponse:
    """Build the stable public response envelope for an application task.

    Args:
        task: Persisted application task whose current state should be exposed.

    Returns:
        Public task state containing a result or safe error when available.

    Raises:
        RuntimeError: If a completed task lacks its required response or Koji
            metadata, or if an internal GitLab task is exposed publicly.
    """
    result: APIResponse | KojiResponse | None = None
    if task.state == AnalysisState.DONE:
        if task.response is None:
            raise RuntimeError("Completed task has no persisted response")
        response = LLMResponseCompressor.unzip(task.response)
        if task.task_type == TaskType.KOJI:
            if task.task_metadata is None:
                raise RuntimeError("Completed Koji task has no metadata")
            try:
                metadata = KojiTaskMetadata.model_validate(task.task_metadata)
            except ValidationError as exc:
                raise RuntimeError(
                    "Completed Koji task metadata is invalid"
                ) from exc
            result = KojiResponse(
                task_id=metadata.task_id,
                log_file_name=metadata.log_file_name,
                response=response,
            )
        else:
            result = response
    error = None
    if task.error_code is not None:
        error = TaskError(
            code=task.error_code,
            message=task.error_message or "Analysis failed",
        )
    if task.task_type == TaskType.GITLAB:
        raise RuntimeError("GitLab webhook tasks do not have a public representation")
    return TaskResponse(
        id=task.task_id,
        taskType=task.task_type,
        createdAt=task.request_received_at,
        status=task.state,
        error=error,
        result=result,
    )


async def accept_task(
    payload: AnalysisRequest | KojiAnalysisRequest,
    request: Request,
    task_type: TaskType,
    request_size: int,
) -> JSONResponse:
    """Admit an analysis task and construct its HTTP 202 response.

    Args:
        payload: Validated generic or Koji analysis request.
        request: FastAPI request carrying the authenticated token identity and URL
            router.
        task_type: Kind of analysis to enqueue.
        request_size: Validated request-body size in bytes.

    Returns:
        A ``202 Accepted`` response containing the stable task envelope, polling
        location, retry interval, and cache policy. Idempotent retries return the
        existing task through the same response contract.

    Raises:
        HTTPException: With status 409 when the public identifier conflicts with a
            different request.
    """
    public_id = payload.id or uuid4()
    data = payload.model_dump(mode="json", by_alias=True)
    data["id"] = str(public_id)
    deferrable = analyze_generic if task_type == TaskType.GENERIC else analyze_koji
    endpoint = (
        EndpointType.ANALYZE
        if task_type == TaskType.GENERIC
        else EndpointType.ANALYZE_KOJI_TASK
    )
    try:
        task, _ = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name=request.state.api_token_name,
            task_type=task_type,
            input_payload=data,
            request_size=request_size,
            deferrable_task=deferrable,
            endpoint=endpoint,
        )
    except TaskConflictError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    representation = task_representation(task)
    return JSONResponse(
        status_code=202,
        content=representation.model_dump(mode="json", by_alias=True),
        headers={
            "Location": str(request.url_for("get_analysis_task", task_id=task.task_id)),
            "Retry-After": str(SERVER_CONFIG.task_queue.retry_after),
            "Cache-Control": "no-store",
        },
    )


@app.post("/analyze", response_model=TaskResponse, status_code=202)
async def analyze(
    payload: AnalysisRequest,
    request: Request,
    request_size: int = Depends(validate_request_size),
) -> JSONResponse:
    """Durably submit generic artifact analysis without downloading in the API."""
    for artifact in payload.files:
        if isinstance(artifact, RemoteArtifactFile) and not RemoteLog.is_valid_url(
            str(artifact.url)
        ):
            raise HTTPException(400, detail="Invalid artifact URL")
    return await accept_task(payload, request, TaskType.GENERIC, request_size)


@app.post("/analyze/rpmbuild/koji", response_model=TaskResponse, status_code=202)
async def analyze_rpmbuild_koji(
    payload: KojiAnalysisRequest,
    request: Request,
    x_koji_token: Annotated[str, Header()] = "",
    request_size: int = Depends(validate_request_size),
) -> JSONResponse:
    """Durably submit one build from a configured Koji instance."""
    instance = SERVER_CONFIG.koji.instances.get(payload.koji_instance)
    if instance is None:
        raise HTTPException(404, detail="Unknown Koji instance")
    supplied = x_koji_token.encode("utf-8")
    if instance.tokens and not any(
        secrets.compare_digest(supplied, token.encode("utf-8"))
        for token in instance.tokens
    ):
        raise HTTPException(401, detail="Invalid or missing Koji token")
    return await accept_task(payload, request, TaskType.KOJI, request_size)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_analysis_task(
    task_id: UUID, request: Request, response: Response
) -> TaskResponse:
    """Return current state and, when available, the result of an owned task.

    Args:
        task_id: Public UUID of the requested application task.
        request: FastAPI request carrying the authenticated token identity.
        response: Mutable FastAPI response used to set polling and cache headers.

    Returns:
        The task's stable public response envelope.

    Raises:
        HTTPException: With status 404 when the task is absent, expired, or owned by
            another API token.
        RuntimeError: If persisted terminal task data violates response invariants.
    """
    try:
        task = await TaskAnalysis.get_owned(task_id, request.state.api_token_name)
    except AnalysisTaskNotFoundError as exc:
        raise HTTPException(404, detail="Task not found") from exc
    response.headers["Cache-Control"] = "no-store"
    if task.state in ACTIVE_STATES:
        response.headers["Retry-After"] = str(SERVER_CONFIG.task_queue.retry_after)
    return task_representation(task)


@app.delete(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    responses={202: {"model": TaskResponse}},
)
async def cancel_analysis_task(
    task_id: UUID, request: Request, response: Response
) -> TaskResponse:
    """Request cancellation of one owned task and report its resulting state.

    Args:
        task_id: Public UUID of the application task to cancel.
        request: FastAPI request carrying the authenticated token identity.
        response: Mutable FastAPI response used to set status and polling headers.

    Returns:
        The task's stable public response envelope. The HTTP status is 202 while
        process cleanup remains pending and 200 after cancellation is confirmed.

    Raises:
        HTTPException: With status 404 when the task is not visible, or status 409
            when its terminal state cannot be cancelled.
    """
    try:
        task = await TaskAnalysis.request_cancellation(
            task_id,
            request.state.api_token_name,
            task_app.job_manager,
            SERVER_CONFIG.task_queue.retention_days,
        )
    except AnalysisTaskNotFoundError as exc:
        raise HTTPException(404, detail="Task not found") from exc
    except TaskTerminalError as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    response.headers["Cache-Control"] = "no-store"
    if task.state == AnalysisState.CANCELLING:
        response.status_code = 202
        response.headers["Retry-After"] = str(SERVER_CONFIG.task_queue.retry_after)
    return task_representation(task)


@app.get("/version", response_class=BasicResponse)
async def get_version_wrapper():
    """Get the version of logdetective"""
    return BasicResponse(content=get_version())


class MetricRoute(str, Enum):
    """Routes for metrics"""

    ANALYZE = "analyze"
    ANALYZE_GITLAB_JOB = "analyze-gitlab"
    ANALYZE_KOJI_TASK = "analyze-koji"


ROUTE_TO_ENDPOINT_TYPES = {
    MetricRoute.ANALYZE: EndpointType.ANALYZE,
    MetricRoute.ANALYZE_GITLAB_JOB: EndpointType.ANALYZE_GITLAB_JOB,
    MetricRoute.ANALYZE_KOJI_TASK: EndpointType.ANALYZE_KOJI_TASK,
}


@app.get("/metrics/{route}/", response_model=MetricResponse)
async def get_metrics(
    route: MetricRoute,
    start_time: AwareDatetime,
    time_period: TimePeriod,
    end_time: Optional[AwareDatetime] = None,
    api_token_name: str | None = None,
):
    """Get a handler returning statistics for the specified endpoint.

    The `start_time` must be strictly < `end_time`.
    """
    endpoint_type = ROUTE_TO_ENDPOINT_TYPES[route]

    if endpoint_type == EndpointType.ANALYZE_GITLAB_JOB and not SERVER_CONFIG.gitlab.instances:
        raise HTTPException(
            status_code=404,
            detail="No gitlab instance configured, skipping metrics collection."
        )

    if end_time and start_time >= end_time:
        raise HTTPException(
            status_code=400,
            detail=f"start_time: {start_time} >= end_time: {end_time}",
        )
    data = await requests_statistics(
        start_time=start_time,
        end_time=end_time,
        time_period=time_period,
        endpoint=endpoint_type,
        api_token_name=api_token_name
    )

    return MetricResponse(
        metrics=[data]
    )
