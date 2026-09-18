"""Direct execution of durable Procrastinate analysis tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import aiohttp
from gitlab import Gitlab
from koji import ClientSession
from procrastinate.exceptions import JobAborted

from logdetective.agent.agent import analyze_artifacts
from logdetective.artifacts import get_artifacts_from_payload
from logdetective.config import LOG, SERVER_CONFIG, get_chat_model
from logdetective.database.models import Forge
from logdetective.database.models.tasks import TaskAnalysis, TaskType
from logdetective.exceptions import (
    LogDetectiveAgentTimeoutError,
    LogDetectiveInferenceError,
    LogDetectiveInferenceTimeout,
    LogDetectiveKojiException,
    RemoteLogError,
)
from logdetective.gitlab import process_gitlab_job_event
from logdetective.koji import get_failed_log_from_task
from logdetective.models import (
    APIResponse,
    AnalysisRequest,
    JobHook,
    KojiAnalysisRequest,
    KojiTaskMetadata,
    TaskMetadata,
)
from logdetective.utils import SSRFProtectedResolver, run_blocking, sanitize_artifact


@dataclass(frozen=True)
class TaskOutcome:
    """Result and metadata produced by one application task."""

    response: APIResponse | None
    metadata: TaskMetadata | None


async def _execute_generic(
    payload_data: dict[str, Any], request_size: int
) -> TaskOutcome:
    payload = AnalysisRequest.model_validate(payload_data)
    connector = aiohttp.TCPConnector(
        resolver=(
            SSRFProtectedResolver()
            if SERVER_CONFIG.general.block_localhost_urls
            else None
        )
    )
    timeout = aiohttp.ClientTimeout(
        total=SERVER_CONFIG.general.log_source_request_timeout,
        connect=3.07,
    )
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http:
        artifacts = await get_artifacts_from_payload(payload, http, request_size)
        response = await analyze_artifacts(
            artifacts,
            get_chat_model(SERVER_CONFIG.inference),
            payload.build_metadata,
        )
    return TaskOutcome(response=response, metadata=None)


async def _close_koji_session(session: ClientSession) -> None:
    """Close Koji's requests session without blocking the event loop."""
    if session.rsession is not None:
        await run_blocking(session.rsession.close)


async def _execute_koji(payload_data: dict[str, Any]) -> TaskOutcome:
    payload = KojiAnalysisRequest.model_validate(payload_data)
    config = SERVER_CONFIG.koji.instances[payload.koji_instance]
    session = ClientSession(
        baseurl=config.xmlrpc_url,
        opts={"timeout": SERVER_CONFIG.koji.api_timeout},
    )
    try:
        async with asyncio.timeout(SERVER_CONFIG.koji.retrieval_timeout):
            filename, content = await get_failed_log_from_task(
                session,
                payload.task_id,
                max_size=SERVER_CONFIG.koji.max_artifact_size,
            )
    finally:
        await _close_koji_session(session)

    response = await analyze_artifacts(
        {filename: await run_blocking(sanitize_artifact, content)},
        get_chat_model(SERVER_CONFIG.inference),
    )
    return TaskOutcome(
        response=response,
        metadata=KojiTaskMetadata(
            log_file_name=filename,
            task_id=payload.task_id,
        ),
    )


async def _execute_gitlab(
    payload_data: dict[str, Any], metrics_id: int | None
) -> TaskOutcome:
    forge = Forge(payload_data["forge"])
    job_hook = JobHook.model_validate(payload_data["job_hook"])
    config = SERVER_CONFIG.gitlab.instances[forge.value]
    token = config.api_token
    connection = Gitlab(url=config.url, private_token=token, timeout=config.timeout)
    try:
        async with aiohttp.ClientSession(
            base_url=config.url,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=aiohttp.ClientTimeout(total=config.timeout, connect=3.07),
        ) as http:
            await process_gitlab_job_event(
                config,
                connection,
                http,
                forge,
                job_hook,
                get_chat_model(SERVER_CONFIG.inference),
                payload_data.get("api_token_name"),
                metrics_id,
            )
    finally:
        await run_blocking(connection.session.close)
    return TaskOutcome(response=None, metadata=None)


async def execute_task(
    task_type: TaskType,
    payload_data: dict[str, Any],
    request_size: int,
    metrics_id: int | None,
) -> TaskOutcome:
    """Execute one validated durable payload in the Procrastinate worker."""
    if task_type == TaskType.GENERIC:
        return await _execute_generic(payload_data, request_size)
    if task_type == TaskType.KOJI:
        return await _execute_koji(payload_data)
    if task_type == TaskType.GITLAB:
        return await _execute_gitlab(payload_data, metrics_id)
    raise ValueError(f"Unknown task type {task_type!r}")


def public_error(exc: Exception) -> tuple[str, str]:
    """Map an internal failure to stable information safe for API clients."""
    if isinstance(
        exc, (TimeoutError, LogDetectiveAgentTimeoutError, LogDetectiveInferenceTimeout)
    ):
        return "timeout", "Analysis exceeded its time limit"
    if isinstance(exc, RemoteLogError) or isinstance(exc.__cause__, RemoteLogError):
        return "artifact_error", "Unable to retrieve an artifact"
    if isinstance(exc, LogDetectiveKojiException):
        return "koji_error", "Unable to retrieve the Koji build"
    if isinstance(exc, LogDetectiveInferenceError):
        return "inference_error", "Inference failed"
    return "analysis_error", "Analysis could not be completed"


async def _confirm_cancellation(task_id: UUID, job_id: int) -> None:
    """Durably acknowledge cancellation despite repeated cancellation signals."""
    confirmation = asyncio.create_task(
        TaskAnalysis.confirm_cancelled(
            task_id, job_id, SERVER_CONFIG.task_queue.retention_days
        )
    )
    while not confirmation.done():
        try:
            await asyncio.shield(confirmation)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            continue
    await confirmation


async def run_task(task_id: str, job_id: int) -> None:
    """Execute and conditionally publish one durably admitted application task."""
    public_id = UUID(task_id)
    task = await TaskAnalysis.mark_started(public_id, job_id)
    if task is None:
        raise JobAborted("Application task is no longer eligible to run")

    try:
        try:
            if task.input_payload is None:
                raise RuntimeError("Durable task input is unavailable")
            outcome = await execute_task(
                task.task_type,
                task.input_payload,
                task.request_size,
                task.response_metrics_id,
            )
            if task.task_type != TaskType.GITLAB and outcome.response is None:
                raise RuntimeError("Analysis task returned no response")
        except Exception as exc:
            LOG.exception("Analysis task %s failed", public_id)
            code, message = public_error(exc)
            published = await TaskAnalysis.publish_error(
                task_id=public_id,
                job_id=job_id,
                generation=task.generation,
                code=code,
                message=message,
                retention_days=SERVER_CONFIG.task_queue.retention_days,
            )
            if not published:
                raise JobAborted(
                    "Application publication fence rejected the error"
                ) from exc
            raise RuntimeError(f"Analysis task {public_id} failed with {code}") from exc

        if outcome.response is None:
            published = await TaskAnalysis.complete_without_result(
                task_id=public_id,
                job_id=job_id,
                generation=task.generation,
                retention_days=SERVER_CONFIG.task_queue.retention_days,
            )
        else:
            published = await TaskAnalysis.publish_result(
                task_id=public_id,
                job_id=job_id,
                generation=task.generation,
                response=outcome.response,
                task_metadata=outcome.metadata,
                retention_days=SERVER_CONFIG.task_queue.retention_days,
            )
        if not published:
            LOG.info("Discarded stale result for task %s", public_id)
            raise JobAborted("Application publication fence rejected the result")
    except asyncio.CancelledError:
        await _confirm_cancellation(public_id, job_id)
        raise
