"""Direct execution of durable Procrastinate analysis tasks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Collection
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import aiohttp
from gitlab import Gitlab
from koji import ClientSession
from procrastinate.exceptions import JobAborted

from logdetective.agents.base_analysis import analyze_artifacts
from logdetective.agents.testing_farm_analysis import (
    analyze_testing_farm_suite,
)
from logdetective.artifacts import (
    create_artifact_session, fetch_suite_artifacts, get_artifacts_from_payload,
)
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
    TestingFarmSuiteRequest,
    TestingFarmSuiteResponse,
    TestingFarmCaseOutcome,
    TestingFarmAgentDiagnosis,
    TestingFarmAgentOutput,
    TestingFarmCase,
    TaskError,
    TestingFarmEvidenceIndex,
    TestingFarmCaseResponse,
    TaskMetadata,
)
from logdetective.utils import (
    check_deadline,
    run_blocking,
    sanitize_artifact,
)


@dataclass(frozen=True)
class TaskOutcome:
    """Result and metadata produced by one application task."""

    response: APIResponse | TestingFarmSuiteResponse | None
    metadata: TaskMetadata | None


async def _execute_generic(
    payload_data: dict[str, Any], request_size: int, deadline: float, stack: AsyncExitStack,
) -> TaskOutcome:
    payload = AnalysisRequest.model_validate(payload_data)
    http = await stack.enter_async_context(create_artifact_session())
    artifacts = await get_artifacts_from_payload(payload, http, request_size)
    check_deadline(deadline)
    response = await analyze_artifacts(
        artifacts, get_chat_model(SERVER_CONFIG.inference), payload.build_metadata,
        deadline=deadline,
    )
    return TaskOutcome(response=response, metadata=None)


def _suite_case_response(
    output: TestingFarmAgentDiagnosis, evidence: TestingFarmEvidenceIndex, allowed: set[str],
    shared_refs: list[str],
) -> TestingFarmCaseResponse:
    """Attach own evidence and reference shared evidence without parsing citations."""
    # The collector owns ordering, variant deduplication, and attempt isolation.
    snippets = [
        snippet for variants in evidence.values() for snippet in variants
        if snippet.source_file in allowed
    ]
    return TestingFarmCaseResponse(
        explanation=output.explanation,
        solution=output.solution if SERVER_CONFIG.general.generate_solution else None,
        no_issue_found=output.no_issue_found,
        snippets=snippets,
        shared_snippet_refs=shared_refs,
    )


def _shared_snippets_by_file(
    evidence: TestingFarmEvidenceIndex, shared_artifacts: Collection[str],
) -> TestingFarmEvidenceIndex:
    """Group shared evidence by artifact, preserving each snippet and its line number."""
    by_file: TestingFarmEvidenceIndex = {}
    for variants in evidence.values():
        for snippet in variants:
            source = snippet.source_file
            if source is not None and source in shared_artifacts:
                by_file.setdefault(source, []).append(snippet)
    return by_file


def _suite_result(
    payload: TestingFarmSuiteRequest,
    eligible: list[TestingFarmCase],
    errors: dict[str, TestingFarmCaseOutcome],
    output: TestingFarmAgentOutput | None,
    evidence: TestingFarmEvidenceIndex,
) -> TestingFarmSuiteResponse:
    """Index validated agent answers and assemble case-scoped public results."""
    results = dict(errors)
    if output is None:
        return TestingFarmSuiteResponse(cases=results)

    # The suite agent validates uniqueness and eligibility before returning.
    answers = {answer.case_id: answer for answer in output.cases}

    fallback = output.suite_fallback

    shared_artifacts = payload.shared_artifacts_by_ref()
    shared_snippets = _shared_snippets_by_file(evidence, shared_artifacts)
    shared_refs = list(shared_snippets)
    for case in eligible:
        answer = answers.get(case.case_id)
        if answer is None or not (answer.explanation or "").strip():
            results[case.case_id] = TestingFarmCaseOutcome(
                status="error", error=TaskError(
                    code="missing_analysis", message="No analysis was returned for this case"
                ),
            )
            continue
        allowed = {f"case/{case.case_id}/{log.name}" for log in case.artifacts}
        results[case.case_id] = TestingFarmCaseOutcome(
            status="done", analysis=_suite_case_response(answer, evidence, allowed, shared_refs)
        )

    return TestingFarmSuiteResponse(
        cases=results, shared_snippets=shared_snippets,
        suite_fallback=(
            _suite_case_response(fallback, evidence, set(), shared_refs)
            if fallback is not None else None
        ),
    )


async def _execute_tft_suite(
    payload_data: dict[str, Any], request_size: int, deadline: float,
) -> TaskOutcome:
    """Fetch and analyze a suite under the caller's execution deadline."""
    started = time.monotonic()
    payload = TestingFarmSuiteRequest.model_validate(payload_data)
    check_deadline(deadline)
    fetched = await fetch_suite_artifacts(payload, request_size)
    check_deadline(deadline)
    if fetched.eligible or not payload.cases:
        LOG.info(
            "Testing Farm suite agent run: cases=%d artifacts=%d",
            len(fetched.eligible), len(fetched.artifacts),
        )
        output, evidence = await analyze_testing_farm_suite(
            fetched.artifacts, get_chat_model(SERVER_CONFIG.inference), payload,
            eligible_case_ids={case.case_id for case in fetched.eligible},
            deadline=deadline,
        )
    else:
        output = None
        evidence = {}
    check_deadline(deadline)
    result = _suite_result(payload, fetched.eligible, fetched.errors, output, evidence)
    result.retrieval_warnings = fetched.warnings
    LOG.info(
        "Testing Farm suite completed: cases=%d artifacts=%d errors=%d latency_seconds=%.3f",
        len(payload.cases),
        len(payload.shared_artifacts) + sum(len(c.artifacts) for c in payload.cases),
        sum(case.status == "error" for case in result.cases.values()),
        time.monotonic() - started,
    )
    check_deadline(deadline)
    return TaskOutcome(response=result, metadata=None)


async def _close_koji_session(session: ClientSession) -> None:
    """Close Koji's requests session without blocking the event loop."""
    if session.rsession is not None:
        await run_blocking(session.rsession.close)


async def _execute_koji(
    payload_data: dict[str, Any], deadline: float, stack: AsyncExitStack,
) -> TaskOutcome:
    payload = KojiAnalysisRequest.model_validate(payload_data)
    config = SERVER_CONFIG.koji.instances[payload.koji_instance]
    session = ClientSession(
        baseurl=config.xmlrpc_url,
        opts={"timeout": SERVER_CONFIG.koji.api_timeout},
    )
    stack.push_async_callback(_close_koji_session, session)
    async with asyncio.timeout(SERVER_CONFIG.koji.retrieval_timeout):
        filename, content = await get_failed_log_from_task(
            session, payload.task_id, max_size=SERVER_CONFIG.koji.max_artifact_size,
        )
    check_deadline(deadline)
    sanitized = await run_blocking(sanitize_artifact, content)
    check_deadline(deadline)
    response = await analyze_artifacts(
        {filename: sanitized},
        get_chat_model(SERVER_CONFIG.inference),
        deadline=deadline,
    )
    return TaskOutcome(
        response=response,
        metadata=KojiTaskMetadata(
            log_file_name=filename,
            task_id=payload.task_id,
        ),
    )


async def _execute_gitlab(
    payload_data: dict[str, Any], metrics_id: int | None,
    timeout: asyncio.Timeout, stack: AsyncExitStack,
) -> TaskOutcome:
    forge = Forge(payload_data["forge"])
    job_hook = JobHook.model_validate(payload_data["job_hook"])
    config = SERVER_CONFIG.gitlab.instances[forge.value]
    token = config.api_token
    connection = Gitlab(url=config.url, private_token=token, timeout=config.timeout)
    stack.push_async_callback(run_blocking, connection.session.close)
    http = await stack.enter_async_context(aiohttp.ClientSession(
        base_url=config.url,
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=aiohttp.ClientTimeout(total=config.timeout, connect=3.07),
    ))
    await process_gitlab_job_event(
        config, connection, http, forge, job_hook,
        get_chat_model(SERVER_CONFIG.inference),
        payload_data.get("api_token_name"), metrics_id, timeout=timeout,
    )
    return TaskOutcome(response=None, metadata=None)


async def execute_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    task_type: TaskType,
    payload_data: dict[str, Any],
    request_size: int,
    metrics_id: int | None,
    timeout: asyncio.Timeout,
    stack: AsyncExitStack,
) -> TaskOutcome:
    """Execute one validated durable payload in the Procrastinate worker."""
    if task_type == TaskType.GITLAB:
        return await _execute_gitlab(payload_data, metrics_id, timeout, stack)
    deadline = timeout.when()
    assert deadline is not None
    if task_type == TaskType.GENERIC:
        return await _execute_generic(payload_data, request_size, deadline, stack)
    if task_type == TaskType.TFT_SUITE:
        return await _execute_tft_suite(payload_data, request_size, deadline)
    if task_type == TaskType.KOJI:
        return await _execute_koji(payload_data, deadline, stack)
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


async def run_task(task_id: str, job_id: int) -> None:  # pylint: disable=too-many-branches,too-many-statements
    """Execute and conditionally publish one durably admitted application task."""
    public_id = UUID(task_id)
    task = await TaskAnalysis.mark_started(public_id, job_id)
    if task is None:
        raise JobAborted("Application task is no longer eligible to run")
    started_at = time.monotonic()

    try:
        try:
            if task.input_payload is None:
                raise RuntimeError("Durable task input is unavailable")
            async with AsyncExitStack() as stack:
                deadline = (
                    asyncio.get_running_loop().time() + SERVER_CONFIG.general.analysis_timeout
                )
                async with asyncio.timeout_at(deadline) as timeout:
                    outcome = await execute_task(
                        task.task_type, task.input_payload, task.request_size,
                        task.response_metrics_id, timeout, stack,
                    )
                    if timeout.when() is not None:
                        check_deadline(deadline)
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
    finally:
        if task.task_type == TaskType.TFT_SUITE:
            LOG.info(
                "Testing Farm suite task latency_seconds=%.3f",
                time.monotonic() - started_at,
            )
