"""Runtime TF admission limits and one execution budget across all phases."""

import asyncio
from datetime import UTC, datetime
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import update
from tenacity import wait_none

from logdetective import artifacts, task_execution
from logdetective.agents import common, testing_farm_analysis as workflow
from logdetective.config import SERVER_CONFIG
from logdetective.database.base import transaction
from logdetective.database.models.tasks import TaskAnalysis, TaskType
from logdetective.exceptions import LogDetectiveAgentTimeoutError, LogDetectiveInferenceTimeout
from logdetective.models import (
    Config, TestingFarmConfig as TFTConfig, TestingFarmSuiteRequest as TFTRequest,
    TestingFarmAgentOutput as TFTOutput,
)
from logdetective.remote_log import RemoteLog
from logdetective.server import app
from logdetective.utils import run_blocking
from tests.test_helpers import DatabaseFactory


async def execute_suite(payload, size, *, deadline):
    """Mirror the worker's timeout boundary for focused execution tests."""
    deadline = min(
        deadline, asyncio.get_running_loop().time() + SERVER_CONFIG.general.analysis_timeout,
    )
    async with asyncio.timeout_at(deadline):
        return await task_execution._execute_tft_suite(payload, size, deadline)


def suite(case_logs=1, shared_artifacts=0, cases=1):
    """Distinct scoped entries deliberately reuse one URL."""
    def logs(count):
        return [{"name": f"log-{i}", "url": "https://example.org/log"} for i in range(count)]

    return TFTRequest(
        suite={"plan": "/plan", "state": "error"}, shared_artifacts=logs(shared_artifacts),
        cases=[{"case_id": f"c{i}", "name": "/test", "result": "failed",
                "artifacts": logs(case_logs)} for i in range(cases)],
    )


@pytest.mark.parametrize("limit", [1, 150])
@pytest.mark.parametrize("scope", [
    "max_artifacts_per_case", "max_shared_artifacts", "max_total_artifacts",
])
def test_configured_boundaries(scope, limit):
    """Lower and higher caps apply to entries, even when URLs repeat."""
    config = TFTConfig(
        max_artifacts_per_case=200, max_shared_artifacts=200, max_total_artifacts=400,
    )
    setattr(config, scope, limit)

    def request(count):
        if scope == "max_shared_artifacts":
            return suite(0, count)
        if scope == "max_total_artifacts":
            return suite(1, count - 1)
        return suite(count)

    request(limit).validate_limits(config)
    with pytest.raises(ValueError, match=scope):
        request(limit + 1).validate_limits(config)


def test_total_can_be_lower_and_no_case_count_cap():
    """Scope allowances do not bypass the total; empty cases are not truncated."""
    with pytest.raises(ValueError, match="max_total_artifacts"):
        suite(2, 2, 2).validate_limits(TFTConfig(max_total_artifacts=5))
    request = suite(0, 1, 100)
    request.validate_limits(TFTConfig())
    assert len(request.cases) == 100
    assert Config().testing_farm == TFTConfig()
    assert TFTConfig().model_dump() == {
        "max_artifacts_per_case": 10, "max_shared_artifacts": 10,
        "max_total_artifacts": 120,
    }


@pytest.mark.parametrize("field", [
    "max_artifacts_per_case", "max_shared_artifacts", "max_total_artifacts",
])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_log_limits(field, value):
    """Log allowances must be positive integers."""
    with pytest.raises(ValidationError):
        TFTConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, 1.5, float("inf"), float("-inf"), float("nan")])
def test_invalid_analysis_timeout(value):
    """The shared clock requires a positive integer duration in seconds."""
    with pytest.raises(ValidationError):
        Config(general={"analysis_timeout": value})


@pytest.mark.asyncio
async def test_excess_artifact_count_rejected_before_enqueue(monkeypatch):
    """A new suite exceeding the configured count limit is not enqueued."""
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_total_artifacts", 1)
    monkeypatch.setattr(TaskAnalysis, "_existing", AsyncMock(return_value=None))
    enqueue = MagicMock()
    monkeypatch.setattr("logdetective.server.analyze_tft_suite", enqueue)
    payload = suite(1, 1).model_dump(mode="json")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/analyze/tft", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "testing_farm.max_total_artifacts exceeded"}
    enqueue.configure.assert_not_called()


@pytest.mark.asyncio
async def test_lowered_cap_preserves_uuid_retry_and_conflict(monkeypatch):
    """Existing UUIDs are resolved before a changed cap, without a second job."""
    public_id = uuid4()
    payload = suite(1, 1).model_dump(mode="json")
    payload["id"] = str(public_id)
    enqueue = AsyncMock(return_value=41)
    job = MagicMock()
    job.configure.return_value.defer_async = enqueue
    monkeypatch.setattr("logdetective.server.analyze_tft_suite", job)
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_total_artifacts", 2)
    async with DatabaseFactory().make_new_db(), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post("/analyze/tft", json=payload)
        assert first.status_code == 202
        monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_total_artifacts", 1)
        retry = await client.post("/analyze/tft", json=payload)
        assert retry.status_code == 202
        assert retry.json() == first.json()
        conflict = await client.post("/analyze/tft", json={
            **payload, "suite": {"plan": "/different", "state": "error"},
        })
        assert conflict.status_code == 409
        async with transaction(commit=True) as session:
            await session.execute(update(TaskAnalysis).where(
                TaskAnalysis.task_id == public_id
            ).values(expires_at=datetime.now(UTC)))
        expired = await client.post("/analyze/tft", json=payload)
        assert expired.status_code == 409
        enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieval_honors_admission_count_and_keeps_url_cache(monkeypatch):
    """Lowering the count cap after validation does not prevent retrieval."""
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_artifacts_per_case", 12)
    payload = suite(12)
    payload.validate_limits(SERVER_CONFIG.testing_farm)
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_artifacts_per_case", 10)
    download = AsyncMock(return_value="failure")
    monkeypatch.setattr(RemoteLog, "get_url_content", download)
    fetched = await artifacts.fetch_suite_artifacts(payload, 100)
    assert len(fetched.artifacts) == 12
    download.assert_awaited_once()


@pytest.mark.parametrize("phase", ["retrieval", "extraction", "inference", "backoff", "assembly"])
@pytest.mark.asyncio
async def test_suite_timeout_is_not_partial_result(monkeypatch, phase):
    """Every phase consumes the same enclosing deadline, including synchronous assembly."""
    monkeypatch.setattr(SERVER_CONFIG.general, "analysis_timeout", 0.03)
    monkeypatch.setattr(task_execution, "get_chat_model", lambda _: MagicMock())
    monkeypatch.setattr(RemoteLog, "get_url_content", AsyncMock(return_value="failure"))
    monkeypatch.setattr(workflow, "_preextract_suite_evidence", AsyncMock(return_value={}))
    run = AsyncMock(return_value=(TFTOutput(), {}))
    monkeypatch.setattr(workflow, "_run_suite_attempt", run)

    async def stall(*_args, **_kwargs):
        await asyncio.sleep(10)

    if phase == "retrieval":
        monkeypatch.setattr(RemoteLog, "get_url_content", stall)
    elif phase == "extraction":
        monkeypatch.setattr(workflow, "_preextract_suite_evidence", stall)
    elif phase == "inference":
        monkeypatch.setattr(workflow, "_run_suite_attempt", stall)
    elif phase == "backoff":
        run.side_effect = LogDetectiveInferenceTimeout("retry")
    else:
        original = task_execution._suite_result

        def assemble(*args):
            time.sleep(0.05)
            return original(*args)
        monkeypatch.setattr(task_execution, "_suite_result", assemble)
    with pytest.raises((TimeoutError, LogDetectiveAgentTimeoutError)) as error:
        await execute_suite(
            suite().model_dump(mode="json"), 100,
            deadline=asyncio.get_running_loop().time() + 1200,
        )
    assert task_execution.public_error(error.value)[0] == "timeout"
    if phase == "backoff":
        assert run.await_count == 1


@pytest.mark.asyncio
async def test_attempts_use_remaining_suite_budget(monkeypatch):
    """Retry attempts keep the original suite deadline; generic budgets stay independent."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1
    for _ in range(2):
        async with common.analysis_attempt(deadline) as attempt:
            assert attempt.deadline == deadline
        await asyncio.sleep(0.001)
    with pytest.raises(LogDetectiveAgentTimeoutError):
        async with common.analysis_attempt(loop.time() - 1):
            pytest.fail("Expired suite must not begin another attempt")


@pytest.mark.parametrize("trigger", [
    "timeout", "cancel", "cancel_then_expiry", "expiry_then_cancel",
])
@pytest.mark.asyncio
# pylint: disable=too-many-statements
async def test_suite_cleanup_and_cancellation_races(monkeypatch, trigger):
    """Nested BeeAI threads finish before publication; explicit cancellation wins races."""

    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        try:
            assert release.wait(timeout=3)
        finally:
            finished.set()

    class Agent:  # pylint: disable=too-few-public-methods
        """Use actual nested BeeAI cancellation and thread ownership."""
        emitter = Emitter.root().child(namespace=["suite_budget_test"])

        def run(self, _prompt, *, signal, **_kwargs):
            async def tool(_context):
                await run_blocking(work)
                return SimpleNamespace(output_structured={"cases": []})

            async def agent(_context):
                return await RunContext.enter(self, tool)

            return RunContext.enter(self, agent, signal=signal)

    monkeypatch.setattr(SERVER_CONFIG.general, "analysis_timeout",
                        10 if trigger == "cancel" else 0.1)
    monkeypatch.setattr(task_execution, "get_chat_model", lambda _: MagicMock())
    monkeypatch.setattr(RemoteLog, "get_url_content", AsyncMock(return_value="failure"))
    monkeypatch.setattr(workflow, "_preextract_suite_evidence", AsyncMock(return_value={}))

    async def run(*args):
        output = await common.run_agent(Agent(), "analyze", TFTOutput, attempt=args[-1])
        return output, {}

    monkeypatch.setattr(workflow, "_run_suite_attempt", run)
    task = SimpleNamespace(
        task_type=TaskType.TFT_SUITE, input_payload=suite().model_dump(mode="json"),
        request_size=100, response_metrics_id=None, generation=1,
    )
    monkeypatch.setattr(TaskAnalysis, "mark_started", AsyncMock(return_value=task))
    error, result, cancelled = (AsyncMock(return_value=True) for _ in range(3))
    monkeypatch.setattr(TaskAnalysis, "publish_error", error)
    monkeypatch.setattr(TaskAnalysis, "publish_result", result)
    monkeypatch.setattr(TaskAnalysis, "confirm_cancelled", cancelled)
    operation = asyncio.create_task(task_execution.run_task(str(uuid4()), 42))
    try:
        async with asyncio.timeout(2):
            while not started.is_set():
                await asyncio.sleep(0.001)
        if trigger in ("cancel", "cancel_then_expiry"):
            operation.cancel()
            await asyncio.sleep(0.01)
            operation.cancel()
        await asyncio.sleep(0.15)
        if trigger == "expiry_then_cancel":
            operation.cancel()
            await asyncio.sleep(0.01)
            operation.cancel()
        assert not operation.done()
        error.assert_not_awaited()
        result.assert_not_awaited()
        cancelled.assert_not_awaited()
    finally:
        release.set()
        with pytest.raises(RuntimeError if trigger == "timeout" else asyncio.CancelledError):
            await operation
    assert finished.is_set()
    result.assert_not_awaited()
    if trigger == "timeout":
        error.assert_awaited_once()
        assert error.await_args.kwargs["code"] == "timeout"
        cancelled.assert_not_awaited()
    else:
        cancelled.assert_awaited_once()
        error.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_suite_retry_preserves_deadline(monkeypatch):
    """A transient failure rebuilds the attempt but consumes the remaining suite time."""

    monkeypatch.setattr(SERVER_CONFIG.general, "analysis_timeout", 1)
    monkeypatch.setattr(task_execution, "get_chat_model", lambda _: MagicMock())
    monkeypatch.setattr(RemoteLog, "get_url_content", AsyncMock(return_value="failure"))
    extract = AsyncMock(return_value={})
    monkeypatch.setattr(workflow, "_preextract_suite_evidence", extract)
    monkeypatch.setattr(
        workflow, "inference_retry", workflow.inference_retry.copy(wait=wait_none()),
    )
    deadlines, remaining = [], []

    async def run(*args):
        attempt = args[-1]
        deadlines.append(attempt.deadline)
        remaining.append(attempt.deadline - asyncio.get_running_loop().time())
        if len(deadlines) == 1:
            await asyncio.sleep(0.01)
            raise LogDetectiveInferenceTimeout("transient")
        return TFTOutput(), {}

    monkeypatch.setattr(workflow, "_run_suite_attempt", run)
    await execute_suite(
        suite().model_dump(mode="json"), 100,
        deadline=asyncio.get_running_loop().time() + 1200,
    )
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert remaining[1] < remaining[0]
    extract.assert_awaited_once()
