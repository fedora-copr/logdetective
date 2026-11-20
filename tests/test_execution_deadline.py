"""Shared worker deadlines, publication boundaries, and cancellation races."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from logdetective import task_execution
from logdetective.config import SERVER_CONFIG
from logdetective.database.models.tasks import TaskAnalysis, TaskType
from logdetective.models import APIResponse, Config
from logdetective.task_execution import TaskOutcome
from logdetective.utils import run_blocking


def test_default_execution_budget():
    """All workflows have the same positive finite default."""
    assert Config().general.analysis_timeout == 1200


def test_configured_analysis_timeout():
    """The established configuration key retains its integer-seconds contract."""
    config = Config(general={"analysis_timeout": 900})
    assert config.general.analysis_timeout == 900
    assert isinstance(config.general.analysis_timeout, int)
    assert config.model_json_schema()["$defs"]["GeneralConfig"]["properties"][
        "analysis_timeout"
    ]["type"] == "integer"


@pytest.mark.asyncio
@pytest.mark.parametrize("task_type", list(TaskType))
@pytest.mark.parametrize("expired", [False, True])
async def test_publication_and_cleanup_boundaries(monkeypatch, task_type, expired):
    """Queue time is free; expiry blocks publication; cleanup/commit are untimed."""
    events = []
    task = SimpleNamespace(task_type=task_type, input_payload={}, request_size=0,
                           response_metrics_id=None, generation=1)
    monkeypatch.setattr(SERVER_CONFIG.general, "analysis_timeout", 0.02)

    async def started(*_args):
        await asyncio.sleep(0.03)
        return task

    async def cleanup():
        await asyncio.sleep(0.03)
        events.append("cleanup")

    async def execute(*args):
        deadline, stack = args[4], args[5]
        assert deadline > asyncio.get_running_loop().time()
        stack.push_async_callback(cleanup)
        if expired:
            # Deliberately deny the event loop a timer callback.
            time.sleep(0.03)
        return TaskOutcome(
            response=None if task_type == TaskType.GITLAB else APIResponse(explanation="done"),
            metadata=None, publication=object() if task_type == TaskType.GITLAB else None,
        )

    async def comment(_publication):
        await asyncio.sleep(0.03)
        events.append("comment")

    async def terminal(**_kwargs):
        assert events[-1] == "cleanup"
        await asyncio.sleep(0.03)
        events.append("terminal")
        return True

    monkeypatch.setattr(TaskAnalysis, "mark_started", started)
    monkeypatch.setattr(task_execution, "execute_task", execute)
    publish = AsyncMock(side_effect=comment)
    monkeypatch.setattr(task_execution, "publish_gitlab_analysis", publish)
    error = AsyncMock(side_effect=terminal)
    monkeypatch.setattr(TaskAnalysis, "publish_error", error)
    monkeypatch.setattr(TaskAnalysis, "publish_result", terminal)
    monkeypatch.setattr(TaskAnalysis, "complete_without_result", terminal)
    if expired:
        with pytest.raises(RuntimeError, match="timeout"):
            await task_execution.run_task(str(uuid4()), 42)
        publish.assert_not_awaited()
        assert error.await_args.kwargs["code"] == "timeout"
    else:
        await task_execution.run_task(str(uuid4()), 42)
        error.assert_not_awaited()
        assert publish.await_count == (task_type == TaskType.GITLAB)
    assert events[-2:] == ["cleanup", "terminal"]


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", [
    "timeout", "cancel", "cancel_then_expiry", "expiry_then_cancel",
])
@pytest.mark.parametrize("fails", [False, True])
async def test_direct_blocking_cancellation_counts(trigger, fails):
    """Joining sanitization preserves explicit cancellation even if work raises."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        try:
            assert release.wait(3)
            if fails:
                raise ValueError("worker failed")
        finally:
            finished.set()

    async def execute():
        async with asyncio.timeout(10 if trigger == "cancel" else 0.05):
            await run_blocking(work)

    operation = asyncio.create_task(execute())
    try:
        async with asyncio.timeout(2):
            while not started.is_set():
                await asyncio.sleep(0.001)
        if trigger in ("cancel", "cancel_then_expiry"):
            operation.cancel()
            await asyncio.sleep(0.001)
            operation.cancel()
        await asyncio.sleep(0.08)
        if trigger == "expiry_then_cancel":
            operation.cancel()
            await asyncio.sleep(0.001)
            operation.cancel()
        assert not operation.done()
    finally:
        release.set()
        with pytest.raises(TimeoutError if trigger == "timeout" else asyncio.CancelledError):
            await operation
    assert finished.is_set()
    assert operation.cancelling() == (0 if trigger == "timeout" else 2)


@pytest.mark.asyncio
async def test_build_retries_share_preparation_deadline(monkeypatch):
    """Build setup consumes the same deadline as fresh inference attempts."""
    from unittest.mock import MagicMock  # pylint: disable=import-outside-toplevel
    from tenacity import wait_none  # pylint: disable=import-outside-toplevel
    from logdetective.agents import base_analysis  # pylint: disable=import-outside-toplevel
    from logdetective.exceptions import LogDetectiveInferenceTimeout  # pylint: disable=import-outside-toplevel

    deadline = asyncio.get_running_loop().time() + 1
    attempts = []
    setup = AsyncMock(side_effect=lambda _artifacts: SimpleNamespace(
        tools=[], requirements=[], extractors=[],
    ))
    monkeypatch.setattr(base_analysis, "build_tool_setup", setup)
    monkeypatch.setattr(base_analysis, "RequirementAgent", MagicMock())

    async def infer(_agent, _prompt, *, attempt):
        attempts.append(attempt)
        assert attempt.deadline == deadline
        if len(attempts) == 1:
            await asyncio.sleep(0.01)
            raise LogDetectiveInferenceTimeout("transient")
        return object()

    monkeypatch.setattr(base_analysis, "run_agent", infer)
    monkeypatch.setattr(
        base_analysis, "to_api_response", lambda *_args: APIResponse(explanation="done"),
    )
    analyze = base_analysis.analyze_artifacts.retry_with(wait=wait_none())
    await analyze({"log": "error"}, MagicMock(), deadline=deadline)
    assert setup.await_count == 2
    assert attempts[0] is not attempts[1]
    assert attempts[0].controller is not attempts[1].controller
