"""Preparation shares the suite attempt budget and cancellation ownership."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from beeai_framework.utils.cancellation import AbortController

from logdetective import task_execution
from logdetective.agents import testing_farm_analysis as suite_agent
from logdetective.agents.common import AnalysisAttempt, analysis_attempt, run_agent
from logdetective.config import SERVER_CONFIG
from logdetective.database.models.tasks import TaskAnalysis, TaskType
from logdetective.exceptions import LogDetectiveAgentTimeoutError, LogDetectiveInferenceTimeout
from logdetective.models import TestingFarmSuiteRequest as SuiteRequest
from logdetective.models import TestingFarmAgentOutput as SuiteOutput


def request():
    """A minimal eligible suite whose artifacts are supplied by the worker."""
    return SuiteRequest(suite={"plan": "/plan"}, cases=[{
        "case_id": "c1", "name": "/test", "result": "failed",
    }])


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", [
    "timeout", "cancel", "cancel_then_expiry", "expiry_then_cancel",
])
async def test_preextraction_cleanup_precedes_publication(monkeypatch, trigger):  # pylint: disable=too-many-statements
    """Expiry/cancellation joins extraction, skips the next log, and never infers."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def extract(_content, *, filename):
        calls.append(filename)
        started.set()
        try:
            assert release.wait(timeout=3)
            return [(1, "error")]
        finally:
            finished.set()

    monkeypatch.setattr(suite_agent, "DrainExtractorTool", MagicMock(
        return_value=SimpleNamespace(extractor=extract),
    ))
    infer = AsyncMock()
    monkeypatch.setattr(suite_agent, "run_agent", infer)
    monkeypatch.setattr(
        SERVER_CONFIG.general, "analysis_timeout", 10 if trigger == "cancel" else 0.04,
    )
    task = SimpleNamespace(
        task_type=TaskType.TFT_SUITE, input_payload={}, request_size=0,
        response_metrics_id=None, generation=1,
    )
    monkeypatch.setattr(TaskAnalysis, "mark_started", AsyncMock(return_value=task))

    async def execute(*_args):
        await suite_agent.analyze_testing_farm_suite(
            {"first": "error", "second": "error"}, MagicMock(), request(),
            eligible_case_ids={"c1"},
            deadline=_args[4].when(),
        )
        raise AssertionError("Expired preparation must not return a result")

    monkeypatch.setattr(task_execution, "execute_task", execute)
    error = AsyncMock(return_value=True)
    cancelled = AsyncMock(return_value=True)
    result = AsyncMock(return_value=True)
    monkeypatch.setattr(TaskAnalysis, "publish_error", error)
    monkeypatch.setattr(TaskAnalysis, "confirm_cancelled", cancelled)
    monkeypatch.setattr(TaskAnalysis, "publish_result", result)
    operation = asyncio.create_task(task_execution.run_task(str(uuid4()), 42))
    try:
        async with asyncio.timeout(2):
            while not started.is_set():
                await asyncio.sleep(0.001)
        if trigger in ("cancel", "cancel_then_expiry"):
            operation.cancel()
            await asyncio.sleep(0.01)
            operation.cancel()
        await asyncio.sleep(0.08)
        if trigger == "expiry_then_cancel":
            operation.cancel()
            await asyncio.sleep(0.001)
            operation.cancel()
        assert not operation.done()
        error.assert_not_awaited()
        cancelled.assert_not_awaited()
        result.assert_not_awaited()
    finally:
        release.set()
        with pytest.raises(RuntimeError if trigger == "timeout" else asyncio.CancelledError):
            await operation
    assert finished.is_set()
    assert calls == ["first"]
    infer.assert_not_awaited()
    if trigger == "timeout":
        assert error.await_args.kwargs["code"] == "timeout"
        cancelled.assert_not_awaited()
    else:
        cancelled.assert_awaited_once()
        error.assert_not_awaited()
    result.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("baseline", [{}, {"first:1": "evidence"}])
async def test_retries_reuse_baseline_with_shared_deadline(monkeypatch, baseline):
    """Even empty extraction is cached; inference gets the preparation deadline."""
    attempts = []
    baselines = []

    async def extract(_artifacts, attempt):
        attempts.append(attempt)
        # Model preparation consuming budget without slowing the test down.
        await asyncio.sleep(0.001)
        return baseline

    async def execute(_artifacts, _model, _request, evidence, _ids, attempt):
        baselines.append(evidence)
        if len(baselines) == 1:
            assert attempt is attempts[0]
            raise LogDetectiveInferenceTimeout("transient")
        assert attempt is not attempts[0]
        assert attempt.deadline == attempts[0].deadline
        return SuiteOutput(), {}

    extraction = AsyncMock(side_effect=extract)
    monkeypatch.setattr(suite_agent, "_preextract_suite_evidence", extraction)
    monkeypatch.setattr(suite_agent, "_run_suite_attempt", execute)
    monkeypatch.setattr(SERVER_CONFIG.general, "analysis_timeout", 10)
    monkeypatch.setattr(suite_agent.inference_retry, "wait", lambda _state: 0)
    await suite_agent.analyze_testing_farm_suite(
        {"first": "error"}, MagicMock(), request(), eligible_case_ids={"c1"},
        deadline=asyncio.get_running_loop().time() + 1200,
    )
    extraction.assert_awaited_once()
    assert all(evidence is baseline for evidence in baselines)


@pytest.mark.asyncio
async def test_expired_monotonic_deadline_skips_extraction_and_inference():
    """An overdue timer callback cannot grant fresh time to either phase."""
    attempt = AnalysisAttempt(asyncio.get_running_loop().time() - 1, AbortController())
    assert not attempt.controller.signal.aborted
    with pytest.raises(LogDetectiveAgentTimeoutError):
        await suite_agent._preextract_suite_evidence({"first": "error"}, attempt)
    agent = MagicMock()
    with pytest.raises(LogDetectiveAgentTimeoutError):
        await run_agent(agent, "prompt", attempt=attempt)
    agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_agent_uses_preparation_signal_without_resetting_budget(monkeypatch):
    """The original attempt's timer aborts inference after preparation."""
    monkeypatch.setattr(SERVER_CONFIG.general, "analysis_timeout", 0.08)
    agent = MagicMock()
    with pytest.raises(LogDetectiveAgentTimeoutError):
        async with analysis_attempt(asyncio.get_running_loop().time() + 0.08) as attempt:
            await asyncio.sleep(0.04)

            async def infer(*_middleware):
                await asyncio.sleep(0.05)
                assert attempt.controller.signal.aborted
                return SimpleNamespace(output_structured={"explanation": {"text": "done"}})

            agent.run.return_value.middleware.side_effect = infer
            await run_agent(agent, "prompt", attempt=attempt)
    assert agent.run.call_args.kwargs["signal"] is attempt.controller.signal
