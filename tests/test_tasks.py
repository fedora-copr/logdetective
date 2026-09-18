"""Tests for periodic reconciliation of application and queue state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from procrastinate.exceptions import NoResult
from procrastinate.jobs import Status

from logdetective.config import SERVER_CONFIG
from logdetective.database.models.tasks import AnalysisState, TaskAnalysis
from logdetective.procrastinate_app import app
from logdetective.tasks import expire_analysis_jobs, reconcile_analysis_jobs


def _manager(**methods) -> SimpleNamespace:
    """Build a job-manager fake with asynchronous methods."""
    defaults = {
        "get_stalled_jobs": AsyncMock(return_value=[]),
        "finish_job": AsyncMock(),
        "cancel_job_by_id_async": AsyncMock(),
        "get_job_status_async": AsyncMock(return_value=Status.DOING),
        "delete_old_jobs": AsyncMock(),
    }
    defaults.update(methods)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_reconcile_fails_stalled_jobs(mocker, monkeypatch):
    """Stalled queue jobs lose publication rights and finish as failed."""
    log = mocker.patch("logdetective.tasks.LOG")
    missing_id = SimpleNamespace(id=None)
    stalled_job = SimpleNamespace(id=37)
    manager = _manager(
        get_stalled_jobs=AsyncMock(return_value=[missing_id, stalled_job])
    )
    monkeypatch.setattr(app, "job_manager", manager)
    mark_queue_failure = mocker.patch.object(
        TaskAnalysis, "mark_queue_failure", new_callable=AsyncMock
    )
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[]
    )

    await reconcile_analysis_jobs(123)

    manager.get_stalled_jobs.assert_awaited_once_with(
        queue="analysis",
        seconds_since_heartbeat=SERVER_CONFIG.task_queue.stalled_worker_timeout,
    )
    mark_queue_failure.assert_awaited_once_with(
        37, SERVER_CONFIG.task_queue.retention_days
    )
    manager.finish_job.assert_awaited_once_with(
        stalled_job, Status.FAILED, delete_job=False
    )
    log.info.assert_called_once_with(
        "Marked stalled analysis job %s as failed", 37
    )


@pytest.mark.parametrize(
    "status", [Status.ABORTED, Status.CANCELLED, Status.SUCCEEDED]
)
@pytest.mark.asyncio
async def test_reconcile_confirms_queue_cancellation(
    status, mocker, monkeypatch
):
    """A cancelling record becomes cancelled after queue confirmation."""
    log = mocker.patch("logdetective.tasks.LOG")
    task_id = uuid4()
    task = SimpleNamespace(
        task_id=task_id,
        procrastinate_job_id=41,
        state=AnalysisState.CANCELLING,
    )
    manager = _manager(get_job_status_async=AsyncMock(return_value=status))
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    confirm_cancelled = mocker.patch.object(
        TaskAnalysis, "confirm_cancelled", new_callable=AsyncMock
    )
    mark_queue_failure = mocker.patch.object(
        TaskAnalysis, "mark_queue_failure", new_callable=AsyncMock
    )

    await reconcile_analysis_jobs(123)

    manager.cancel_job_by_id_async.assert_awaited_once_with(41, abort=True)
    confirm_cancelled.assert_awaited_once_with(
        task_id, 41, SERVER_CONFIG.task_queue.retention_days
    )
    log.info.assert_called_once_with(
        "Confirmed cancellation of analysis task %s for job %s", task_id, 41
    )
    mark_queue_failure.assert_not_awaited()


@pytest.mark.parametrize(
    "status",
    [Status.FAILED, Status.ABORTED, Status.CANCELLED],
)
@pytest.mark.parametrize("transitioned", [True, False])
@pytest.mark.asyncio
async def test_reconcile_handles_unexpected_terminal_jobs(
    status, transitioned, mocker, monkeypatch
):
    """Only a still-active task emits an inconsistency log."""
    log = mocker.patch("logdetective.tasks.LOG")
    task = SimpleNamespace(
        task_id=uuid4(),
        procrastinate_job_id=43,
        state=AnalysisState.IN_PROGRESS,
    )
    manager = _manager(get_job_status_async=AsyncMock(return_value=status))
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    mark_queue_failure = mocker.patch.object(
        TaskAnalysis,
        "mark_queue_failure",
        new_callable=AsyncMock,
        return_value=transitioned,
    )

    await reconcile_analysis_jobs(123)

    mark_queue_failure.assert_awaited_once_with(
        43, SERVER_CONFIG.task_queue.retention_days
    )
    if transitioned:
        log.error.assert_called_once_with(
            "Inconsistent status %s of analysis job %s and state %s of task %s",
            status,
            43,
            AnalysisState.IN_PROGRESS,
            task.task_id,
        )
    else:
        log.error.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("transitioned", [True, False])
async def test_reconcile_marks_lost_result_for_succeeded_job(
    transitioned, mocker, monkeypatch
):
    """A succeeded queue job fails only while its application task is active."""
    log = mocker.patch("logdetective.tasks.LOG")
    task_id = uuid4()
    task = SimpleNamespace(
        task_id=task_id,
        procrastinate_job_id=47,
        state=AnalysisState.IN_PROGRESS,
    )
    manager = _manager(get_job_status_async=AsyncMock(return_value=Status.SUCCEEDED))
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    mark_result_lost = mocker.patch.object(
        TaskAnalysis,
        "mark_result_lost",
        new_callable=AsyncMock,
        return_value=transitioned,
    )
    mark_queue_failure = mocker.patch.object(
        TaskAnalysis, "mark_queue_failure", new_callable=AsyncMock
    )

    await reconcile_analysis_jobs(123)

    mark_result_lost.assert_awaited_once_with(
        47, SERVER_CONFIG.task_queue.retention_days
    )
    mark_queue_failure.assert_not_awaited()
    if transitioned:
        log.error.assert_called_once_with(
            "Analysis job %s succeeded without publishing task %s", 47, task_id
        )
    else:
        log.error.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_fences_missing_queue_jobs(mocker, monkeypatch):
    """An active application record without a queue job is failed durably."""
    task = SimpleNamespace(
        task_id=uuid4(),
        procrastinate_job_id=47,
        state=AnalysisState.SCHEDULED,
    )
    manager = _manager(
        get_job_status_async=AsyncMock(side_effect=NoResult("missing job"))
    )
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    mark_queue_failure = mocker.patch.object(
        TaskAnalysis, "mark_queue_failure", new_callable=AsyncMock
    )

    await reconcile_analysis_jobs(123)

    mark_queue_failure.assert_awaited_once_with(
        47, SERVER_CONFIG.task_queue.retention_days
    )


@pytest.mark.asyncio
async def test_reconcile_confirms_cancellation_when_queue_job_is_missing(
    mocker, monkeypatch
):
    """A missing queue job completes an already durable cancellation request."""
    task_id = uuid4()
    task = SimpleNamespace(
        task_id=task_id,
        procrastinate_job_id=53,
        state=AnalysisState.CANCELLING,
    )
    manager = _manager(
        get_job_status_async=AsyncMock(side_effect=NoResult("missing job"))
    )
    monkeypatch.setattr(app, "job_manager", manager)
    mocker.patch.object(
        TaskAnalysis, "list_active", new_callable=AsyncMock, return_value=[task]
    )
    confirm_cancelled = mocker.patch.object(
        TaskAnalysis, "confirm_cancelled", new_callable=AsyncMock
    )
    mark_queue_failure = mocker.patch.object(
        TaskAnalysis, "mark_queue_failure", new_callable=AsyncMock
    )

    await reconcile_analysis_jobs(123)

    confirm_cancelled.assert_awaited_once_with(
        task_id, 53, SERVER_CONFIG.task_queue.retention_days
    )
    mark_queue_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_expiration_applies_shared_retention(mocker, monkeypatch):
    """Application and Procrastinate records use the same retention window."""
    manager = _manager()
    monkeypatch.setattr(app, "job_manager", manager)
    expire = mocker.patch.object(
        TaskAnalysis, "expire", new_callable=AsyncMock, return_value=3
    )

    await expire_analysis_jobs(456)

    expire.assert_awaited_once_with()
    manager.delete_old_jobs.assert_awaited_once_with(
        SERVER_CONFIG.task_queue.retention_days * 24,
        include_failed=True,
        include_cancelled=True,
        include_aborted=True,
    )
