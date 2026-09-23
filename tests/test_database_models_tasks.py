"""Tests for application-owned analysis records and fenced transitions."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4
import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from procrastinate import PsycopgConnector
from procrastinate.jobs import Status
from sqlalchemy import text

from tests.test_helpers import DatabaseFactory

from logdetective.database.models.tasks import (
    TaskAnalysis,
    TaskType,
    AnalysisState,
)
from logdetective.database.models import AnalyzeRequestMetrics, EndpointType
from logdetective.database.models.exceptions import (
    AnalysisTaskNotFoundError,
    TaskConflictError,
)
from logdetective.models import APIResponse, KojiTaskMetadata
from logdetective.procrastinate_app import app
from logdetective.tasks import analyze_generic


def test_task_enum_values_are_lowercase_and_match_database_labels() -> None:
    """Keep Python enum values aligned with lowercase PostgreSQL labels."""
    assert [state.value for state in AnalysisState] == [
        "scheduled",
        "in_progress",
        "cancelling",
        "cancelled",
        "done",
        "error",
    ]
    assert [task_type.value for task_type in TaskType] == [
        "generic",
        "koji",
        "gitlab",
    ]
    assert TaskAnalysis.__table__.c.state.type.enums == [
        state.value for state in AnalysisState
    ]
    assert TaskAnalysis.__table__.c.task_type.type.enums == [
        task_type.value for task_type in TaskType
    ]


class FakeDeferrable:
    """Minimal external-connection deferrer used without queue-owned tables."""

    def __init__(self, job_id: int = 41) -> None:
        self.job_id = job_id
        self.connection = None

    def configure(self, *, connection):
        self.connection = connection
        return self

    async def defer_async(self, **_kwargs) -> int:
        return self.job_id


class FakeJobManager:
    """Cancellation delivery fake exposing the released Procrastinate API."""

    def __init__(self, status: Status) -> None:
        self.status = status
        self.cancelled: list[int] = []

    async def cancel_job_by_id_async(
        self, job_id: int, abort: bool = False
    ) -> bool:
        assert abort
        self.cancelled.append(job_id)
        return True

    async def get_job_status_async(self, job_id: int) -> Status:
        assert job_id in self.cancelled
        return self.status


class FailingJobManager:  # pylint: disable=too-few-public-methods
    """Simulate API failure after the durable cancellation transition."""

    async def cancel_job_by_id_async(
        self, _job_id: int, abort: bool = False
    ) -> bool:
        assert abort
        raise RuntimeError("queue connection lost")


@pytest_asyncio.fixture
async def procrastinate_schema() -> AsyncGenerator[None, None]:
    """Install Procrastinate tables for tests that defer real jobs.

    Yields:
        ``None`` while the test-specific Procrastinate connector remains open.
    """
    connector = PsycopgConnector(conninfo=DatabaseFactory.get_pg_test_conninfo())
    with app.replace_connector(connector):
        async with app.open_async():
            await app.schema_manager.apply_schema_async()
            yield


def payload(task_id) -> dict:
    return {
        "id": str(task_id),
        "files": [{"name": "build.log", "content": "error"}],
        "build_metadata": None,
    }


@pytest.mark.asyncio
async def test_admission_and_matching_retry_create_one_record():
    public_id = uuid4()
    deferrer = FakeDeferrable()
    database = DatabaseFactory()
    async with database.make_new_db():
        first, created = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name="client",
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=deferrer,
            endpoint=EndpointType.ANALYZE,
        )
        second, duplicate_created = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name="client",
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(99),
            endpoint=EndpointType.ANALYZE,
        )

        assert created is True
        assert duplicate_created is False
        assert second.id == first.id
        assert first.procrastinate_job_id == 41
        assert first.state == AnalysisState.SCHEDULED
        assert deferrer.connection is not None
        async with database.SessionFactory() as session:
            metric = await session.get(
                AnalyzeRequestMetrics, first.response_metrics_id
            )
            assert metric is not None
            assert metric.response_sent_at is not None
            assert metric.response_sent_at >= metric.request_received_at


@pytest.mark.asyncio
async def test_expired_task_id_cannot_be_reused_before_cleanup():
    public_id = uuid4()
    database = DatabaseFactory()
    async with database.make_new_db():
        task, _ = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name="client",
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=None,
        )
        async with database.SessionFactory.begin() as session:
            stored = await session.get(TaskAnalysis, task.id)
            assert stored is not None
            stored.state = AnalysisState.DONE
            stored.expires_at = datetime.now(UTC) - timedelta(seconds=1)

        with pytest.raises(AnalysisTaskNotFoundError):
            await TaskAnalysis.get_owned(public_id, "client")
        with pytest.raises(TaskConflictError, match="expired"):
            await TaskAnalysis.admit(
                task_id=public_id,
                owner_token_name="client",
                task_type=TaskType.GENERIC,
                input_payload=payload(public_id),
                request_size=10,
                deferrable_task=FakeDeferrable(99),
                endpoint=None,
            )


@pytest.mark.asyncio
async def test_expired_source_id_cannot_be_reused_before_cleanup():
    database = DatabaseFactory()
    async with database.make_new_db():
        task, _ = await TaskAnalysis.admit(
            task_id=uuid4(),
            owner_token_name=None,
            task_type=TaskType.GITLAB,
            input_payload={"forge": "gitlab.com", "status": "failed"},
            request_size=0,
            deferrable_task=FakeDeferrable(),
            endpoint=None,
            source_id="gitlab.com:123",
        )
        async with database.SessionFactory.begin() as session:
            stored = await session.get(TaskAnalysis, task.id)
            assert stored is not None
            stored.state = AnalysisState.DONE
            stored.expires_at = datetime.now(UTC) - timedelta(seconds=1)

        with pytest.raises(TaskConflictError, match="expired"):
            await TaskAnalysis.admit(
                task_id=uuid4(),
                owner_token_name=None,
                task_type=TaskType.GITLAB,
                input_payload={"forge": "gitlab.com", "status": "failed"},
                request_size=0,
                deferrable_task=FakeDeferrable(99),
                endpoint=None,
                source_id="gitlab.com:123",
            )


@pytest.mark.asyncio
async def test_expired_task_id_is_rejected_after_insert_race(mocker):
    public_id = uuid4()
    database = DatabaseFactory()
    async with database.make_new_db():
        task, _ = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name="client",
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=None,
        )
        async with database.SessionFactory.begin() as session:
            stored = await session.get(TaskAnalysis, task.id)
            assert stored is not None
            stored.state = AnalysisState.DONE
            stored.expires_at = datetime.now(UTC) - timedelta(seconds=1)

        expired = await TaskAnalysis._existing(public_id)
        assert expired is not None
        mocker.patch.object(
            TaskAnalysis, "_existing", AsyncMock(side_effect=[None, expired])
        )
        with pytest.raises(TaskConflictError, match="expired"):
            await TaskAnalysis.admit(
                task_id=public_id,
                owner_token_name="client",
                task_type=TaskType.GENERIC,
                input_payload=payload(public_id),
                request_size=10,
                deferrable_task=FakeDeferrable(99),
                endpoint=None,
            )


@pytest.mark.asyncio
async def test_concurrent_admission_atomically_creates_one_real_job(
    procrastinate_schema: None,
) -> None:
    """Verify the losing insert also rolls back its deferred job.

    Args:
        procrastinate_schema: Session fixture providing the queue-owned tables.

    Returns:
        ``None`` after verifying one application record and one queue job exist.
    """
    public_id = uuid4()
    database = DatabaseFactory()
    async with database.make_new_db():
        first, second = await asyncio.gather(
            *(
                TaskAnalysis.admit(
                    task_id=public_id,
                    owner_token_name="client",
                    task_type=TaskType.GENERIC,
                    input_payload=payload(public_id),
                    request_size=10,
                    deferrable_task=analyze_generic,
                    endpoint=EndpointType.ANALYZE,
                )
                for _ in range(2)
            )
        )

        assert first[0].task_id == second[0].task_id == public_id
        assert sorted((first[1], second[1])) == [False, True]
        assert first[0].response_metrics_id is not None
        assert second[0].response_metrics_id == first[0].response_metrics_id
        async with database.SessionFactory() as session:
            job_count = await session.scalar(
                text("SELECT count(*) FROM task_analysis WHERE task_id = :task_id"),
                {"task_id": public_id},
            )
            queue_count = await session.scalar(
                text(
                    "SELECT count(*) FROM procrastinate_jobs "
                    "WHERE args ->> 'task_id' = :task_id"
                ),
                {"task_id": str(public_id)},
            )
            assert job_count is not None
            assert queue_count is not None
            assert job_count == queue_count == 1


@pytest.mark.asyncio
async def test_task_id_rejects_changed_input_or_owner():
    public_id = uuid4()
    async with DatabaseFactory().make_new_db():
        await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name="client",
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE,
        )
        changed = payload(public_id)
        changed["files"][0]["content"] = "different"
        with pytest.raises(TaskConflictError):
            await TaskAnalysis.admit(
                task_id=public_id,
                owner_token_name="client",
                task_type=TaskType.GENERIC,
                input_payload=changed,
                request_size=10,
                deferrable_task=FakeDeferrable(),
                endpoint=EndpointType.ANALYZE,
            )
        with pytest.raises(TaskConflictError):
            await TaskAnalysis.admit(
                task_id=public_id,
                owner_token_name="other",
                task_type=TaskType.GENERIC,
                input_payload=payload(public_id),
                request_size=10,
                deferrable_task=FakeDeferrable(),
                endpoint=EndpointType.ANALYZE,
            )


@pytest.mark.asyncio
async def test_gitlab_redelivery_uses_stable_source_identity():
    source_id = "gitlab.com:123"
    async with DatabaseFactory().make_new_db():
        first, created = await TaskAnalysis.admit(
            task_id=uuid4(),
            owner_token_name=None,
            task_type=TaskType.GITLAB,
            input_payload={"forge": "gitlab.com", "status": "failed"},
            request_size=0,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE_GITLAB_JOB,
            source_id=source_id,
        )
        duplicate, duplicate_created = await TaskAnalysis.admit(
            task_id=uuid4(),
            owner_token_name=None,
            task_type=TaskType.GITLAB,
            input_payload={"forge": "gitlab.com", "status": "failed", "duration": 10},
            request_size=0,
            deferrable_task=FakeDeferrable(99),
            endpoint=EndpointType.ANALYZE_GITLAB_JOB,
            source_id=source_id,
        )

        assert created is True
        assert duplicate_created is False
        assert duplicate.task_id == first.task_id


@pytest.mark.asyncio
async def test_owned_lookup_hides_another_owner():
    public_id = uuid4()
    async with DatabaseFactory().make_new_db():
        await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name="client",
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE,
        )
        with pytest.raises(AnalysisTaskNotFoundError):
            await TaskAnalysis.get_owned(public_id, "other")


@pytest.mark.asyncio
async def test_fenced_result_publication_checks_optional_metric():
    public_id = uuid4()
    async with DatabaseFactory().make_new_db():
        task, _ = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name=None,
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE,
        )
        started = await TaskAnalysis.mark_started(public_id, task.procrastinate_job_id)
        assert started is not None
        assert await TaskAnalysis.publish_result(
            task_id=public_id,
            job_id=task.procrastinate_job_id,
            generation=started.generation,
            response=APIResponse(explanation="done"),
            task_metadata=KojiTaskMetadata(
                task_id=123,
                log_file_name="build.log",
            ),
            retention_days=30,
        )
        published = await TaskAnalysis.get_owned(public_id, None)
        assert published.task_metadata == {
            "task_id": 123,
            "log_file_name": "build.log",
        }
        assert not await TaskAnalysis.publish_result(
            task_id=public_id,
            job_id=task.procrastinate_job_id,
            generation=started.generation,
            response=APIResponse(explanation="stale"),
            task_metadata=None,
            retention_days=30,
        )


@pytest.mark.asyncio
async def test_scheduled_cancellation_is_confirmed_from_queue_status():
    public_id = uuid4()
    manager = FakeJobManager(Status.CANCELLED)
    async with DatabaseFactory().make_new_db():
        await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name=None,
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE,
        )
        cancelled = await TaskAnalysis.request_cancellation(
            public_id, None, manager, retention_days=30
        )
        assert cancelled.state == AnalysisState.CANCELLED
        assert cancelled.input_payload is None
        assert await TaskAnalysis.mark_started(public_id, 41) is None


@pytest.mark.asyncio
async def test_durable_cancellation_blocks_work_when_abort_delivery_fails():
    public_id = uuid4()
    async with DatabaseFactory().make_new_db():
        task, _ = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name=None,
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE,
        )
        cancelling = await TaskAnalysis.request_cancellation(
            public_id, None, FailingJobManager(), retention_days=30
        )

        assert cancelling.state == AnalysisState.CANCELLING
        assert await TaskAnalysis.mark_started(
            public_id, task.procrastinate_job_id
        ) is None


@pytest.mark.asyncio
async def test_queue_failure_fences_late_publication():
    public_id = uuid4()
    async with DatabaseFactory().make_new_db():
        task, _ = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name=None,
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE,
        )
        started = await TaskAnalysis.mark_started(
            public_id, task.procrastinate_job_id
        )
        assert started is not None
        assert await TaskAnalysis.mark_queue_failure(
            task.procrastinate_job_id, retention_days=30
        )
        failed = await TaskAnalysis.get_owned(public_id, None)
        assert failed.error_code == "worker_lost"
        assert not await TaskAnalysis.publish_result(
            task_id=public_id,
            job_id=task.procrastinate_job_id,
            generation=started.generation,
            response=APIResponse(explanation="too late"),
            task_metadata=None,
            retention_days=30,
        )


@pytest.mark.asyncio
async def test_result_loss_fences_late_publication():
    """A missing successful outcome becomes terminal and rejects late results."""
    public_id = uuid4()
    async with DatabaseFactory().make_new_db():
        task, _ = await TaskAnalysis.admit(
            task_id=public_id,
            owner_token_name=None,
            task_type=TaskType.GENERIC,
            input_payload=payload(public_id),
            request_size=10,
            deferrable_task=FakeDeferrable(),
            endpoint=EndpointType.ANALYZE,
        )
        started = await TaskAnalysis.mark_started(
            public_id, task.procrastinate_job_id
        )
        assert started is not None

        assert await TaskAnalysis.mark_result_lost(
            task.procrastinate_job_id, retention_days=30
        )
        failed = await TaskAnalysis.get_owned(public_id, None)
        assert failed.state == AnalysisState.ERROR
        assert failed.error_code == "result_lost"
        expected_message = "Analysis completed without publishing a result"
        assert failed.error_message == expected_message
        assert not await TaskAnalysis.publish_result(
            task_id=public_id,
            job_id=task.procrastinate_job_id,
            generation=started.generation,
            response=APIResponse(explanation="too late"),
            task_metadata=None,
            retention_days=30,
        )
