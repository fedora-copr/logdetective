"""Application-owned analysis records and their durable state transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from datetime import datetime, timedelta, UTC
import hashlib
import json
import logging
from uuid import uuid4
from uuid import UUID as UUIDType

from procrastinate.jobs import Status
from procrastinate.manager import JobManager
from procrastinate.tasks import Task
from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    LargeBinary,
    JSON,
    UUID,
    select,
    BigInteger,
    Index,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship

from logdetective.compressors import LLMResponseCompressor
from logdetective.database.models.metrics import AnalyzeRequestMetrics, EndpointType
from logdetective.database.base import Base, enum_values, transaction
from logdetective.database.models.exceptions import (
    AnalysisTaskNotFoundError,
    TaskConflictError,
    TaskTerminalError,
)
from logdetective.models import AnalysisState, TaskMetadata, TaskType

if TYPE_CHECKING:
    from logdetective.models import APIResponse

LOG = logging.getLogger("logdetective")


ACTIVE_STATES = (
    AnalysisState.SCHEDULED,
    AnalysisState.IN_PROGRESS,
    AnalysisState.CANCELLING,
)
TERMINAL_STATES = (
    AnalysisState.CANCELLED,
    AnalysisState.DONE,
    AnalysisState.ERROR,
)


class TaskAnalysis(Base):  # pylint: disable=too-many-instance-attributes
    """Durable input, public state, and result for one analysis operation."""

    __tablename__ = "task_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[UUIDType] = mapped_column(
        UUID, nullable=False, index=True, unique=True, default=uuid4
    )
    owner_token_name: Mapped[str | None] = mapped_column(String, index=True)
    task_type: Mapped[TaskType] = mapped_column(
        Enum(TaskType, values_callable=enum_values),
        nullable=False,
    )
    source_id: Mapped[str | None] = mapped_column(String, index=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    request_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    procrastinate_job_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, unique=True, index=True
    )
    state: Mapped[AnalysisState] = mapped_column(
        Enum(AnalysisState, values_callable=enum_values),
        nullable=False,
        index=True,
        default=AnalysisState.SCHEDULED,
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
        default=lambda: datetime.now(UTC),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    response_metrics_id: Mapped[int | None] = mapped_column(
        ForeignKey("analyze_request_metrics.id")
    )
    response: Mapped[bytes | None] = mapped_column(LargeBinary)
    task_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(String(255))

    analysis_metrics: Mapped[AnalyzeRequestMetrics | None] = relationship(
        "AnalyzeRequestMetrics", back_populates="analysis_tasks"
    )

    __table_args__ = (
        Index(
            "uix_task_analysis_source",
            "task_type",
            "source_id",
            unique=True,
            postgresql_where=text("source_id IS NOT NULL"),
        ),
    )

    @staticmethod
    def hash_payload(task_type: TaskType, payload: dict[str, Any]) -> str:
        """Calculate the stable identity of validated task input.

        Args:
            task_type: Kind of analysis represented by the payload.
            payload: JSON-compatible durable task input. Its public ``id`` field is
                excluded because the identifier is stored separately.

        Returns:
            A hexadecimal SHA-256 digest of the normalized task type and payload.
        """
        normalized = dict(payload)
        normalized.pop("id", None)
        serialized = json.dumps(
            {"task_type": task_type.value, "payload": normalized},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    async def _existing(cls, task_id: UUIDType) -> TaskAnalysis | None:
        """Look up an application task by its public identifier.

        Args:
            task_id: Public UUID assigned by the client or API server.

        Returns:
            The matching task, or ``None`` when the identifier is unused.
        """
        async with transaction() as session:
            return await session.scalar(select(cls).where(cls.task_id == task_id))

    @classmethod
    async def _existing_source(
        cls, task_type: TaskType, source_id: str | None
    ) -> TaskAnalysis | None:
        """Look up a task by its external source identity.

        Args:
            task_type: Kind of task to which the source identifier belongs.
            source_id: Stable identifier supplied by an external system, or ``None``
                when the task has no external identity.

        Returns:
            The matching task, or ``None`` when no source identity was supplied or
            no record matches it.
        """
        if source_id is None:
            return None
        async with transaction() as session:
            return await session.scalar(
                select(cls).where(
                    cls.task_type == task_type, cls.source_id == source_id
                )
            )

    @staticmethod
    def _verify_unexpired(task: TaskAnalysis) -> None:
        """Reject a retained task once its public polling window has ended.

        Args:
            task: Existing application task considered for admission reuse.

        Raises:
            TaskConflictError: If the task is no longer visible through polling.
        """
        if task.expires_at is not None and task.expires_at <= datetime.now(UTC):
            raise TaskConflictError("Task has expired")

    @classmethod
    def _verify_retry(
        cls,
        task: TaskAnalysis,
        owner_token_name: str | None,
        task_type: TaskType,
        request_hash: str,
    ) -> None:
        """Verify that an existing task is an idempotent retry of a request.

        Args:
            task: Existing application task found for the public identifier.
            owner_token_name: Non-secret identity of the submitting API token.
            task_type: Kind of task requested by the retry.
            request_hash: Hash of the retry's normalized payload.

        Returns:
            ``None`` when the retry matches the existing task.

        Raises:
            TaskConflictError: If ownership, task type, or input differs from the
            existing request.
        """
        if any(
            (
                task.owner_token_name != owner_token_name,
                task.task_type != task_type,
                task.request_hash != request_hash,
            )
        ):
            raise TaskConflictError("Task id is already in use")

    @classmethod
    async def _finish_metric(
        cls,
        session: AsyncSession,
        metric_id: int | None,
        finished_at: datetime,
        response_length: int | None = None,
    ) -> None:
        """Mark an optional request metric as completed.

        Args:
            session: Session participating in the task's completion transaction.
            metric_id: Related metrics primary key, or ``None`` for unmetered work.
            finished_at: UTC time at which task processing finished.
            response_length: Optional serialized response length in characters.

        Returns:
            ``None`` after the optional metric has been updated.

        Raises:
            RuntimeError: If a referenced metrics record no longer exists.
        """
        if metric_id is None:
            return
        metric = await session.get(AnalyzeRequestMetrics, metric_id)
        if metric is None:
            raise RuntimeError("Analysis metric disappeared before task completion")
        metric.analysis_completed_at = finished_at
        if response_length is not None:
            metric.response_length = response_length

    @classmethod
    async def admit(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        cls,
        *,
        task_id: UUIDType,
        owner_token_name: str | None,
        task_type: TaskType,
        input_payload: dict[str, Any],
        request_size: int,
        deferrable_task: Task,
        endpoint: EndpointType | None,
        source_id: str | None = None,
    ) -> tuple[TaskAnalysis, bool]:
        """Atomically admit a durable application task and Procrastinate job.

        Args:
            task_id: Public UUID selected by the client or API server.
            owner_token_name: Non-secret identity of the submitting API token.
            task_type: Kind of analysis to schedule.
            input_payload: Validated, JSON-compatible input required by the worker.
            request_size: Submitted request size in bytes.
            deferrable_task: Procrastinate task definition to enqueue.
            endpoint: Metrics endpoint associated with the request, or ``None`` when
                the operation is not metered.
            source_id: Optional stable identity supplied by an external system.

        Returns:
            A pair containing the durable task and ``True`` when this call created
            it. An idempotent retry returns the existing task and ``False``.

        Raises:
            TaskConflictError: If an existing identifier or source belongs to
                incompatible input, or its task has expired.
            RuntimeError: If Procrastinate or metrics creation fails to return an
                identifier.
        """
        request_hash = cls.hash_payload(task_type, input_payload)
        request_received_at = datetime.now(UTC)
        existing = await cls._existing(task_id)
        if existing is not None:
            cls._verify_unexpired(existing)
            cls._verify_retry(existing, owner_token_name, task_type, request_hash)
            return existing, False
        existing = await cls._existing_source(task_type, source_id)
        if existing is not None:
            cls._verify_unexpired(existing)
            # GitLab can redeliver the same job event with updated incidental
            # fields. Its stable forge/build identity owns one internal task.
            if task_type == TaskType.GITLAB:
                return existing, False
            if existing.request_hash != request_hash:
                raise TaskConflictError("Source task already exists with different input")
            return existing, False

        try:
            async with transaction(commit=True) as session:
                connection = await session.connection()
                raw_connection = (await connection.get_raw_connection()).driver_connection
                job_id = await deferrable_task.configure(
                    connection=raw_connection
                ).defer_async(task_id=str(task_id))
                if job_id is None:
                    raise RuntimeError("Procrastinate did not return a job id")

                metric = None
                if endpoint is not None:
                    metric = await AnalyzeRequestMetrics.create_in_session(
                        session,
                        endpoint=endpoint,
                        request_received_at=request_received_at,
                        api_token_name=owner_token_name,
                    )
                    if metric is None:
                        raise RuntimeError("Unable to create analysis metrics")

                task = cls(
                    task_id=task_id,
                    owner_token_name=owner_token_name,
                    task_type=task_type,
                    source_id=source_id,
                    request_hash=request_hash,
                    input_payload=input_payload,
                    request_size=request_size,
                    procrastinate_job_id=job_id,
                    response_metrics_id=metric.id if metric is not None else None,
                    state=AnalysisState.SCHEDULED,
                    request_received_at=request_received_at,
                )
                session.add(task)
                await session.flush()
                await session.refresh(task)
                if metric is not None:
                    metric.response_sent_at = datetime.now(UTC)
                return task, True
        except IntegrityError as exc:
            existing = await cls._existing(task_id)
            if existing is None:
                existing = await cls._existing_source(task_type, source_id)
            if existing is None:
                raise
            cls._verify_unexpired(existing)
            if existing.task_id == task_id:
                cls._verify_retry(existing, owner_token_name, task_type, request_hash)
            elif task_type == TaskType.GITLAB:
                return existing, False
            elif existing.request_hash != request_hash:
                raise TaskConflictError(
                    "Source task already exists with different input"
                ) from exc
            return existing, False

    @classmethod
    async def get_owned(
        cls, task_id: UUIDType, owner_token_name: str | None
    ) -> TaskAnalysis:
        """Retrieve an unexpired task visible to one API owner.

        Args:
            task_id: Public task UUID to retrieve.
            owner_token_name: Non-secret token identity attached during admission.

        Returns:
            The matching task record.

        Raises:
            AnalysisTaskNotFoundError: If the task is absent, expired, or belongs to
                another owner.
        """
        now = datetime.now(UTC)
        async with transaction() as session:
            task = await session.scalar(
                select(cls).where(
                    cls.task_id == task_id,
                    cls.owner_token_name == owner_token_name,
                    (cls.expires_at.is_(None) | (cls.expires_at > now)),
                )
            )
            if task is None:
                raise AnalysisTaskNotFoundError(f"Task {task_id} does not exist")
            return task

    @classmethod
    async def mark_started(
        cls, task_id: UUIDType, job_id: int
    ) -> TaskAnalysis | None:
        """Claim publication rights for a scheduled Procrastinate job.

        Args:
            task_id: Public UUID of the application task.
            job_id: Procrastinate job identifier attempting to start the task.

        Returns:
            The task transitioned to ``in_progress``, or ``None`` if the task is no
            longer scheduled or the job does not own it.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            task = await session.scalar(
                update(cls)
                .where(
                    cls.task_id == task_id,
                    cls.procrastinate_job_id == job_id,
                    cls.state == AnalysisState.SCHEDULED,
                )
                .values(state=AnalysisState.IN_PROGRESS, started_at=now)
                .returning(cls)
            )
            return task

    @classmethod
    async def publish_result(  # pylint: disable=too-many-arguments
        cls,
        *,
        task_id: UUIDType,
        job_id: int,
        generation: int,
        response: APIResponse,
        task_metadata: TaskMetadata | None,
        retention_days: int,
    ) -> bool:
        """Publish a successful result while the same attempt still owns the task.

        Args:
            task_id: Public UUID of the application task.
            job_id: Procrastinate job identifier that produced the result.
            generation: Cancellation fence generation captured when work started.
            response: Validated analysis response to compress and persist.
            task_metadata: Optional validated task-specific result metadata.
            retention_days: Number of days to retain the terminal task.

        Returns:
            ``True`` when the fenced update published the result; ``False`` when the
            task changed state or ownership before publication.

        Raises:
            RuntimeError: If the related request metrics record disappeared.
        """

        now = datetime.now(UTC)
        compressed = LLMResponseCompressor(response).zip_response()
        serialized_metadata = (
            task_metadata.model_dump(mode="json")
            if task_metadata is not None
            else None
        )
        async with transaction(commit=True) as session:
            task = await session.scalar(
                update(cls)
                .where(
                    cls.task_id == task_id,
                    cls.procrastinate_job_id == job_id,
                    cls.generation == generation,
                    cls.state == AnalysisState.IN_PROGRESS,
                )
                .values(
                    state=AnalysisState.DONE,
                    response=compressed,
                    task_metadata=serialized_metadata,
                    input_payload=None,
                    finished_at=now,
                    expires_at=now + timedelta(days=retention_days),
                )
                .returning(cls)
            )
            if task is None:
                return False
            await cls._finish_metric(
                session,
                task.response_metrics_id,
                now,
                len(response.model_dump_json()),
            )
            return True

    @classmethod
    async def publish_error(  # pylint: disable=too-many-arguments
        cls,
        *,
        task_id: UUIDType,
        job_id: int,
        generation: int,
        code: str,
        message: str,
        retention_days: int,
    ) -> bool:
        """Publish a safe terminal error through the attempt fence.

        Args:
            task_id: Public UUID of the application task.
            job_id: Procrastinate job identifier reporting the failure.
            generation: Cancellation fence generation captured by the worker.
            code: Stable public error code.
            message: Sanitized public error message.
            retention_days: Number of days to retain the terminal task.

        Returns:
            ``True`` when the error was published; ``False`` when the attempt no
            longer owns an active task.

        Raises:
            RuntimeError: If the related request metrics record disappeared.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            task = await session.scalar(
                update(cls)
                .where(
                    cls.task_id == task_id,
                    cls.procrastinate_job_id == job_id,
                    cls.generation == generation,
                    cls.state.in_((AnalysisState.SCHEDULED, AnalysisState.IN_PROGRESS)),
                )
                .values(
                    state=AnalysisState.ERROR,
                    error_code=code,
                    error_message=message,
                    input_payload=None,
                    finished_at=now,
                    expires_at=now + timedelta(days=retention_days),
                )
                .returning(cls)
            )
            if task is None:
                return False
            await cls._finish_metric(session, task.response_metrics_id, now)
            return True

    @classmethod
    async def complete_without_result(
        cls,
        *,
        task_id: UUIDType,
        job_id: int,
        generation: int,
        retention_days: int,
    ) -> bool:
        """Complete an internal webhook operation that has no public result.

        Args:
            task_id: Public UUID used internally for the webhook task.
            job_id: Procrastinate job identifier that processed the webhook.
            generation: Cancellation fence generation captured by the worker.
            retention_days: Number of days to retain the terminal task.

        Returns:
            ``True`` when completion was published; ``False`` when the attempt no
            longer owns an in-progress task.

        Raises:
            RuntimeError: If the related request metrics record disappeared.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            task = await session.scalar(
                update(cls)
                .where(
                    cls.task_id == task_id,
                    cls.procrastinate_job_id == job_id,
                    cls.generation == generation,
                    cls.state == AnalysisState.IN_PROGRESS,
                )
                .values(
                    state=AnalysisState.DONE,
                    input_payload=None,
                    finished_at=now,
                    expires_at=now + timedelta(days=retention_days),
                )
                .returning(cls)
            )
            if task is None:
                return False
            await cls._finish_metric(session, task.response_metrics_id, now)
            return True

    @classmethod
    async def request_cancellation(
        cls,
        task_id: UUIDType,
        owner_token_name: str | None,
        job_manager: JobManager,
        retention_days: int,
    ) -> TaskAnalysis:
        """Persist cancellation and ask Procrastinate to deliver it.

        Args:
            task_id: Public UUID of the task to cancel.
            owner_token_name: Non-secret identity of the task owner.
            job_manager: Procrastinate manager used to abort and inspect the job.
            retention_days: Number of days to retain a confirmed cancellation.

        Returns:
            The current task record. Its state is ``cancelled`` when process cleanup
            was confirmed immediately, otherwise ``cancelling`` while reconciliation
            remains pending.

        Raises:
            AnalysisTaskNotFoundError: If the task does not exist or belongs to a
                different owner.
            TaskTerminalError: If a completed or failed task cannot be cancelled.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            task = await session.scalar(
                select(cls)
                .where(cls.task_id == task_id, cls.owner_token_name == owner_token_name)
                .with_for_update()
            )
            if task is None:
                raise AnalysisTaskNotFoundError(f"Task {task_id} does not exist")
            if task.state in (AnalysisState.DONE, AnalysisState.ERROR):
                raise TaskTerminalError("Completed tasks cannot be cancelled")
            if task.state == AnalysisState.CANCELLED:
                return task
            if task.state != AnalysisState.CANCELLING:
                task.cancellation_requested_at = now
                task.generation += 1
                task.state = AnalysisState.CANCELLING
            await session.flush()
            await session.refresh(task)

        # Procrastinate 3.9 has no public external-connection cancellation API.
        # A repeated DELETE deliberately retries this best-effort delivery.
        try:
            cancelled = await job_manager.cancel_job_by_id_async(
                task.procrastinate_job_id, abort=True
            )
            if not cancelled:
                LOG.warning(
                    "Cancellation delivery found no job for task %s", task.task_id
                )
                return task
            status = await job_manager.get_job_status_async(task.procrastinate_job_id)
            if status == Status.CANCELLED:
                await cls.confirm_cancelled(task.task_id, task.procrastinate_job_id, retention_days)
                return await cls.get_owned(task.task_id, owner_token_name)
        # Delivery failures must leave the durable cancelling state available
        # for the periodic reconciler. Connector exceptions are deliberately
        # not narrowed because Procrastinate wraps driver failures dynamically.
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            LOG.warning("Cancellation delivery deferred for task %s: %s", task.task_id, exc)
        return task

    @classmethod
    async def confirm_cancelled(
        cls, task_id: UUIDType, job_id: int, retention_days: int
    ) -> bool:
        """Confirm cancellation after process-tree cleanup completes.

        Args:
            task_id: Public UUID of the application task.
            job_id: Procrastinate job identifier whose cleanup was confirmed.
            retention_days: Number of days to retain the cancelled task.

        Returns:
            ``True`` when the task transitioned from ``cancelling`` to ``cancelled``;
            ``False`` when the identifiers or current state no longer match.

        Raises:
            RuntimeError: If the related request metrics record disappeared.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            task = await session.scalar(
                update(cls)
                .where(
                    cls.task_id == task_id,
                    cls.procrastinate_job_id == job_id,
                    cls.state == AnalysisState.CANCELLING,
                )
                .values(
                    state=AnalysisState.CANCELLED,
                    input_payload=None,
                    finished_at=now,
                    expires_at=now + timedelta(days=retention_days),
                )
                .returning(cls)
            )
            if task is None:
                return False
            await cls._finish_metric(session, task.response_metrics_id, now)
            return True

    @classmethod
    async def expire(cls) -> int:
        """Remove terminal application records after their retention window.

        Returns:
            Number of task records deleted in the transaction.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            tasks = (
                await session.scalars(
                    select(cls).where(
                        cls.state.in_(TERMINAL_STATES), cls.expires_at <= now
                    )
                )
            ).all()
            for task in tasks:
                await session.delete(task)
            return len(tasks)

    @classmethod
    async def list_active(cls) -> list[TaskAnalysis]:
        """List application records that still require queue reconciliation.

        Returns:
            All scheduled, in-progress, and cancelling task records.
        """
        async with transaction() as session:
            return list(
                (await session.scalars(select(cls).where(cls.state.in_(ACTIVE_STATES)))).all()
            )

    @classmethod
    async def mark_queue_failure(cls, job_id: int, retention_days: int) -> bool:
        """Fail an active task whose queue job cannot deliver an outcome.

        The reconciler uses this for a stalled or missing job, or for a job in
        FAILED, ABORTED, or CANCELLED state without the expected application
        transition.

        Args:
            job_id: Procrastinate identifier of the affected queue job.
            retention_days: Number of days to retain the terminal failure.

        Returns:
            ``True`` when an active task was marked failed; ``False`` when no active
            task is owned by the job.

        Raises:
            RuntimeError: If the related request metrics record disappeared.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            task = await session.scalar(
                update(cls)
                .where(
                    cls.procrastinate_job_id == job_id,
                    cls.state.in_(ACTIVE_STATES),
                )
                .values(
                    state=AnalysisState.ERROR,
                    generation=cls.generation + 1,
                    error_code="worker_lost",
                    error_message="The analysis worker was lost",
                    input_payload=None,
                    finished_at=now,
                    expires_at=now + timedelta(days=retention_days),
                )
                .returning(cls)
            )
            if task is None:
                return False
            await cls._finish_metric(session, task.response_metrics_id, now)
            return True

    @classmethod
    async def mark_result_lost(cls, job_id: int, retention_days: int) -> bool:
        """Fail a task whose queue job succeeded without publishing its outcome.

        Args:
            job_id: Procrastinate identifier of the successfully completed job.
            retention_days: Number of days to retain the terminal failure.

        Returns:
            ``True`` when a scheduled or in-progress task was marked failed;
            ``False`` when the task is already terminal or cancellation owns it.

        Raises:
            RuntimeError: If the related request metrics record disappeared.
        """
        now = datetime.now(UTC)
        async with transaction(commit=True) as session:
            task = await session.scalar(
                update(cls)
                .where(
                    cls.procrastinate_job_id == job_id,
                    cls.state.in_(
                        (AnalysisState.SCHEDULED, AnalysisState.IN_PROGRESS)
                    ),
                )
                .values(
                    state=AnalysisState.ERROR,
                    generation=cls.generation + 1,
                    error_code="result_lost",
                    error_message="Analysis completed without publishing a result",
                    input_payload=None,
                    finished_at=now,
                    expires_at=now + timedelta(days=retention_days),
                )
                .returning(cls)
            )
            if task is None:
                return False
            await cls._finish_metric(session, task.response_metrics_id, now)
            return True
