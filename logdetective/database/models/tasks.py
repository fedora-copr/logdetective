from __future__ import annotations
from typing import Optional
from datetime import datetime, timedelta, timezone
import enum
from uuid import uuid4
from uuid import UUID as uuid_type
from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    LargeBinary,
    JSON,
    Select,
    UUID,
    select,
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

from logdetective.config import SERVER_CONFIG
from logdetective.compressors import LLMResponseCompressor
from logdetective.database.models.metrics import AnalyzeRequestMetrics
from logdetective.database.base import Base, transaction
from logdetective.database.models.exceptions import (
    TaskNotAnalyzedError,
    TaskAnalysisTimeoutError,
    AnalyzeRequestMetricsNotFoundError,
    AnalysisTaskNotFoundError,
)
from logdetective.models import APIResponse
from logdetective.utils import retry_database_error


class AnalysisState(enum.Enum):
    """State of the analysis task"""

    SCHEDULED = "scheduled"
    DONE = "done"
    IN_PROGRESS = "in_progress"
    ERROR = "error"


class TaskType(enum.Enum):
    """Type of the task"""

    GENERIC = "generic"
    KOJI = "koji"


class TaskAnalysis(Base):
    """Store information about analysis tasks"""

    __tablename__ = "task_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[uuid_type] = mapped_column(
        UUID,
        nullable=False,
        index=True,
        unique=True,
        insert_default=uuid4,
        comment="Task ID as UUID-4 string",
    )
    # pylint: disable=duplicate-code
    request_received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        insert_default=lambda: datetime.now(timezone.utc),
        comment="Timestamp when the request was received",
    )
    response_returned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=True,
        comment="Timestamp when the analysis result was returned to caller",
    )
    response_metrics_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("analyze_request_metrics.id"),
        nullable=True,
        index=False,
        comment="The id of the analyze_request_metrics record for this task",
    )
    state: Mapped[AnalysisState] = mapped_column(
        Enum(AnalysisState),
        nullable=False,
        index=True,
        default=AnalysisState.SCHEDULED,
        comment="State of the analysis task",
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of attempts made to finish analysis",
    )
    task_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional information supplied with the task"
    )
    response: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary(length=314572800),  # 300MB limit (300 * 1024 * 1024)
        nullable=True,
        index=False,
        comment="Given response (with explanation and snippets) saved in a zip format",
    )
    task_type: Mapped[TaskType] = mapped_column(
        Enum(TaskType), nullable=False, comment="Type of the task being processed"
    )
    external_task_id: Mapped[Optional[str]] = mapped_column(
        String,
        index=True,
        unique=True,
        comment="Task ID supplied by external system"
    )
    analysis_metrics: Mapped[Optional["AnalyzeRequestMetrics"]] = relationship(
        "AnalyzeRequestMetrics", back_populates="analysis_tasks"
    )

    @classmethod
    @retry_database_error
    async def create(
        cls,
        task_type: TaskType,
        external_task_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> uuid_type:
        """Create new analysis task and return task_id (UUID-4)"""

        async with transaction(commit=True) as session:
            task = TaskAnalysis()
            task.request_received_at = datetime.now(timezone.utc)
            task.external_task_id = external_task_id
            task.task_type = task_type
            task.task_metadata = metadata
            task.state = AnalysisState.SCHEDULED
            session.add(task)
            await session.flush()
            return task.task_id

    @classmethod
    @retry_database_error
    async def create_or_restart(
        cls, task_type: TaskType, external_task_id: str, metadata: Optional[dict] = None
    ) -> uuid_type:
        """Create or restart new task with external id"""
        query = select(cls).filter(
            cls.external_task_id == external_task_id).with_for_update(skip_locked=True)
        async with transaction(commit=True) as session:
            # Check if the task analysis already exists
            query_result = await session.execute(query)
            task = query_result.scalars().first()
            if task:
                # If it does, update the request_received_at timestamp
                task.request_received_at = datetime.now(timezone.utc)
                task.attempt_count += 1
                session.add(task)
                await session.flush()
                return task.task_id

        # If it doesn't, create a new one
        return await cls.create(
            task_type=task_type, external_task_id=external_task_id, metadata=metadata
        )

    @classmethod
    @retry_database_error
    async def add_response(
        cls, task_id: uuid_type, metric_id: int, response: APIResponse
    ):
        """Add response to analysis task and update metrics"""

        query = (
            select(cls).filter(cls.task_id == task_id).with_for_update(skip_locked=True)
        )
        metrics_query = select(AnalyzeRequestMetrics).filter(
            AnalyzeRequestMetrics.id == metric_id
        )

        async with transaction(commit=True) as session:
            query_result = await session.execute(query)
            task_analysis = query_result.scalars().first()
            if not task_analysis:
                raise AnalysisTaskNotFoundError(
                    f"No TaskAnalysis record found for id {task_id}"
                )

            if task_analysis.response:
                return

            metrics_query_result = await session.execute(metrics_query)
            metric = metrics_query_result.scalars().first()
            if not metric:
                raise AnalyzeRequestMetricsNotFoundError(
                    f"No AnalyzeRequestMetrics record found for id {metric_id}"
                )

            metric.response_length = len(response.model_dump_json())
            metric.analysis_completed_at = datetime.now(timezone.utc)
            task_analysis.response = LLMResponseCompressor(response).zip_response()
            task_analysis.state = AnalysisState.DONE
            task_analysis.response_metrics_id = metric_id
            session.add(task_analysis)
            session.add(metric)
            await session.flush()

    @classmethod
    async def get_task(cls, query: Select, identifier: str) -> TaskAnalysis:
        """Get analysis response using query."""
        async with transaction(commit=True) as session:
            query_result = await session.execute(query)
            task_analysis = query_result.scalars().first()
            if not task_analysis:
                raise AnalysisTaskNotFoundError(f"Task {identifier} does not exist")

            if not task_analysis.response:
                # Check if the task analysis has timed out
                if task_analysis.request_received_at.replace(
                    tzinfo=timezone.utc
                ) + timedelta(
                    seconds=SERVER_CONFIG.general.analysis_timeout
                ) < datetime.now(timezone.utc):
                    raise TaskAnalysisTimeoutError(
                        f"Task {identifier} analysis has timed out"
                    )

                # Task analysis is still in progress, so we need to let the
                # consumer know
                raise TaskNotAnalyzedError(
                    f"Task {identifier} analysis is still in progress"
                )

            task_analysis.response_returned_at = datetime.now(timezone.utc)
            session.add(task_analysis)
            await session.flush()
            await session.refresh(task_analysis)
            return task_analysis

    @classmethod
    @retry_database_error
    async def get_task_by_id(cls, task_id: uuid_type) -> TaskAnalysis:
        """Get analysis task response by task id"""

        query = (
            select(cls).filter(cls.task_id == task_id).with_for_update(skip_locked=True)
        )
        return await cls.get_task(query=query, identifier=str(task_id))

    @classmethod
    @retry_database_error
    async def get_task_by_external_id(cls, external_task_id: str) -> TaskAnalysis:
        """Get analysis task response by task id"""

        query = (
            select(cls)
            .filter(cls.external_task_id == external_task_id)
            .with_for_update(skip_locked=True)
        )
        return await cls.get_task(query=query, identifier=external_task_id)
