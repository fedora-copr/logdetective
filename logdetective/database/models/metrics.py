from __future__ import annotations
import enum
from datetime import datetime, timezone
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import (
    Integer,
    String,
    DateTime,
    Enum,
    func,
    select,
    ForeignKey,
    literal_column,
    extract,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from logdetective.database.base import Base, transaction
from logdetective.database.models.merge_request_jobs import (
    GitlabMergeRequestJobs,
)
from logdetective.utils import retry_database_error


if TYPE_CHECKING:
    from logdetective.database.models.tasks import TaskAnalysis


class EndpointType(enum.Enum):
    """Different analyze endpoints"""

    ANALYZE = "analyze"
    ANALYZE_GITLAB_JOB = "analyze_gitlab_job"
    ANALYZE_KOJI_TASK = "analyze_koji_task"


class TimePeriod(enum.Enum):
    """Time periods for metrics aggregation"""

    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class AnalyzeRequestMetrics(Base):
    """Store data related to received requests and given responses"""

    __tablename__ = "analyze_request_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    endpoint: Mapped[EndpointType] = mapped_column(
        Enum(EndpointType),
        nullable=False,
        index=True,
        comment="The service endpoint that was called",
    )
    request_received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=datetime.now(timezone.utc),
        comment="Timestamp when the request was received",
    )
    analysis_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=True,
        comment="Timestamp when the analysis was completed",
    )
    response_sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the response was sent back",
    )
    response_length: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Length of the response in chars"
    )
    api_token_name: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        index=True,
        comment="Non-secret name of the API token used for the request",
    )

    merge_request_job_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("gitlab_merge_request_jobs.id"),
        nullable=True,
        index=False,
        comment="Is this an analyze request coming from a merge request?",
    )

    mr_job: Mapped[Optional["GitlabMergeRequestJobs"]] = relationship(
        "GitlabMergeRequestJobs", back_populates="request_metrics"
    )

    analysis_tasks: Mapped[List["TaskAnalysis"]] = relationship(
        "TaskAnalysis",
        back_populates="analysis_metrics",
    )

    @classmethod
    @retry_database_error
    async def create(
        cls,
        endpoint: EndpointType,
        request_received_at: Optional[datetime] = None,
        api_token_name: Optional[str] = None,
    ) -> int:
        """Create AnalyzeRequestMetrics record with data about received request"""
        async with transaction(commit=True) as session:
            metrics = AnalyzeRequestMetrics()
            metrics.endpoint = endpoint
            metrics.request_received_at = request_received_at or datetime.now(
                timezone.utc
            )
            metrics.api_token_name = api_token_name
            session.add(metrics)
            await session.flush()
            await session.refresh(metrics)
            return metrics.id

    @classmethod
    @retry_database_error
    async def update(
        cls,
        id_: int,
        response_sent_at: datetime,
        response_length: Optional[int] = None,
    ) -> None:
        """Update a row
        with data related to the given response"""
        query = select(AnalyzeRequestMetrics).filter(AnalyzeRequestMetrics.id == id_)
        async with transaction(commit=True) as session:
            query_result = await session.execute(query)
            metrics = query_result.scalars().first()
            if metrics is None:
                raise ValueError("Returned `AnalyzeRequestMetrics` table is empty.")
            metrics.response_sent_at = response_sent_at
            metrics.response_length = response_length
            session.add(metrics)

    @classmethod
    @retry_database_error
    async def get_metric_by_id(
        cls,
        id_: int,
    ) -> "AnalyzeRequestMetrics":
        """Update a row
        with data related to the given response"""
        query = select(AnalyzeRequestMetrics).filter(AnalyzeRequestMetrics.id == id_)
        async with transaction(commit=True) as session:
            query_result = await session.execute(query)
            metric = query_result.scalars().first()
            if metric is None:
                raise ValueError("Returned `AnalyzeRequestMetrics` table is empty.")
            return metric

    @classmethod
    async def get_requests_stats_for_period(
        cls,
        start_time: datetime,
        end_time: datetime,
        time_period: TimePeriod = TimePeriod.DAY,
        endpoint: EndpointType = EndpointType.ANALYZE,
        api_token_name: Optional[str] = None,
    ) -> list[list]:
        """
        Get request counts, average response times and response lengths
        grouped by time units within a specified period.

        Args:
            start_time (datetime): The start of the time period to query
            end_time (datetime): The end of the time period to query
            time_format (str): The strftime format string to format timestamps (e.g., '%Y-%m-%d')
            endpoint (EndpointType): The analyze API endpoint to query
            api_token_name (str): Optional named-token filter

        Returns:
            list[list]: aggregated metrics for the given time period
        """
        query = (
            select(
                func.date_trunc(  # pylint: disable=not-callable
                    time_period.value, AnalyzeRequestMetrics.request_received_at
                ).label("period_start"),
                func.count(AnalyzeRequestMetrics.id).label("total_count"),  # pylint: disable=not-callable
                extract(  # Convert time.timedelta to a second float
                    "epoch",
                    func.avg(
                        AnalyzeRequestMetrics.response_sent_at - AnalyzeRequestMetrics.request_received_at  # pylint: disable=line-too-long
                    ),
                ).label("average_response_time"),
                func.avg(cls.response_length).label("average_response_length"),
                extract(  # Convert time.timedelta to a second float
                    "epoch",
                    func.avg(
                        AnalyzeRequestMetrics.analysis_completed_at - AnalyzeRequestMetrics.request_received_at  # pylint: disable=line-too-long
                    ),
                ).label("average_completion_time"),
            )
            .where(AnalyzeRequestMetrics.request_received_at > start_time)
            .where(AnalyzeRequestMetrics.request_received_at < end_time)
            .where(AnalyzeRequestMetrics.endpoint == endpoint)
            .group_by(
                literal_column("period_start")  # Group by col we have labeled
            )
            .order_by(
                literal_column("period_start")  # Order from the oldest
            )
        )
        if api_token_name is not None:
            query = query.where(AnalyzeRequestMetrics.api_token_name == api_token_name)
        async with transaction(commit=False) as session:
            query_results = await session.execute(query)
            query_results = query_results.all()
        if query_results:
            return [list(col) for col in zip(*query_results)]
        return [[], [], [], [], []]
