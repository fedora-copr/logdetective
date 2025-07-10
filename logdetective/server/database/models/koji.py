from datetime import datetime, timedelta, timezone
from sqlalchemy import Column, BigInteger, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.exc import OperationalError
import backoff

from logdetective.server.config import SERVER_CONFIG
from logdetective.server.compressors import LLMResponseCompressor
from logdetective.server.database.models.metrics import AnalyzeRequestMetrics
from logdetective.server.database.base import Base, transaction, DB_MAX_RETRIES
from logdetective.server.database.models.exceptions import (
    KojiTaskNotFoundError,
    KojiTaskNotAnalyzedError,
    KojiTaskAnalysisTimeoutError,
)
from logdetective.server.models import KojiStagedResponse


class KojiTaskAnalysis(Base):
    """Store details for the koji task analysis"""

    __tablename__ = "koji_task_analysis"

    id = Column(Integer, primary_key=True)
    koji_instance = Column(String(255), nullable=False, index=True)
    task_id = Column(BigInteger, nullable=False, index=True, unique=True)
    log_file_name = Column(String(255), nullable=False, index=True)
    request_received_at = Column(
        DateTime,
        nullable=False,
        index=True,
        default=datetime.now(timezone.utc),
        comment="Timestamp when the request was received",
    )
    response_id = Column(
        Integer,
        ForeignKey("analyze_request_metrics.id"),
        nullable=True,
        index=False,
        comment="The id of the analyze request metrics for this task",
    )
    response = relationship("AnalyzeRequestMetrics")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def create_or_restart(cls, koji_instance: str, task_id: int, log_file_name: str):
        """Create a new koji task analysis"""
        with transaction(commit=True) as session:
            # Check if the task analysis already exists
            koji_task_analysis = (
                session.query(cls)
                .filter_by(koji_instance=koji_instance, task_id=task_id)
                .first()
            )
            if koji_task_analysis:
                # If it does, update the request_received_at timestamp
                koji_task_analysis.request_received_at = datetime.now(timezone.utc)
                session.add(koji_task_analysis)
                session.flush()
                return

            # If it doesn't, create a new one
            koji_task_analysis = KojiTaskAnalysis()
            koji_task_analysis.koji_instance = koji_instance
            koji_task_analysis.task_id = task_id
            koji_task_analysis.log_file_name = log_file_name
            session.add(koji_task_analysis)
            session.flush()

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def add_response(cls, task_id: int, metric_id: int):
        """Add a response to a koji task analysis"""
        with transaction(commit=True) as session:
            koji_task_analysis = session.query(cls).filter_by(task_id=task_id).first()
            # Ensure that the task analysis doesn't already have a response
            if koji_task_analysis.response:
                # This is probably due to an analysis that took so long that
                # a follow-up analysis was started before this one completed.
                # We want to maintain consistency between the response we
                # returned to the consumer, so we'll just drop this extra one
                # on the floor and keep the one saved in the database.
                return

            metric = (
                session.query(AnalyzeRequestMetrics).filter_by(id=metric_id).first()
            )
            koji_task_analysis.response = metric
            session.add(koji_task_analysis)
            session.flush()

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def get_response_by_task_id(cls, task_id: int) -> KojiStagedResponse:
        """Get a koji task analysis by task id"""
        with transaction(commit=False) as session:
            koji_task_analysis = session.query(cls).filter_by(task_id=task_id).first()
            if not koji_task_analysis:
                raise KojiTaskNotFoundError(f"Task {task_id} not yet analyzed")

            if not koji_task_analysis.response:
                # Check if the task analysis has timed out
                if koji_task_analysis.request_received_at.replace(
                    tzinfo=timezone.utc
                ) + timedelta(
                    minutes=SERVER_CONFIG.koji.analysis_timeout
                ) < datetime.now(timezone.utc):
                    raise KojiTaskAnalysisTimeoutError(
                        f"Task {task_id} analysis has timed out"
                    )

                # Task analysis is still in progress, so we need to let the
                # consumer know
                raise KojiTaskNotAnalyzedError(
                    f"Task {task_id} analysis is still in progress"
                )

            # We need to decompress the response message and return it
            response = LLMResponseCompressor.unzip(
                koji_task_analysis.response.compressed_response
            )
            return KojiStagedResponse(
                task_id=task_id,
                log_file_name=koji_task_analysis.log_file_name,
                response=response,
            )
