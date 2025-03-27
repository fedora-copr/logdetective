import enum
import datetime

from typing import Optional
from sqlalchemy import Column, Integer, Float, DateTime, String, Enum

from logdetective.server.database.base import Base, transaction


class EndpointType(enum.Enum):
    """Different analyze endpoints"""

    ANALYZE = "analyze_log"
    ANALYZE_STAGED = "analyze_log_staged"
    ANALYZE_STREAM = "analyze_log_stream"


class AnalyzeRequestMetrics(Base):
    """Store data related to received requests and given responses"""

    __tablename__ = "analyze_request_metrics"

    id = Column(Integer, primary_key=True)
    endpoint = Column(
        Enum(EndpointType),
        nullable=False,
        index=True,
        comment="The service endpoint that was called",
    )
    request_received_at = Column(
        DateTime,
        nullable=False,
        index=True,
        default=datetime.datetime.now(datetime.timezone.utc),
        comment="Timestamp when the request was received",
    )
    log_url = Column(
        String,
        nullable=False,
        index=False,
        comment="Log url for which analysis was requested",
    )
    response_sent_at = Column(
        DateTime, nullable=True, comment="Timestamp when the response was sent back"
    )
    response_length = Column(
        Integer, nullable=True, comment="Length of the response in chars"
    )
    response_certainty = Column(
        Float, nullable=True, comment="Certainty for generated response"
    )

    @classmethod
    def create(
        cls,
        endpoint: EndpointType,
        log_url: str,
        request_received_at: Optional[datetime.datetime] = None,
    ) -> int:
        """Create AnalyzeRequestMetrics new line
        with data related to a received request"""
        with transaction(commit=True) as session:
            metrics = AnalyzeRequestMetrics()
            metrics.endpoint = endpoint
            metrics.request_received_at = request_received_at or datetime.datetime.now(
                datetime.timezone.utc
            )
            metrics.log_url = log_url
            session.add(metrics)
            session.flush()
            return metrics.id

    @classmethod
    def update(
        cls,
        id_: int,
        response_sent_at: datetime,
        response_length: int,
        response_certainty: float,
    ) -> None:
        """Update an AnalyzeRequestMetrics line
        with data related to the given response"""
        with transaction(commit=True) as session:
            metrics = session.query(AnalyzeRequestMetrics).filter_by(id=id_).first()
            metrics.response_sent_at = response_sent_at
            metrics.response_length = response_length
            metrics.response_certainty = response_certainty
            session.add(metrics)
