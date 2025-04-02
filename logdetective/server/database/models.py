import enum
import datetime

from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    Float,
    DateTime,
    String,
    Enum,
    func,
    select,
    distinct,
)

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

    @classmethod
    def _get_requests_by_time_for_postgres(
        cls, start_time, end_time, time_format, endpoint
    ):
        """func.to_char is PostgreSQL specific.
        Let's unit tests replace this function with the SQLite version.
        """
        if time_format == "%Y-%m-%d":
            pgsql_time_format = "YYYY-MM-DD"
        else:
            pgsql_time_format = "YYYY-MM-DD HH24"

        requests_by_time_format = (
            select(
                cls.id,
                func.to_char(cls.request_received_at, pgsql_time_format).label(
                    "time_format"
                ),
            )
            .filter(cls.request_received_at.between(start_time, end_time))
            .filter(cls.endpoint == endpoint)
            .cte("requests_by_time_format")
        )
        return requests_by_time_format

    @classmethod
    def _get_requests_by_time_for_sqllite(
        cls, start_time, end_time, time_format, endpoint
    ):
        """func.strftime is SQLite specific.
        Use this function in unit test using flexmock:

        flexmock(AnalyzeRequestMetrics).should_receive("_get_requests_by_time_for_postgres")
        .replace_with(AnalyzeRequestMetrics._get_requests_by_time_for_sqllite)
        """
        requests_by_time_format = (
            select(
                cls.id,
                func.strftime(time_format, cls.request_received_at).label(
                    "time_format"
                ),
            )
            .filter(cls.request_received_at.between(start_time, end_time))
            .filter(cls.endpoint == endpoint)
            .cte("requests_by_time_format")
        )
        return requests_by_time_format

    @classmethod
    def get_requests_in_period(
        cls,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        time_format: str,
        endpoint: Optional[EndpointType] = EndpointType.ANALYZE,
    ) -> dict[datetime.datetime, int]:
        """
        Get a dictionary with request counts grouped by time units within a specified period.

        Args:
            start_time (datetime): The start of the time period to query
            end_time (datetime): The end of the time period to query
            time_format (str): The strftime format string to format timestamps (e.g., '%Y-%m-%d')
            endpoint (EndpointType): The analyze API endpoint to query

        Returns:
            dict[datetime, int]: A dictionary mapping datetime objects to request counts
        """
        with transaction(commit=False) as session:
            requests_by_time_format = cls._get_requests_by_time_for_postgres(
                start_time, end_time, time_format, endpoint
            )

            count_requests_by_time_format = select(
                requests_by_time_format.c.time_format,
                func.count(distinct(requests_by_time_format.c.id)),  # pylint: disable=not-callable
            ).group_by("time_format")

            counts = session.execute(count_requests_by_time_format)
            results = counts.fetchall()

            # Convert results to a dictionary with proper datetime keys
            counts_dict = {
                datetime.datetime.strptime(r[0], time_format): r[1] for r in results
            }

            return counts_dict
