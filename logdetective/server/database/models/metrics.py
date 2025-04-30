import io
import enum
import datetime
from typing import Optional, List

import backoff

from sqlalchemy import (
    Column,
    Integer,
    Float,
    DateTime,
    Enum,
    func,
    select,
    distinct,
    ForeignKey,
    LargeBinary,
)
from sqlalchemy.orm import relationship, aliased
from sqlalchemy.exc import OperationalError

from logdetective.server.database.base import Base, transaction, DB_MAX_RETRIES
from logdetective.server.database.models.merge_request_jobs import (
    GitlabMergeRequestJobs,
    Forge,
)


class EndpointType(enum.Enum):
    """Different analyze endpoints"""

    ANALYZE = "analyze_log"
    ANALYZE_STAGED = "analyze_log_staged"
    ANALYZE_STREAM = "analyze_log_stream"
    ANALYZE_GITLAB_JOB = "analyze_gitlab_job"


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
    compressed_log = Column(
        LargeBinary(length=314572800),  # 300MB limit (300 * 1024 * 1024)
        nullable=False,
        index=False,
        comment="Log processed, saved in a zip format",
    )
    compressed_response = Column(
        LargeBinary(length=314572800),  # 300MB limit (300 * 1024 * 1024)
        nullable=True,
        index=False,
        comment="Given response (with explanation and snippets) saved in a zip format",
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

    merge_request_job_id = Column(
        Integer,
        ForeignKey("gitlab_merge_request_jobs.id"),
        nullable=True,
        index=False,
        comment="Is this an analyze request coming from a merge request?",
    )

    mr_job = relationship("GitlabMergeRequestJobs", back_populates="request_metrics")

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def create(
        cls,
        endpoint: EndpointType,
        compressed_log: io.BytesIO,
        request_received_at: Optional[datetime.datetime] = None,
    ) -> int:
        """Create AnalyzeRequestMetrics new line
        with data related to a received request"""
        with transaction(commit=True) as session:
            metrics = AnalyzeRequestMetrics()
            metrics.endpoint = endpoint
            metrics.compressed_log = compressed_log
            metrics.request_received_at = request_received_at or datetime.datetime.now(
                datetime.timezone.utc
            )
            session.add(metrics)
            session.flush()
            return metrics.id

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def update(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
        cls,
        id_: int,
        response_sent_at: datetime,
        response_length: int,
        response_certainty: float,
        compressed_response: bytes,
    ) -> None:
        """Update a row
        with data related to the given response"""
        with transaction(commit=True) as session:
            metrics = session.query(AnalyzeRequestMetrics).filter_by(id=id_).first()
            metrics.response_sent_at = response_sent_at
            metrics.response_length = response_length
            metrics.response_certainty = response_certainty
            metrics.compressed_response = compressed_response
            session.add(metrics)

    @classmethod
    @backoff.on_exception(backoff.expo, OperationalError, max_tries=DB_MAX_RETRIES)
    def get_metric_by_id(
        cls,
        id_: int,
    ) -> "AnalyzeRequestMetrics":
        """Update a row
        with data related to the given response"""
        with transaction(commit=True) as session:
            metric = session.query(AnalyzeRequestMetrics).filter_by(id=id_).first()
            return metric

    def add_mr_job(
        self,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
    ) -> None:
        """This request was triggered by a merge request job.
        Link it.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
        """
        mr_job = GitlabMergeRequestJobs.get_or_create(forge, project_id, mr_iid, job_id)
        self.merge_request_job_id = mr_job.id
        with transaction(commit=True) as session:
            session.merge(self)

    @classmethod
    def get_requests_metrics_for_mr_job(
        cls,
        forge: Forge,
        project_id: int,
        mr_iid: int,
        job_id: int,
    ) -> List["AnalyzeRequestMetrics"]:
        """Search for all requests triggered by the specified merge request job.

        Args:
          forge: forge name
          project_id: forge project id
          mr_iid: merge request forge iid
          job_id: forge job id
        """
        with transaction(commit=False) as session:
            mr_job_alias = aliased(GitlabMergeRequestJobs)
            metrics = (
                session.query(cls)
                .join(mr_job_alias, cls.merge_request_job_id == mr_job_alias.id)
                .filter(
                    mr_job_alias.forge == forge,
                    mr_job_alias.mr_iid == mr_iid,
                    mr_job_alias.project_id == project_id,
                    mr_job_alias.job_id == job_id,
                )
                .all()
            )
            return metrics

    @classmethod
    def get_postgres_time_format(cls, time_format):
        """Map python time format in the PostgreSQL format."""
        if time_format == "%Y-%m-%d":
            pgsql_time_format = "YYYY-MM-DD"
        else:
            pgsql_time_format = "YYYY-MM-DD HH24"
        return pgsql_time_format

    @classmethod
    def get_dictionary_with_datetime_keys(
        cls, time_format: str, values_dict: dict[str, any]
    ) -> dict[datetime.datetime, any]:
        """Convert from a dictionary with str keys to a dictionary with datetime keys"""
        new_dict = {
            datetime.datetime.strptime(r[0], time_format): r[1] for r in values_dict
        }
        return new_dict

    @classmethod
    def _get_requests_by_time_for_postgres(
        cls, start_time, end_time, time_format, endpoint
    ):
        """Get total requests number in time period.

        func.to_char is PostgreSQL specific.
        Let's unit tests replace this function with the SQLite version.
        """
        pgsql_time_format = cls.get_postgres_time_format(time_format)

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

            return cls.get_dictionary_with_datetime_keys(time_format, results)

    @classmethod
    def _get_average_responses_times_for_postgres(
        cls, start_time, end_time, time_format, endpoint
    ):
        """Get average responses time.

        func.to_char is PostgreSQL specific.
        Let's unit tests replace this function with the SQLite version.
        """
        with transaction(commit=False) as session:
            pgsql_time_format = cls.get_postgres_time_format(time_format)

            average_responses_times = (
                select(
                    func.to_char(cls.request_received_at, pgsql_time_format).label(
                        "time_range"
                    ),
                    (
                        func.avg(
                            func.extract(  # pylint: disable=not-callable
                                "epoch", cls.response_sent_at - cls.request_received_at
                            )
                        )
                    ).label("average_response_seconds"),
                )
                .filter(cls.request_received_at.between(start_time, end_time))
                .filter(cls.endpoint == endpoint)
                .group_by("time_range")
                .order_by("time_range")
            )

            results = session.execute(average_responses_times).fetchall()
            return results

    @classmethod
    def get_responses_average_time_in_period(
        cls,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        time_format: str,
        endpoint: Optional[EndpointType] = EndpointType.ANALYZE,
    ) -> dict[datetime.datetime, int]:
        """
        Get a dictionary with average responses times
        grouped by time units within a specified period.

        Args:
            start_time (datetime): The start of the time period to query
            end_time (datetime): The end of the time period to query
            time_format (str): The strftime format string to format timestamps (e.g., '%Y-%m-%d')
            endpoint (EndpointType): The analyze API endpoint to query

        Returns:
            dict[datetime, int]: A dictionary mapping datetime objects
            to average responses times
        """
        with transaction(commit=False) as _:
            average_responses_times = cls._get_average_responses_times_for_postgres(
                start_time, end_time, time_format, endpoint
            )

            return cls.get_dictionary_with_datetime_keys(
                time_format, average_responses_times
            )

    @classmethod
    def _get_average_responses_lengths_for_postgres(
        cls, start_time, end_time, time_format, endpoint
    ):
        """Get average responses length.

        func.to_char is PostgreSQL specific.
        Let's unit tests replace this function with the SQLite version.
        """
        with transaction(commit=False) as session:
            pgsql_time_format = cls.get_postgres_time_format(time_format)

            average_responses_lengths = (
                select(
                    func.to_char(cls.request_received_at, pgsql_time_format).label(
                        "time_range"
                    ),
                    (func.avg(cls.response_length)).label("average_responses_length"),
                )
                .filter(cls.request_received_at.between(start_time, end_time))
                .filter(cls.endpoint == endpoint)
                .group_by("time_range")
                .order_by("time_range")
            )

            results = session.execute(average_responses_lengths).fetchall()
            return results

    @classmethod
    def get_responses_average_length_in_period(
        cls,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        time_format: str,
        endpoint: Optional[EndpointType] = EndpointType.ANALYZE,
    ) -> dict[datetime.datetime, int]:
        """
        Get a dictionary with average responses length
        grouped by time units within a specified period.

        Args:
            start_time (datetime): The start of the time period to query
            end_time (datetime): The end of the time period to query
            time_format (str): The strftime format string to format timestamps (e.g., '%Y-%m-%d')
            endpoint (EndpointType): The analyze API endpoint to query

        Returns:
            dict[datetime, int]: A dictionary mapping datetime objects
            to average responses lengths
        """
        with transaction(commit=False) as _:
            average_responses_lengths = cls._get_average_responses_lengths_for_postgres(
                start_time, end_time, time_format, endpoint
            )

            return cls.get_dictionary_with_datetime_keys(
                time_format, average_responses_lengths
            )
