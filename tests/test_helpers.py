import datetime
from contextlib import contextmanager
from typing import Generator, Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flexmock import flexmock

from logdetective.server.models import Response, Explanation
from logdetective.server.database import base
from logdetective.server.database.base import init, destroy
from logdetective.server.database.models import AnalyzeRequestMetrics, EndpointType
from logdetective.server.compressors import LLMResponseCompressor, RemoteLogCompressor


class DatabaseFactory:  # pylint: disable=too-few-public-methods
    @staticmethod
    def get_pg_test_url() -> str:
        """Create PostgreSql connection string to a testing db.
        Database container is started by `tox -e pytest` command,
        connection details for the container are specified in tox.ini"""

        return "postgresql+psycopg2://user:password@localhost:5432/test_db"

    def __init__(self):
        """Connect to a postgres container for testing purposes."""
        self.engine = create_engine(self.get_pg_test_url())
        self.SessionFactory = sessionmaker(autoflush=True, bind=self.engine)
        flexmock(base, engine=self.engine, SessionFactory=self.SessionFactory)

    @contextmanager
    def make_new_db(self):
        try:
            init()
            yield self.SessionFactory
        finally:
            destroy()


class PopulateDatabase:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.db_factory = DatabaseFactory()

    @contextmanager
    def populate_db_at_regular_intervals(  # pylint: disable=too-many-positional-arguments
        self,
        interval: datetime.timedelta = datetime.timedelta(minutes=15),
        duration: datetime.timedelta = datetime.timedelta(hours=23),
        end_time: Optional[datetime.datetime] = None,
        endpoint_type: Optional[EndpointType] = EndpointType.ANALYZE,
    ) -> Generator:
        with self.db_factory.make_new_db() as session_factory:  # pylint: disable=contextmanager-generator-missing-cleanup
            end_time = end_time or datetime.datetime.now(datetime.timezone.utc)
            start_time = end_time - duration

            current_time = start_time
            increasing_response_time = 1
            increasing_response_length = 500
            while current_time < end_time:
                id_ = AnalyzeRequestMetrics.create(
                    endpoint=endpoint_type,
                    compressed_log=RemoteLogCompressor.zip_text(
                        "Some log for a failed build"
                    ),
                    request_received_at=current_time,
                )
                response_time = current_time + datetime.timedelta(
                    seconds=increasing_response_time
                )
                increasing_response_length += 500
                AnalyzeRequestMetrics.update(
                    id_=id_,
                    response_sent_at=response_time,
                    response_length=increasing_response_length,
                    response_certainty=70,
                    compressed_response=LLMResponseCompressor(
                        Response(
                            explanation=Explanation(text="a small error", logprobs=[]),
                            response_certainty=0.1,
                        )
                    ).zip_response(),
                )
                current_time += interval
                increasing_response_time += 1
                increasing_response_time = increasing_response_time % 4
                increasing_response_length = increasing_response_length % 5000

            yield session_factory

    @classmethod
    @contextmanager
    def populate_db(cls, duration=datetime.timedelta, endpoint=EndpointType):
        """Populate the db, one request every 15 minutes.
        and responses increasing for 1 hour, and then back to 1.
        For the last duration time.
        Mock the model class so that a SQLite query can be performed,
        the db used for the tests runs in a SQLite db.
        """
        with cls().populate_db_at_regular_intervals(
            duration=duration,
            endpoint_type=endpoint,
        ) as session_factory:
            yield session_factory


@pytest.fixture(scope="function")
@contextmanager
def populate_db_with_analyze_request_every_15_minutes_for_15_hours():
    duration = datetime.timedelta(hours=15)
    with PopulateDatabase().populate_db_at_regular_intervals(
        duration=duration
    ) as session_factory:
        yield session_factory


@pytest.fixture(scope="function")
@contextmanager
def populate_db_with_analyze_request_every_15_minutes_for_9_days():
    duration = datetime.timedelta(days=9)
    with PopulateDatabase().populate_db_at_regular_intervals(
        duration=duration
    ) as session_factory:
        yield session_factory


@pytest.fixture(scope="function")
@contextmanager
def populate_db_with_analyze_request_every_15_minutes_for_3_weeks():
    duration = datetime.timedelta(weeks=3)
    with PopulateDatabase().populate_db_at_regular_intervals(
        duration=duration
    ) as session_factory:
        yield session_factory
