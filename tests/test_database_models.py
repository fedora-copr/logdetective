import datetime
from contextlib import contextmanager

import pytest
from flexmock import flexmock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from logdetective.server import database


@pytest.fixture(scope="function")
@contextmanager
def db():
    """Create an in-memory SQLite database for testing.
    Instead of depending on a PostgreSQL service in a separate container.
    Hopefully SqlAlchemy will manage all differences."""
    engine = create_engine("sqlite:///:memory:")
    SessionFactory = sessionmaker(autoflush=True, bind=engine)
    flexmock(database.base, engine=engine, SessionFactory=SessionFactory)
    database.init()
    yield SessionFactory
    database.base.destroy()


def test_create_and_update_AnalyzeRequestMetrics(db):
    with db as session_factory:
        metrics_id = database.model.AnalyzeRequestMetrics.create(
            endpoint=database.model.AnalyzeRequestMetrics.EndpointType.ANALYZE,
            log_url="https://example.com/logs/123",
        )
        assert metrics_id == 1
        database.model.AnalyzeRequestMetrics.update(
            id_=metrics_id,
            response_sent_at=datetime.datetime.now(datetime.timezone.utc),
            response_length=0,
            response_certainty=37.7,
        )

        metrics = (
            session_factory()
            .query(database.model.AnalyzeRequestMetrics)
            .filter_by(id=metrics_id)
            .first()
        )

        assert metrics is not None
        assert metrics.log_url == "https://example.com/logs/123"
        assert metrics.response_length == 0
        assert metrics.response_certainty == 37.7
