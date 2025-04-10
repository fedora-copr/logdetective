from os import getenv
import inspect
from functools import wraps
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base

from logdetective import logger


def get_pg_url() -> str:
    """create postgresql connection string"""
    return (
        f"postgresql+psycopg2://{getenv('POSTGRESQL_USER')}"
        f":{getenv('POSTGRESQL_PASSWORD')}@{getenv('POSTGRESQL_HOST', 'postgres')}"
        f":{getenv('POSTGRESQL_PORT', '5432')}/{getenv('POSTGRESQL_DATABASE')}"
    )


# To log SQL statements, set SQLALCHEMY_ECHO env. var. to True|T|Yes|Y|1
sqlalchemy_echo = getenv("SQLALCHEMY_ECHO", "False").lower() in (
    "true",
    "t",
    "yes",
    "y",
    "1",
)
engine = create_engine(get_pg_url(), echo=sqlalchemy_echo)
SessionFactory = sessionmaker(autoflush=True, bind=engine)
Base = declarative_base()


@contextmanager
def transaction(commit: bool = False):
    """
    Context manager for 'framing' a db transaction.

    Args:
        commit: Whether to call `Session.commit()` upon exiting the context. Should be set to True
            if any changes are made within the context. Defaults to False.

    Raises:
        re-raise every exception catched inside the context manager and rolls back the transaction
    """

    session = SessionFactory()
    try:
        yield session
        if commit:
            session.commit()
    except Exception as ex:
        logger.warning("Exception while working with database: %s", str(ex))
        session.rollback()
        raise
    finally:
        session.close()


def retry_db_operations():
    """
    Decorator for retrying a failing list of db operations
    wrapped inside the trasaction context manager.
    """
    DB_OPERATIONS_RETRIES = 3

    def decorator(f):
        @wraps(f)
        async def async_decorated_function(*args, **kwargs):
            i = 0
            while i < DB_OPERATIONS_RETRIES:
                try:
                    response = await f(*args, **kwargs)
                    return response
                except OperationalError as e:
                    i += 1
            raise e

        @wraps(f)
        def sync_decorated_function(*args, **kwargs):
            i = 0
            while i < DB_OPERATIONS_RETRIES:
                try:
                    response = f(*args, **kwargs)
                    return response
                except OperationalError as e:
                    i += 1
            raise e

        if inspect.iscoroutinefunction(f):
            return async_decorated_function
        return sync_decorated_function

    return decorator

def init():
    """Init db"""
    Base.metadata.create_all(engine)
    logger.debug("Database initialized")


def destroy():
    """Destroy db"""
    Base.metadata.drop_all(engine)
    logger.warning("Database cleaned")
