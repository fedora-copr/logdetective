from os import getenv
from contextlib import contextmanager
from sqlalchemy import create_engine
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


def init():
    """Init db"""
    Base.metadata.create_all(engine)
    logger.debug("Database initialized")


def destroy():
    """Destroy db"""
    Base.metadata.drop_all(engine)
    logger.warning("Database cleaned")


DB_MAX_RETRIES = 3  # How many times retry a db operation
