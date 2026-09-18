from os import getenv
import enum
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from psycopg.conninfo import make_conninfo
from sqlalchemy import text, URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from logdetective import logger


def get_pg_url() -> URL:
    """Create an escaped SQLAlchemy psycopg URL from database settings."""
    return URL.create(
        "postgresql+psycopg",
        username=getenv("POSTGRESQL_USER"),
        password=getenv("POSTGRESQL_PASSWORD"),
        host=getenv("POSTGRESQL_HOST", "postgres"),
        port=int(getenv("POSTGRESQL_PORT", "5432")),
        database=getenv("POSTGRESQL_DATABASE"),
    )


def get_pg_conninfo() -> str:
    """Build a libpq connection string from the database environment variables.

    Returns:
        An escaped connection string suitable for Procrastinate and psycopg.

    Raises:
        ProgrammingError: If an environment variable cannot be represented in a
            libpq connection string.
    """
    return make_conninfo(
        user=getenv("POSTGRESQL_USER"),
        password=getenv("POSTGRESQL_PASSWORD"),
        host=getenv("POSTGRESQL_HOST", "postgres"),
        port=getenv("POSTGRESQL_PORT", "5432"),
        dbname=getenv("POSTGRESQL_DATABASE"),
    )


# To log SQL statements, set SQLALCHEMY_ECHO env. var. to True|T|Yes|Y|1
sqlalchemy_echo = getenv("SQLALCHEMY_ECHO", "False").lower() in (
    "true",
    "t",
    "yes",
    "y",
    "1",
)
engine = create_async_engine(get_pg_url(), echo=sqlalchemy_echo, pool_pre_ping=True)
SessionFactory = async_sessionmaker(autoflush=True, expire_on_commit=False, bind=engine)  # pylint: disable=invalid-name


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


def enum_values(enum_type: type[enum.Enum]) -> list[str]:
    """Return string values for a SQLAlchemy enum's persisted labels.

    SQLAlchemy persists Python enum member names by default. Database labels in
    Log Detective instead follow the lowercase values exposed by the public API.
    """
    return [str(member.value) for member in enum_type]


@asynccontextmanager
async def transaction(commit: bool = False) -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for 'framing' a db transaction.

    Args:
        commit: Whether to call `Session.commit()` upon exiting the context. Should be set to True
            if any changes are made within the context. Defaults to False.
    """

    session = SessionFactory()
    async with session:
        try:
            yield session
            if commit:
                await session.commit()
        except Exception as ex:
            logger.warning("Exception while working with database: %s", str(ex))
            await session.rollback()
            raise
        finally:
            await session.close()


async def check() -> None:
    """Check database"""
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
        logger.debug("Database checked")


async def destroy() -> None:
    """Destroy db"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("Database cleaned")


DB_MAX_RETRIES = 3  # How many times retry a db operation

retry_database_error = retry(
    stop=stop_after_attempt(DB_MAX_RETRIES),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception_type(OperationalError),
    reraise=True,
)
