from os import getenv
from contextlib import asynccontextmanager
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from logdetective import logger


def get_pg_url() -> str:
    """create postgresql connection string"""
    return (
        f"postgresql+asyncpg://{getenv('POSTGRESQL_USER')}"
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
engine = create_async_engine(get_pg_url(), echo=sqlalchemy_echo)
SessionFactory = async_sessionmaker(autoflush=True, bind=engine)  # pylint: disable=invalid-name


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


@asynccontextmanager
async def transaction(commit: bool = False):
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


async def init():
    """Init db"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.debug("Database initialized")


async def destroy():
    """Destroy db"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("Database cleaned")


DB_MAX_RETRIES = 3  # How many times retry a db operation
