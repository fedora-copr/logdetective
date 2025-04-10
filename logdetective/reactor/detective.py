import asyncio
import aiohttp

from fastapi import FastAPI

from logdetective.reactor.config import get_config
from logdetective.reactor.logging import get_log
from logdetective.server.models import StagedResponse

LOG = get_log()
REACTOR_CONFIG = get_config()


async def submit_to_log_detective(app: FastAPI, log_url: str) -> StagedResponse:
    """Submit the log URL to the staged endpoint of Log Detective and
    retrieve the results.
    """
    try:
        async with app.logdetective_http.post(
            "/analyze/staged", json={"url": log_url}
        ) as resp:
            return await resp.json()
    except aiohttp.client_exceptions.ClientError as e:
        LOG.exception(e)
        raise
