import aiohttp

from logdetective.reactor.config import get_config
from logdetective.reactor.logging import get_log
from logdetective.server.models import StagedResponse
from logdetective.reactor.dependencies import HttpConnections

LOG = get_log()
REACTOR_CONFIG = get_config()


async def submit_to_log_detective(http_connections: HttpConnections, log_url: str) -> StagedResponse:
    """Submit the log URL to the staged endpoint of Log Detective and
    retrieve the results.
    """
    try:
        async with http_connections.logdetective.post(
            "/analyze/staged", json={"url": log_url}
        ) as resp:
            return await resp.json()
    except aiohttp.client_exceptions.ClientError as e:
        LOG.exception(e)
        raise
