import ipaddress
import socket
from typing import List
from importlib.metadata import version

import aiohttp
from aiohttp.abc import ResolveResult
from sqlalchemy.exc import OperationalError
from tenacity import (
    retry,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from logdetective.server.config import LOG
from logdetective.server.database.base import DB_MAX_RETRIES
from logdetective.server.exceptions import LogDetectiveConnectionError


retry_database_error = retry(
    stop=stop_after_attempt(DB_MAX_RETRIES),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception_type(OperationalError),
    reraise=True,
)


def connection_error_giveup(retry_state: RetryCallState) -> None:
    """Too many connection errors, give up."""
    exception = retry_state.outcome.exception()
    LOG.error(
        "Too many connection errors, giving up. %s",
        exception,
    )
    raise LogDetectiveConnectionError() from exception


def inference_retry_backoff(retry_state: RetryCallState) -> None:
    """Log when an LLM inference call is being retried."""
    exception = retry_state.outcome.exception()
    LOG.warning(
        "LLM inference retry %d after %s (%.1fs elapsed): %s",
        retry_state.attempt_number,
        exception.__class__.__name__,
        retry_state.seconds_since_start,
        exception,
    )


def inference_retry_giveup(retry_state: RetryCallState) -> None:
    """Log when all LLM inference retries are exhausted."""
    exception = retry_state.outcome.exception()
    LOG.error(
        "LLM inference failed after %d retries (%.1fs elapsed): %s",
        retry_state.attempt_number,
        retry_state.seconds_since_start,
        exception,
    )
    raise exception


def get_version() -> str:
    """Obtain the version number using importlib"""
    return version("logdetective")


class SSRFProtectedResolver(aiohttp.ThreadedResolver):
    """Resolver raising exception if URL evaluates to local address."""

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> List[ResolveResult]:
        """Resolve IP for given hostname, raise exception if the IP is local."""
        ips = await super().resolve(host, port, family)

        for resolved_ip in ips:
            try:
                ip_address = ipaddress.ip_address(resolved_ip["host"])
            except ValueError as ex:
                raise socket.gaierror(socket.EAI_FAIL) from ex
            if ip_address.is_private:
                msg = (
                    f"Request to host: {host} port: {port} socket: "
                    f"{family} resolved to internal IP: {ip_address}."
                )
                LOG.error(msg=msg)
                raise socket.gaierror(socket.EAI_FAIL, msg)

        return ips
