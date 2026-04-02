import ipaddress
import socket
from typing import List
from importlib.metadata import version

import aiohttp
from aiohttp.abc import ResolveResult
from fastapi import Request, HTTPException

from logdetective.utils import (
    ContentSizeCheck,
    check_content_size,
)
from logdetective.server.config import LOG, SERVER_CONFIG
from logdetective.server.exceptions import LogDetectiveConnectionError
from logdetective.remote_log import RemoteLog
from logdetective.exceptions import RemoteLogError
from logdetective.server.models import AnalysisRequest, ArtifactFile, RemoteArtifactFile


def connection_error_giveup(details: dict) -> None:
    """Too many connection errors, give up.
    """
    LOG.error("Too many connection errors, giving up. %s", details["exception"])
    raise LogDetectiveConnectionError() from details["exception"]


def should_we_giveup(exc: aiohttp.ClientResponseError) -> bool:
    """From backoff's docs:

    > a function which accepts the exception and returns
    > a truthy value if the exception should not be retried
    """
    LOG.info("Should we give up on retrying error %s", exc)
    return exc.status < 400


def we_give_up(details: dict):
    """Retries didn't work (or we got a different exc)
    we give up and raise proper 500 for our API endpoint
    """
    LOG.error("Last exception: %s", details["exception"])
    LOG.error("Inference error: %s", details["args"])
    raise HTTPException(500, "Request to the inference API failed")


async def get_artifacts_from_payload(
    payload: AnalysisRequest,
    http_session: aiohttp.ClientSession,
) -> dict[str, str]:
    """Retrieve artifact contents based on the type of artifact.
    Raise ValueError on unsupported element types."""
    build_artifacts: dict[str, str] = {}

    for artifact in payload.files:
        if isinstance(artifact, RemoteArtifactFile):
            LOG.info("Downloading artifact %s from %s", artifact.name, artifact.url)
            remote_log = RemoteLog(
                str(artifact.url),
                http_session,
                limit_bytes=SERVER_CONFIG.general.max_artifact_size
            )
            try:
                log_text = await remote_log.get_url_content()
            except RemoteLogError as ex:
                raise HTTPException(status_code=ex.status_code, detail=f"{ex}") from ex
            build_artifacts[artifact.name] = log_text
        elif isinstance(artifact, ArtifactFile):
            LOG.info("Handling artifact %s as raw string", artifact.name)
            build_artifacts[artifact.name] = artifact.content
        else:
            raise ValueError("Invalid element type ", type(artifact))

    total_payload_len = sum(len(content) for _, content in build_artifacts.items())
    LOG.info("Total artifact size from the obtained payload (in chars): %d", total_payload_len)
    return build_artifacts


def get_version() -> str:
    """Obtain the version number using importlib"""
    return version('logdetective')


def validate_request_size(request: Request) -> None:
    """
    FastAPI Depend function checking request's Content-Length before loading body into memory.

    Note:
        This function handles direct file submissions. For URL access, see `RemoteLog` class.
        In the case of URL requests, we limit the URL's content to 300 MiB.
        With the direct files raw log content, we limit the whole request size to 300Mib,
        so this fails even if all provided logs are under the limit, but exceed it together.

    Raises:
        HTTPException(411): If Content-Length header is missing or invalid
        HTTPException(413): If Content-Length exceeds maximum allowed size
    """
    size_check: ContentSizeCheck = check_content_size(
        request.headers,
        SERVER_CONFIG.general.max_artifact_size
    )
    if not (size_check.value_present and size_check.size_in_bytes is not None):
        raise HTTPException(status_code=411, detail="Content-Length is missing or invalid.")
    if not size_check.result:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Content-Length is too large: "
                f"{size_check.size_in_bytes} B "
                f"({size_check.size_in_bytes / (1024 * 1024):.2f} MiB) > "
                f"{SERVER_CONFIG.general.max_artifact_size} B "
                f"({SERVER_CONFIG.general.max_artifact_size / (1024 * 1024):.2f} MiB)"
            )
        )


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
            if (
                ip_address.is_private
            ):
                msg = (
                    f"Request to host: {host} port: {port} socket: "
                    f"{family} resolved to internal IP: {ip_address}."
                )
                LOG.error(msg=msg)
                raise socket.gaierror(socket.EAI_FAIL, msg)

        return ips
