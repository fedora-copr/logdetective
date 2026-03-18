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
from logdetective.server.config import LOG, SERVER_CONFIG, GENERIC_LOG_NAME
from logdetective.server.exceptions import LogDetectiveConnectionError
from logdetective.remote_log import RemoteLog
from logdetective.exceptions import RemoteLogError
from logdetective.server.models import (
    AnalyzedSnippet,
    RatedSnippetAnalysis,
    BuildLogRequest,
)


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


def select_relevance(snippet: AnalyzedSnippet) -> float:
    """Retrieve relevance value from structure, if there is one."""
    if not isinstance(snippet.explanation, RatedSnippetAnalysis):
        LOG.exception("Only rated snippets can be ordered by relevance.")
        raise ValueError
    return snippet.explanation.relevance


def select_line_number(explanation: AnalyzedSnippet) -> int:
    """Returns line number of original snippet."""
    return explanation.line_number


def filter_snippets(
    processed_snippets: List[AnalyzedSnippet], top_k: int
) -> List[AnalyzedSnippet]:
    """Filter snippets according to criteria in config while keeping them ordered by line number.
    If all snippets recieved the same score, return them all.
    AnalyzedSnippet objects must have `explanation` attribute set to `RatedSnippetAnalysis`,
    otherwise raise `ValueError`."""

    if top_k >= len(processed_snippets):
        LOG.warning(
            "The `top-k` parameter >= number of original snippets, skipping filtering."
        )
        return processed_snippets

    # Sorting invokes `select_relevance` which also tests if objects actually
    # have the score assigned. Otherwise it raises exception.
    processed_snippets = sorted(processed_snippets, key=select_relevance, reverse=True)

    # Check for failure mode when all snippets have
    # the same relevance. In such cases there is no point in filtering
    # and all snippets are returned.
    max_relevance = processed_snippets[0].explanation.relevance
    min_relevance = processed_snippets[-1].explanation.relevance

    LOG.info(
        "Analyzed snippets sorted. Max relevance: %d Min relevance: %e",
        max_relevance,
        min_relevance,
    )
    if max_relevance == min_relevance:
        LOG.warning("All snippets recieved the same rating. Filtering disabled.")
        return processed_snippets

    processed_snippets = processed_snippets[:top_k]

    # Re-sorting snippets by line number
    processed_snippets = sorted(processed_snippets, key=select_line_number)

    return processed_snippets


async def get_artifacts_from_payload(
    payload: BuildLogRequest,
    http_session: aiohttp.ClientSession,
) -> dict[str, str]:
    """Retrieve artifact content based on the type of request: URL or raw string."""
    build_artifacts: dict[str, str] = {}
    if payload.url:
        LOG.debug("Handling log as URL")
        remote_log = RemoteLog(
            payload.url,
            http_session,
            limit_bytes=SERVER_CONFIG.general.max_artifact_size
        )
        try:
            log_text = await remote_log.get_url_content()
        except RemoteLogError as ex:
            raise HTTPException(status_code=ex.status_code, detail=f"{ex}") from ex
        build_artifacts = {GENERIC_LOG_NAME: log_text}
    elif payload.files:
        # pydantic field validators make sure at least one element is present,
        # and logs are not over the maximum log size
        LOG.info("Handling log as raw string")
        build_artifacts = {file.name: file.content for file in payload.files}

    total_payload_len = sum(len(content) for _, content in build_artifacts.items())
    LOG.info("Total artifact size from the obtained payload (in chars): %d", total_payload_len)
    return build_artifacts


def construct_final_prompt(formatted_snippets: str, prompt_template: str) -> str:
    """Create final prompt from processed snippets and csgrep output, if it is available."""

    final_prompt = prompt_template.format(formatted_snippets)
    return final_prompt


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
