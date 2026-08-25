import ipaddress
import re
import socket
from collections.abc import Mapping
from importlib.metadata import version
from typing import (
    List,
    Tuple,
    NamedTuple,
)

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

from logdetective.config import LOG
from logdetective.database.base import DB_MAX_RETRIES
from logdetective.exceptions import LogDetectiveConnectionError


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


SANITIZE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (  # Emails
        # we don't want to match invalid subdomains, starting/ending with - or .
        # such as @-domain.com or @domain-.com or @.domain.com or @domain..com
        re.compile(
            (
                r"\b[\w.%+-]+"  # username
                r"(?:@|\(at\)|\[at\])"  # "at" symbol
                r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,10}"  # subdomains
                r"[a-z]{2,4}\b"  # top level domain
            ),
            re.IGNORECASE,
        ),
        "copr-team@redhat.com",
    ),
    (  # GPG fingerprints
        re.compile(
            r"\bFingerprint:\s*([0-9A-F]{32,64}|(?:\s*[0-9A-F]{4}){8,16})\b",
            re.IGNORECASE,
        ),
        f"Fingerprint:{' FFFF' * 10}",
    ),
    (  # RSA keys, sometimes they are as short as 16 hexa characters
        re.compile(r"\bRSA\s+key\s+[0-9A-F]{16,512}\b", re.IGNORECASE),
        f"RSA key {'FFFF' * 10}",
    ),
    (  # Pubkeys, sometimes pubkey-deadbeef-01234567, or pubkey-40hexacharacters-other8
        re.compile(r"\bpubkey-[0-9A-F]{8}[0-9A-F-]{8,128}\b", re.IGNORECASE),
        f"pubkey-{'ffff' * 10}",
    ),
]


def sanitize_artifact(log: str) -> str:
    """Redact personal identifiers from the artifact content before it is sent to the LLM.

    Redaction is done by replacing emails, and various public keys/signatures.
    """
    for pattern, replacement in SANITIZE_PATTERNS:
        log = re.sub(pattern, replacement, log)
    return log


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


class ContentSizeCheck(NamedTuple):
    """
    Aggregate requests content-size info for checks.

    Args:
        proceed: True if download should proceed; False to reject.
        size_in_bytes: Parsed Content-Length value, or None if absent or invalid.
    """

    proceed: bool
    size_in_bytes: int | None


def check_content_size(
    headers: Mapping,
    size_limit: int,
    require_header: bool = True,
) -> ContentSizeCheck:
    """
    Validate that a request's content size doesn't exceed a maximum based on headers.

    Args:
        headers: HTTP headers
        size_limit: Maximum allowed size in bytes
        require_header:
            True(defualt): request with no Content-Length header is rejected.
            False: request with an absent header is allowed (limit is then checked while reading).

    Returns:
        ContentSizeCheck.`.proceed=True` means safe to download. `.proceed=False` means reject.
        `.size_in_bytes` is None when the header is absent or unparseable.
    """
    content_length = headers.get("content-length")

    if content_length is None:
        transfer_encoding = headers.get("transfer-encoding", "None")
        LOG.info(
            "No Content-Length header. Transfer-Encoding: %s",
            transfer_encoding,
        )
        return ContentSizeCheck(proceed=not require_header, size_in_bytes=None)

    try:
        size = int(content_length)
    except (ValueError, TypeError):
        LOG.error("Invalid Content-Length header value: %s", content_length)
        return ContentSizeCheck(proceed=False, size_in_bytes=None)

    is_valid = size <= size_limit
    if not is_valid:
        LOG.warning(
            "Content-Length: %d B (%.2f MiB) exceeds max %d B (%.2f MiB)",
            size,
            size / (1024 * 1024),
            size_limit,
            size_limit / (1024 * 1024),
        )
    return ContentSizeCheck(proceed=is_valid, size_in_bytes=size)
