import asyncio
import ipaddress
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from typing import (
    List,
    ParamSpec,
    Tuple,
    NamedTuple,
    TypeVar,
)

import aiohttp
from aiohttp.abc import ResolveResult
from aiohttp.helpers import is_ip_address
from pydantic import ValidationError
import sentry_sdk
import yaml
from tenacity import (
    RetryCallState,
)

from logdetective.config import LOG, SERVER_CONFIG
from logdetective.exceptions import LogDetectiveConnectionError
from logdetective.models import (
    APITokens,
    DEFAULT_SSRF_BLOCKED_CLUSTER_CIDRS,
    normalize_ssrf_hostname,
)
from logdetective.yaml_utils import DuplicateKeySafeLoader

P = ParamSpec("P")
T = TypeVar("T")
SSRFNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)


def init_sentry() -> None:
    """Enable Sentry for this process when the shared configuration has a DSN."""
    if sentry_dsn := SERVER_CONFIG.general.sentry_dsn:
        sentry_sdk.init(dsn=str(sentry_dsn), traces_sample_rate=1.0)


async def run_blocking(function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run blocking work without abandoning its thread during cancellation.

    Python cannot safely stop a running thread. If the caller is cancelled, this
    helper waits for the blocking call to return before propagating cancellation.
    This makes cancellation eventual while keeping the work owned by the task.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancellation:
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                if current is not None:
                    current.uncancel()
        try:
            worker.result()
        except Exception:  # pylint: disable=broad-exception-caught
            LOG.exception("Blocking operation failed while cancellation was pending")
        raise cancellation


def load_api_tokens(path: str | None) -> APITokens | None:
    """Load and validate named API bearer tokens from a YAML file."""
    if not path:
        LOG.warning(
            "Log Detective launched without set API tokens. "
            "All requests will be accepted without authentication!"
        )
        return None

    with open(path, encoding="utf-8") as token_file:
        token_data = yaml.load(token_file, Loader=DuplicateKeySafeLoader)

    try:
        return APITokens.model_validate(token_data)
    except ValidationError as exc:
        messages = "; ".join(
            error["msg"] for error in exc.errors(include_input=False)
        )
        raise ValueError(f"Invalid API token file: {messages}") from None


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


EMAIL_AT_PATTERN = re.compile(r"@|\(at\)|\[at\]", re.IGNORECASE)
EMAIL_DOMAIN_CANDIDATE = re.compile(r"[a-z0-9._-]+", re.IGNORECASE)
EMAIL_REPLACEMENT = "copr-team@redhat.com"
# These suffixes identify systemd units in logs, even when the name contains @.
SYSTEMD_UNIT_SUFFIXES = (
    ".automount",
    ".device",
    ".mount",
    ".path",
    ".scope",
    ".service",
    ".slice",
    ".socket",
    ".swap",
    ".target",
    ".timer",
)

SANITIZE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (  # GPG fingerprints
        re.compile(
            r"\bFingerprint:\s*+([0-9A-F]{32,64}|(?:\s*[0-9A-F]{4}){8,16})\b",
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


def _is_email_word_char(char: str) -> bool:
    """Match the Unicode word characters used by the original ``\b`` rule."""
    return char.isalnum() or char == "_"


def _is_email_local_char(char: str) -> bool:
    """Match the original local-part character class without regex backtracking."""
    return _is_email_word_char(char) or char in ".%+-"


def _is_email_boundary(log: str, position: int) -> bool:
    """Match ``\b`` at the start of a candidate local part."""
    previous_is_word = position > 0 and _is_email_word_char(log[position - 1])
    return previous_is_word != _is_email_word_char(log[position])


def _email_domain_end(log: str, position: int) -> int | None:
    """Accept dotted malformed domains without consuming ordinary log identifiers.

    Numeric hosts, systemd unit names, and journal names also appear after
    ``@`` in build logs; they are not treated as email domains. A unit suffix
    exempts the whole candidate, including names with additional dots.
    """
    candidate = EMAIL_DOMAIN_CANDIDATE.match(log, position)
    if candidate is None:
        return None
    value = candidate.group().rstrip(".")
    if "." not in value or not any(char.isalpha() for char in value):
        return None
    if value.lower().endswith((*SYSTEMD_UNIT_SUFFIXES, ".journal")):
        return None
    return position + len(value)


def _redact_emails(log: str) -> str:
    """Find email separators and inspect each adjacent local part only once.

    Searching for a local part from every word boundary retries long failed
    candidates at each character. Separator-centered scanning avoids that cost
    while retaining the original left boundary and replacement rules.
    """
    pieces = []
    previous_end = 0
    for separator in EMAIL_AT_PATTERN.finditer(log):
        if separator.start() < previous_end:
            continue

        local_start = separator.start()
        while local_start > previous_end and _is_email_local_char(log[local_start - 1]):
            local_start -= 1
        while local_start < separator.start() and not _is_email_boundary(log, local_start):
            local_start += 1
        if local_start == separator.start():
            continue

        domain_end = _email_domain_end(log, separator.end())
        if domain_end is None:
            continue
        pieces.extend((log[previous_end:local_start], EMAIL_REPLACEMENT))
        previous_end = domain_end

    if not pieces:
        return log
    pieces.append(log[previous_end:])
    return "".join(pieces)


def sanitize_artifact(log: str) -> str:
    """Redact personal identifiers from the artifact content before it is sent to the LLM.

    Redaction replaces emails and various public keys/signatures. Email
    separators are scanned directly; the remaining patterns avoid repartitioning
    whitespace when a fingerprint candidate fails.
    """
    log = _redact_emails(log)
    for pattern, replacement in SANITIZE_PATTERNS:
        log = pattern.sub(replacement, log)
    return log


def get_version() -> str:
    """Obtain the version number using importlib"""
    return version("logdetective")


class SSRFProtectedResolver(aiohttp.ThreadedResolver):
    """Reject unsafe addresses in every DNS answer before connecting."""

    def __init__(
        self,
        allowed_hostnames: list[str] | tuple[str, ...] = (),
        blocked_cluster_cidrs: list[SSRFNetwork] | tuple[str, ...] | None = None,
    ) -> None:
        """Normalize trusted names and blocked ranges for later address checks.

        ``None`` selects the protective defaults; an explicit empty CIDR list
        replaces them with no additional deployment-specific blocked ranges.
        """
        super().__init__()
        self._allowed_hostnames = frozenset(
            normalize_ssrf_hostname(hostname) for hostname in allowed_hostnames
        )
        self._blocked_cluster_cidrs = tuple(
            ipaddress.ip_network(cidr)
            for cidr in (
                DEFAULT_SSRF_BLOCKED_CLUSTER_CIDRS
                if blocked_cluster_cidrs is None
                else blocked_cluster_cidrs
            )
        )

    def check_address(
        self, address_text: str, host: str, port: int, family: socket.AddressFamily
    ) -> None:
        """Apply the same policy to DNS answers and literal URL addresses."""
        try:
            ip_address = ipaddress.ip_address(address_text)
        except ValueError as ex:
            raise socket.gaierror(socket.EAI_FAIL) from ex
        try:
            allowed = normalize_ssrf_hostname(host) in self._allowed_hostnames
        except ValueError:
            allowed = False
        # IPv4-mapped IPv6 addresses must obey IPv4 CIDRs and classifications.
        address = getattr(ip_address, "ipv4_mapped", None) or ip_address
        blocked_cluster = any(
            address in network for network in self._blocked_cluster_cidrs
            if address.version == network.version
        )
        private_network = any(
            address in network for network in _PRIVATE_NETWORKS
            if address.version == network.version
        )
        # Only ordinary RFC1918/ULA addresses may use the hostname exception.
        # Other non-global addresses include loopback, link-local, metadata,
        # unspecified, reserved, multicast and shared address space.
        special = any(
            (
                address.is_multicast,
                address.is_reserved,
                address.is_unspecified,
                address.is_loopback,
                address.is_link_local,
            )
        )
        blocked = blocked_cluster or special or not (
            address.is_global or (allowed and private_network)
        )
        if blocked:
            msg = (
                f"Request to host: {host} port: {port} socket: "
                f"{family} resolved to internal IP: {ip_address}."
            )
            LOG.error(msg=msg)
            raise socket.gaierror(socket.EAI_FAIL, msg)

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> List[ResolveResult]:
        """Resolve a hostname and reject unsafe answers, including mixed sets.

        Public addresses and allow-listed RFC1918/ULA addresses are accepted.
        Cluster CIDRs and special or non-global addresses remain blocked even
        for allow-listed names; a rejected answer raises ``socket.gaierror``.
        """
        ips = await super().resolve(host, port, family)

        for resolved_ip in ips:
            self.check_address(resolved_ip["host"], host, port, family)

        return ips


class SSRFProtectedConnector(aiohttp.TCPConnector):
    """Apply the resolver policy to IP-literal URLs that aiohttp skips resolving."""

    def __init__(self, resolver: SSRFProtectedResolver) -> None:
        """Create a connector applying ``resolver`` to DNS and IP literals."""
        super().__init__(resolver=resolver)
        self._ssrf_resolver = resolver

    async def _resolve_host(  # pylint: disable=protected-access
        self, host: str, port: int, traces: Sequence | None = None
    ) -> list[ResolveResult]:
        """Check literal IP hosts before delegating normal DNS resolution.

        aiohttp bypasses its resolver for IP literals, so this override invokes
        the same address policy directly and raises ``socket.gaierror`` when
        the literal is blocked.
        """
        if is_ip_address(host):
            self._ssrf_resolver.check_address(host, host, port, self._family)
        return await super()._resolve_host(host, port, traces)


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
