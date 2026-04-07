import os
import logging
from urllib.parse import urlparse

import aiohttp

from logdetective.constants import DEFAULT_MAXIMUM_ARTIFACT_MIB
from logdetective.exceptions import (
    RemoteLogRequestError,
    RemoteLogHeaderError,
    RemoteLogAccessError,
    RemoteLogTooLargeError,
)
from logdetective.utils import (
    ContentSizeCheck,
    check_content_size,
    mib_to_bytes,
)

LOG = logging.getLogger("logdetective")


class RemoteLog:
    """
    Handles retrieval of remote log files.
    """

    remote_log_size: int = 0

    def __init__(
        self,
        url: str,
        http_session: aiohttp.ClientSession,
        limit_bytes: int = mib_to_bytes(DEFAULT_MAXIMUM_ARTIFACT_MIB),
    ):
        """
        Initialize with a remote log URL and HTTP session.

        Args:
            url: A remote URL pointing to a log file
            http_session: The HTTP session used to retrieve the remote file
            limit_bytes: For checking the log size on the accessed URL
        """
        self._url = url
        self._http_session = http_session
        self._limit_bytes = limit_bytes

    @property
    def url(self) -> str:
        """The remote log url."""
        return self._url

    @property
    async def content(self) -> str:
        """Content of the url."""
        return await self.get_url_content()

    def validate_url(self) -> bool:
        """Validate incoming URL to be at least somewhat sensible for log files.
        Only http and https protocols permitted. No result, params or query fields allowed.
        Either netloc or path must have non-zero length.
        """
        result = urlparse(self.url)
        if result.scheme not in ["http", "https"]:
            return False
        if any([result.params, result.query, result.fragment]):
            return False
        if not (result.path or result.netloc):
            return False
        return True

    async def get_url_content(self) -> str:
        """Validate log url, check the content size from header, and return log text."""
        if not self.validate_url():
            LOG.error("Invalid URL received ")
            raise RemoteLogRequestError(f"Invalid log URL: {self.url}")
        LOG.debug("process url %s", self.url)
        # obtain the head for size-check
        try:
            head_response = await self._http_session.head(
                self.url, raise_for_status=True
            )
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectorError) as ex:
            raise RemoteLogAccessError(f"We couldn't obtain the headers from {self.url}") from ex
        size_check: ContentSizeCheck = check_content_size(
            head_response.headers, self._limit_bytes
        )
        if not size_check.result:
            if size_check.size_in_bytes is None:
                if not size_check.value_present:
                    raise RemoteLogHeaderError("Content-Length is missing")
                raise RemoteLogHeaderError(
                    f"Content-Length is invalid: `{size_check.size_in_bytes}`"
                )
            raise RemoteLogTooLargeError(
                f"Content-Length is over the limit: `{size_check.size_in_bytes}`"
            )
        self.remote_log_size = size_check.size_in_bytes
        # if size-check passes, we obtain the whole content
        try:
            response = await self._http_session.get(self.url, raise_for_status=True)
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectorError) as ex:
            raise RemoteLogAccessError(f"We couldn't obtain the log from {self.url}") from ex
        return await response.text()


async def retrieve_log_content(
    http: aiohttp.ClientSession, log_path: str, size_limit: int
) -> str:
    """Get content of the file on the log_path path.
    Path is assumed to be valid URL if it has a scheme.
    Otherwise it attempts to pull it from local filesystem."""
    parsed_url = urlparse(log_path)
    log = ""

    if not parsed_url.scheme:
        if not os.path.exists(log_path):
            raise ValueError(f"Local log {log_path} doesn't exist!")

        with open(log_path, "rt") as f:
            log = f.read()

    else:
        remote_log = RemoteLog(log_path, http, limit_bytes=size_limit)
        # limited to DEFAULT_MAXIMUM_ARTIFACT_MIB (50 MiB)
        log = await remote_log.get_url_content()

    return log
