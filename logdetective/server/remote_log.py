import io
import logging
from typing import Union
from urllib.parse import urlparse

import aiohttp

from logdetective.server.compressors import TextCompressor


LOG = logging.getLogger("logdetective")


class RemoteLog:
    """
    Handles retrieval and compression of remote log files.
    """

    LOG_FILE_NAME = "log.txt"
    COMPRESSOR = TextCompressor()

    def __init__(self, url: str, http_session: aiohttp.ClientSession):
        """
        Initialize with a remote log URL and HTTP session.

        Args:
            url: A remote URL pointing to a log file
            http_session: The HTTP session used to retrieve the remote file
        """
        self._url = url
        self._http_session = http_session

    @property
    def url(self) -> str:
        """The remote log url."""
        return self._url

    @property
    async def content(self) -> str:
        """Content of the url."""
        return await self.get_url_content()

    @classmethod
    def zip_text(cls, text: str) -> bytes:
        """
        Compress the given text.

        Returns:
            bytes: Compressed text
        """
        return cls.COMPRESSOR.zip({cls.LOG_FILE_NAME: text})

    async def zip_content(self) -> bytes:
        """
        Compress the content of the remote log.

        Returns:
            bytes: Compressed log content
        """
        content_text = await self.content
        return self.zip_text(content_text)

    @classmethod
    def unzip(cls, zip_data: Union[bytes, io.BytesIO]) -> str:
        """
        Uncompress the zipped content of the remote log.

        Args:
            zip_data: Compressed data as bytes or BytesIO

        Returns:
            str: The decompressed log content
        """
        return cls.COMPRESSOR.unzip(zip_data)[cls.LOG_FILE_NAME]

    def validate_url(self) -> bool:
        """Validate incoming URL to be at least somewhat sensible for log files
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
        """validate log url and return log text."""
        if self.validate_url():
            LOG.debug("process url %s", self.url)
            try:
                response = await self._http_session.get(self.url, raise_for_status=True)
            except aiohttp.ClientResponseError as ex:
                raise RuntimeError(f"We couldn't obtain the logs: {ex}") from ex
            return await response.text()
        LOG.error("Invalid URL received ")
        raise RuntimeError(f"Invalid log URL: {self.url}")

    async def process_url(self) -> str:
        """Validate log URL and return log text."""
        try:
            return await self.get_url_content()
        except RuntimeError as ex:
            raise aiohttp.HTTPException(
                status_code=400, detail=f"We couldn't obtain the logs: {ex}"
            ) from ex
