import io
import logging
import zipfile

from urllib.parse import urlparse

import aiohttp


LOG = logging.getLogger("logdetective")


class RemoteLog:
    """An object cabaple of dealing with
    a log represented by its url.
    """

    ZIP_FILE_NAME = "log.txt"

    def __init__(self, url: str, http_session: aiohttp.ClientSession):
        """Returns an object cabaple of dealing with
        a log represented by its url.
        Using a specified http_session.

        Args:
          url str: a remote url
          http_session aiohttp.ClientSession: the http session used for
              retrieve the remote file.

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
    def zip(cls, text: str) -> io.BytesIO:
        """Compress a text."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(cls.ZIP_FILE_NAME, text)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    @property
    async def zip_content(self) -> io.BytesIO:
        """Compress the content of the url log."""
        return self.zip(await self.content)

    @classmethod
    def unzip(cls, zip_data: io.BytesIO) -> str:
        """Uncompress data created by Log.zip_content()."""
        zip_buffer = io.BytesIO(zip_data)
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            content = zip_file.read(cls.ZIP_FILE_NAME)
        return content.decode("utf-8")

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
