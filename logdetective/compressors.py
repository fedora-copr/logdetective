import io
import zipfile

from typing import Dict
from logdetective.models import (
    APIResponse,
)


class TextCompressor:
    """
    Encapsulates one or more texts in one or more files with the specified names
    and provides methods to retrieve them later.
    """

    def zip(self, items: Dict[str, str]) -> bytes:
        """
        Compress multiple texts into different files within a zip archive.

        Args:
            items: Dictionary where keys are file names and values are text content
                  to be compressed

        Returns:
            bytes: The compressed zip archive as bytes
        """
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for key, value in items.items():
                zip_file.writestr(key, value)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def unzip(self, zip_data: bytes) -> Dict[str, str]:
        """
        Uncompress data created by TextCompressor.zip().

        Args:
            zip_data: A zipped stream of bytes

        Returns:
            {file_name: str}: The decompressed content as a dict of file names and UTF-8 strings
        """
        zip_buffer = io.BytesIO(zip_data)

        content = {}
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            file_list = zip_file.namelist()
            for file_name in file_list:
                content[file_name] = zip_file.read(file_name).decode("utf-8")

        return content


class LLMResponseCompressor:
    """
    Handles compression and decompression of LLM responses.
    """

    RESPONSE_FILE_NAME = "response.json"
    COMPRESSOR = TextCompressor()

    def __init__(self, response: APIResponse):
        """
        Initialize with an LLM response.

        Args:
            response: Response object
        """
        self._response = response

    def zip_response(self) -> bytes:
        """
        Compress the content of the LLM response.

        Returns:
            bytes: Compressed response as bytes
        """
        items = {
            self.RESPONSE_FILE_NAME: self._response.model_dump_json(),
        }

        return self.COMPRESSOR.zip(items)

    @classmethod
    def unzip(
        cls, zip_data: bytes
    ) -> APIResponse:
        """
        Uncompress the zipped content of the LLM response.

        Args:
            zip_data: Compressed data as bytes

        Returns:
            Union[Response]: The decompressed (partial) response object
        """
        items = cls.COMPRESSOR.unzip(zip_data)
        if cls.RESPONSE_FILE_NAME not in items:
            raise KeyError(
                f"Required file {cls.RESPONSE_FILE_NAME} not found in zip archive"
            )
        response = APIResponse.model_validate_json(items[cls.RESPONSE_FILE_NAME])

        return response
