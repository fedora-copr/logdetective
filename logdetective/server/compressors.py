import io
import zipfile

from typing import Union, Dict
from logdetective.remote_log import RemoteLog
from logdetective.server.models import (
    StagedResponse,
    Response,
    AnalyzedSnippet,
    Explanation,
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

    def unzip(self, zip_data: Union[bytes, io.BytesIO]) -> str:
        """
        Uncompress data created by TextCompressor.zip().

        Args:
            zip_data: A zipped stream of bytes or BytesIO object

        Returns:
            {file_name: str}: The decompressed content as a dict of file names and UTF-8 strings
        """
        if isinstance(zip_data, bytes):
            zip_buffer = io.BytesIO(zip_data)
        else:
            zip_buffer = zip_data

        content = {}
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            file_list = zip_file.namelist()
            for file_name in file_list:
                content[file_name] = zip_file.read(file_name).decode("utf-8")

        return content


class RemoteLogCompressor:
    """
    Handles compression of remote log files.
    """

    LOG_FILE_NAME = "log.txt"
    COMPRESSOR = TextCompressor()

    def __init__(self, remote_log: RemoteLog):
        """
        Initialize with a RemoteLog object.
        """
        self._remote_log = remote_log

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
        content_text = await self._remote_log.content
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


class LLMResponseCompressor:
    """
    Handles compression and decompression of LLM responses.
    """

    EXPLANATION_FILE_NAME = "explanation.txt"
    SNIPPET_FILE_NAME = "snippet_{number}.txt"
    COMPRESSOR = TextCompressor()

    def __init__(self, response: Union[StagedResponse, Response]):
        """
        Initialize with an LLM response.

        Args:
            response: Either a StagedResponse or Response object
        """
        self._response = response

    def zip_response(self) -> bytes:
        """
        Compress the content of the LLM response.

        Returns:
            bytes: Compressed response as bytes
        """
        items = {
            self.EXPLANATION_FILE_NAME: self._response.explanation.model_dump_json()
        }

        if isinstance(self._response, StagedResponse):
            for i, snippet in enumerate(self._response.snippets):
                items[self.SNIPPET_FILE_NAME.format(number=i)] = (
                    snippet.model_dump_json()
                )

        return self.COMPRESSOR.zip(items)

    @classmethod
    def unzip(
        cls, zip_data: Union[bytes, io.BytesIO]
    ) -> Union[StagedResponse, Response]:
        """
        Uncompress the zipped content of the LLM response.

        Args:
            zip_data: Compressed data as bytes or BytesIO

        Returns:
            Union[StagedResponse, Response]: The decompressed (partial) response object,
            missing response_certainty.
        """
        items = cls.COMPRESSOR.unzip(zip_data)
        if cls.EXPLANATION_FILE_NAME not in items:
            raise KeyError(
                f"Required file {cls.EXPLANATION_FILE_NAME} not found in zip archive"
            )
        explanation = Explanation.model_validate_json(items[cls.EXPLANATION_FILE_NAME])

        snippets = []
        snippet_files = {
            k: v
            for k, v in items.items()
            if cls.SNIPPET_FILE_NAME.replace("{number}.txt", "") in k
        }
        for i in range(len(snippet_files)):
            snippets.append(
                AnalyzedSnippet.model_validate_json(
                    items[cls.SNIPPET_FILE_NAME.format(number=i)]
                )
            )

        if snippets:
            response = StagedResponse(
                explanation=explanation, snippets=snippets, response_certainty=0
            )
        else:
            response = Response(explanation=explanation, response_certainty=0)

        return response
