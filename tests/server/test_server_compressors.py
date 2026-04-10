import pytest

from logdetective.server.models import (
    Response,
    Explanation,
)
from logdetective.server.compressors import LLMResponseCompressor


@pytest.mark.asyncio
async def test_server_response_compressor():
    RESPONSE_EXPLANATION = Explanation(text="A response explanation")

    response = Response(explanation=RESPONSE_EXPLANATION, snippets=None)
    response_compressor = LLMResponseCompressor(response=response)
    zip_data = response_compressor.zip_response()
    uncompressed_response = response_compressor.unzip(zip_data)
    assert isinstance(uncompressed_response, Response)
    assert uncompressed_response.explanation == RESPONSE_EXPLANATION
