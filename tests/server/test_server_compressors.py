import pytest

from logdetective.server.models import (
    Response,
    Explanation,
)
from logdetective.server.compressors import LLMResponseCompressor


@pytest.mark.asyncio
async def test_server_response_compressor():
    LOGPROBS = [{"logprob": 66.6}, {"logprob": 99.9}, {"logprob": 1.0}]
    RESPONSE_EXPLANATION = Explanation(text="A response explanation", logprobs=LOGPROBS)

    response = Response(explanation=RESPONSE_EXPLANATION, response_certainty=99, snippets=None)
    response_compressor = LLMResponseCompressor(response=response)
    zip_data = response_compressor.zip_response()
    uncompressed_response = response_compressor.unzip(zip_data)
    assert isinstance(uncompressed_response, Response)
    assert uncompressed_response.explanation == RESPONSE_EXPLANATION
