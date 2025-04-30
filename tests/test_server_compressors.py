import pytest

from logdetective.server.models import (
    Response,
    StagedResponse,
    Explanation,
    AnalyzedSnippet,
)
from logdetective.server.compressors import LLMResponseCompressor


@pytest.mark.asyncio
async def test_server_response_compressor():
    LOGPROBS = [{"logprob": 66.6}, {"logprob": 99.9}, {"logprob": 1.0}]
    RESPONSE_EXPLANATION = Explanation(text="A response explanation", logprobs=LOGPROBS)

    response = Response(explanation=RESPONSE_EXPLANATION, response_certainty=99)
    response_compressor = LLMResponseCompressor(response=response)
    zip_data = response_compressor.zip_response()
    uncompressed_response = response_compressor.unzip(zip_data)
    assert isinstance(uncompressed_response, Response)
    assert uncompressed_response.explanation == RESPONSE_EXPLANATION

    STAGED_RESPONSE_EXPLANATION = Explanation(
        text="A staged response explanation", logprobs=LOGPROBS
    )
    SNIPPETS = [
        AnalyzedSnippet(
            text=e[1],
            line_number=e[0],
            explanation=Explanation(
                text=f"Comment for snippet on line {e[0]}",
                logprobs=LOGPROBS,
            ),
        )
        for e in [(10, "Snippet 1"), (120, "Snippet 1"), (240, "Snippet 1")]
    ]
    staged_response = StagedResponse(
        explanation=STAGED_RESPONSE_EXPLANATION,
        snippets=SNIPPETS,
        response_certainty=99,
    )
    response_compressor = LLMResponseCompressor(response=staged_response)
    zip_data = response_compressor.zip_response()
    uncompressed_response = response_compressor.unzip(zip_data)
    assert isinstance(uncompressed_response, StagedResponse)
    assert uncompressed_response.explanation == STAGED_RESPONSE_EXPLANATION
    assert uncompressed_response.snippets == SNIPPETS
