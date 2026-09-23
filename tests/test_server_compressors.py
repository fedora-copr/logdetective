import pytest

from logdetective.models import (
    APIResponse,
    AnalyzedSnippet,
    Explanation,
    Snippet,
    Solution,
)
from logdetective.compressors import LLMResponseCompressor

RESPONSE_EXPLANATION = Explanation(text="A response explanation")


@pytest.mark.parametrize(
    "snippets",
    [
        None,
        [],
        [
            Snippet(
                text="This is a snippet text", line_number=10, source_file="source.log"
            )
        ],
    ],
)
@pytest.mark.parametrize(
    "no_issue_found",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "response",
    [
        APIResponse(explanation=RESPONSE_EXPLANATION, snippets=None),
        APIResponse(
            explanation=RESPONSE_EXPLANATION,
            solution=Solution(text="Solution text"),
            snippets=None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_server_response_compressor(
    response: APIResponse, no_issue_found: bool, snippets: list
):

    response.snippets = snippets
    response.no_issue_found = no_issue_found
    response_compressor = LLMResponseCompressor(response=response)
    zip_data = response_compressor.zip_response()
    uncompressed_response = response_compressor.unzip(zip_data)
    assert isinstance(uncompressed_response, APIResponse)
    assert uncompressed_response == response


@pytest.mark.parametrize(
    "snippet",
    [
        Snippet(text="compiler output", line_number=12, source_file="build.log"),
        AnalyzedSnippet(
            text="compiler output",
            line_number=12,
            source_file="build.log",
            snippet_analysis="The compiler rejected the expression.",
        ),
    ],
)
def test_response_compressor_preserves_supported_snippet_types(
    snippet: Snippet,
) -> None:
    """Round-trip both plain and LLM-analyzed snippets.

    Args:
        snippet: Supported snippet variant to compress and restore.

    Returns:
        None. The assertions verify that decompression retains the concrete model
        and all of its fields.
    """
    response = APIResponse(
        explanation=Explanation(text="A response explanation"), snippets=[snippet]
    )

    uncompressed = LLMResponseCompressor(response).unzip(
        LLMResponseCompressor(response).zip_response()
    )

    snippets = uncompressed.snippets
    assert snippets is not None
    assert snippets == [snippet]
    assert type(snippets[0]) is type(snippet)  # pylint: disable=unsubscriptable-object
