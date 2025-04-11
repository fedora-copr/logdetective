from unittest import mock

import aiohttp
import aioresponses
import pytest

from logdetective.utils import (
    compute_certainty,
    format_snippets,
    load_prompts,
    get_url_content,
)
from logdetective.server.utils import format_analyzed_snippets
from logdetective.server.models import AnalyzedSnippet, Explanation
from logdetective.models import PromptConfig
from logdetective import constants

test_snippets = [
    # Simple
    ["Snippet 1", "Snippet 2", "Snippet 3"],
    # Tuples
    [(10, "Snippet 1"), (120, "Snippet 1"), (240, "Snippet 1")],
]

test_analyzed_snippets = [
    [
        AnalyzedSnippet(
            text=e[1],
            line_number=e[0],
            explanation=Explanation(
                text=f"Comment for snippet on line {e[0]}",
                logprobs=[{"logprob": 66.6}, {"logprob": 99.9}, {"logprob": 1.0}],
            ),
        )
        for e in test_snippets[1]
    ]
]

test_prompts = """
prompt_template: This is basic template.

snippet_prompt_template: This is template for snippets.
"""


@pytest.mark.parametrize(
    "probs", ([{"logprob": 66.6}], [{"logprob": 99.9}, {"logprob": 1.0}])
)
def test_compute_certainty(probs):
    """test compute_certainty and make sure we can use numpy correctly"""
    compute_certainty(probs)


@pytest.mark.parametrize("snippets", test_snippets)
def test_format_snippets(snippets):
    """Test snippet formatting with both simple snippets, and line numbers"""
    format_snippets(snippets)


@pytest.mark.parametrize("snippets", test_analyzed_snippets)
def test_format_analyzed_snippets(snippets):
    """Test snippet formatting for snippets with LLM generated explanations"""
    format_analyzed_snippets(snippets)


def test_load_prompts_wrong_path():
    """Test behavior for case when the path doesn't lead to a any file."""
    prompts_config = load_prompts("/there/is/nothing/to/read.yml")

    assert isinstance(prompts_config, PromptConfig)

    assert prompts_config.prompt_template == constants.PROMPT_TEMPLATE
    assert prompts_config.snippet_prompt_template == constants.SNIPPET_PROMPT_TEMPLATE
    assert prompts_config.prompt_template_staged == constants.PROMPT_TEMPLATE_STAGED
    assert prompts_config.summarization_prompt_template == constants.SUMMARIZATION_PROMPT_TEMPLATE


def test_load_prompts_correct_path():
    """Test behavior for case when the path is correct and only
    some prompts are overriden with user settings, the rest must remain
    set to defaults in `constants`."""

    with mock.patch("logdetective.utils.open", mock.mock_open(read_data=test_prompts)):
        prompts_config = load_prompts("/there/is/nothing/to/read.yml")

    assert isinstance(prompts_config, PromptConfig)

    assert prompts_config.prompt_template == "This is basic template."
    assert prompts_config.snippet_prompt_template == "This is template for snippets."
    assert prompts_config.prompt_template_staged == constants.PROMPT_TEMPLATE_STAGED
    assert prompts_config.summarization_prompt_template == constants.SUMMARIZATION_PROMPT_TEMPLATE


@pytest.mark.asyncio
async def test_get_url_content():
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.get('http://localhost:8999/', status=200, body=mock_response)
        async with aiohttp.ClientSession() as http:
            url_output_cr = get_url_content(http, "http://localhost:8999/", 4)
            url_output = await url_output_cr
            assert url_output == "123"
