from unittest import mock

import aiohttp
import aioresponses
import pytest

from logdetective.utils import (
    compute_certainty,
    format_snippets,
    load_prompts,
)

from logdetective.remote_log import RemoteLog
from logdetective.models import PromptConfig
from logdetective import constants

from tests.base.test_helpers import test_snippets, test_prompts


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


def test_load_prompts_wrong_path():
    """Test behavior for case when the path doesn't lead to a any file."""
    prompts_config = load_prompts("/there/is/nothing/to/read.yml")

    assert isinstance(prompts_config, PromptConfig)

    assert prompts_config.prompt_template == constants.PROMPT_TEMPLATE
    assert prompts_config.snippet_prompt_template == constants.SNIPPET_PROMPT_TEMPLATE
    assert prompts_config.prompt_template_staged == constants.PROMPT_TEMPLATE_STAGED
    assert (
        prompts_config.summarization_prompt_template
        == constants.SUMMARIZATION_PROMPT_TEMPLATE  # noqa: W503 flake vs lint
    )


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
    assert (
        prompts_config.summarization_prompt_template
        == constants.SUMMARIZATION_PROMPT_TEMPLATE  # noqa: W503 flake vs lint
    )


@pytest.mark.asyncio
async def test_get_url_content():
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.get("http://localhost:8999/", status=200, body=mock_response)
        async with aiohttp.ClientSession() as http:
            url_output_cr = RemoteLog("http://localhost:8999/", http).get_url_content()
            url_output = await url_output_cr
            assert url_output == "123"
