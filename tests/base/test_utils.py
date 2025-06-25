from unittest import mock

import aiohttp
import aioresponses
import pytest

from logdetective.utils import (
    compute_certainty,
    format_snippets,
    load_prompts,
    prompt_to_messages,
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


@pytest.mark.asyncio
async def test_get_url_content():
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.get("http://localhost:8999/", status=200, body=mock_response)
        async with aiohttp.ClientSession() as http:
            url_output_cr = RemoteLog("http://localhost:8999/", http).get_url_content()
            url_output = await url_output_cr
            assert url_output == "123"


@pytest.mark.parametrize("user_role", ["user", "something"])
@pytest.mark.parametrize("system_role", ["developer", "user"])
def test_message_formatting(system_role, user_role):
    """Test message formatting utility function."""
    user_msg = "Hello world!"
    system_msg = "This is a system message!"
    expected_messages_separate_roles = [
        {
            "role": system_role,
            "content": system_msg,
        },
        {
            "role": user_role,
            "content": user_msg,
        },
    ]

    expected_messages_single_role = [
        {"role": user_role, "content": f"{system_msg}\n{user_msg}"}
    ]

    messages = prompt_to_messages(user_msg, system_msg, system_role, user_role)
    # Test concatenation of messages if system_role and user_role are the same,
    # this behavior is necessary for Log Detective to work with models that were
    # not trained with a separate system user.
    if system_role and user_role and system_role == user_role:
        assert expected_messages_single_role == messages
    else:
        assert expected_messages_separate_roles == messages
