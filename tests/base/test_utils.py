from unittest import mock

import aiohttp
import aioresponses
import pytest


from logdetective.utils import (
    compute_certainty,
    format_snippets,
    load_prompts,
    prompt_to_messages,
    filter_snippet_patterns,
    load_skip_snippet_patterns,
    get_chunks,
    mib_to_bytes,
)
from logdetective.constants import DEFAULT_MAXIMUM_ARTIFACT_MIB
from logdetective.remote_log import RemoteLog
from logdetective.exceptions import (
    RemoteLogAccessError,
    RemoteLogHeaderError,
    RemoteLogRequestError,
    RemoteLogTooLargeError,
)
from logdetective.models import PromptConfig, SkipSnippets
from logdetective import constants

from tests.base.test_helpers import (
    test_snippets,
    test_prompts,
    test_filter_patterns,
    test_snippets_filtering,
    simple_log,
)


@pytest.mark.parametrize(
    "probs", ([{"logprob": 66.6}], [{"logprob": 99.9}, {"logprob": 1.0}])
)
def test_compute_certainty(probs):
    """test compute_certainty and make sure we can use numpy correctly"""
    compute_certainty(probs)


@pytest.mark.parametrize("snippets", test_snippets)
def test_format_snippets(snippets):
    """Test snippet formatting with both simple snippets, and line numbers"""
    formatted_snippets = format_snippets(snippets)

    for snippet in snippets:
        if isinstance(snippet, tuple):
            assert str(snippet[0]) in formatted_snippets
            assert snippet[1] in formatted_snippets
        else:
            assert snippet in formatted_snippets


def test_load_prompts_wrong_path():
    """Test behavior for case when the path doesn't lead to a any file."""
    prompts_config = load_prompts("/there/is/nothing/to/read.yml")

    assert isinstance(prompts_config, PromptConfig)

    assert prompts_config.prompt_template == constants.PROMPT_TEMPLATE


def test_load_prompts_correct_path():
    """Test behavior for case when the path is correct and only
    some prompts are overriden with user settings, the rest must remain
    set to defaults in `constants`."""

    with mock.patch("logdetective.utils.open", mock.mock_open(read_data=test_prompts)):
        prompts_config = load_prompts("/there/is/nothing/to/read.yml")

    assert isinstance(prompts_config, PromptConfig)

    assert prompts_config.prompt_template == "This is basic template."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, mock_header, exc_type, exc_match",
    [
        ("http://example.com/build.log", {"Content-Length": "3"}, None, None),
        (
            "http://example.com/build.log",
            {"Content-Length": "test"},
            RemoteLogHeaderError,
            "Content-Length is invalid",
        ),
        (
            "http://example.com/build.log",
            {"Content-Length": f"{mib_to_bytes(DEFAULT_MAXIMUM_ARTIFACT_MIB) + 1}"},
            RemoteLogTooLargeError,
            "Content-Length is over the limit",
        ),
        ("not-a-valid-url", {}, RemoteLogRequestError, "Invalid log URL"),
        ("http://example.com/build.log", {}, RemoteLogHeaderError, "Content-Length is missing")
    ],
    indirect=False,
)
async def test_get_url_content(url, mock_header, exc_type, exc_match):
    """Test various URL requests and correct Exceptions during RemoteLog access."""
    mock_response = "123"
    with aioresponses.aioresponses() as mock:
        mock.head(url, status=200, headers=mock_header)
        mock.get(url, status=200, body=mock_response)
        async with aiohttp.ClientSession() as http:
            if exc_type:
                with pytest.raises(exc_type, match=exc_match):
                    remote_log = RemoteLog(url, http)
                    await remote_log.get_url_content()
            else:
                remote_log = RemoteLog(url, http)
                url_output = await remote_log.get_url_content()
                assert url_output == "123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "head_status, get_status",
    [(404, 200), (500, 200), (503, 200), (200, 404), (200, 500), (200, 503)],
    indirect=False,
)
async def test_get_url_content_connection_fails(head_status, get_status):
    """Test HEAD/GET failures during RemoteLog access. All should raise RemoteLogAccessError."""
    url = "http://example.com/build.log"
    mock_head_response = {"Content-Length": "11"} if head_status <= 399 else None
    mock_get_response = "Lorem Ipsum"
    with aioresponses.aioresponses() as mock:
        mock.head(url, status=head_status, headers=mock_head_response)
        mock.get(url, status=get_status, body=mock_get_response)
        async with aiohttp.ClientSession() as http:
            remote_log = RemoteLog(url, http)
            with pytest.raises(RemoteLogAccessError):
                await remote_log.get_url_content()


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


def test_snippet_filtering():
    """Test snippet filtering"""

    skip_snippets = SkipSnippets(test_filter_patterns)
    for snippet in test_snippets_filtering:
        assert (
            filter_snippet_patterns(snippet[0], skip_snippets=skip_snippets) == snippet[1]
        )


def test_load_skip_snippet_patterns_wrong_path():
    """Test behavior for case when the path doesn't lead to a any file."""

    default_skip_pattern = load_skip_snippet_patterns("/there/is/nothing/to/read.yml")

    assert isinstance(default_skip_pattern, SkipSnippets)
    assert len(default_skip_pattern.snippet_patterns) == 0


def test_load_skip_snippet_patterns_correct_path():
    """Test behavior for case when the path is correct.
    All patterns must be parsed successfully and match
    those from original source."""

    test_skip_snippet_data = ""

    for key, value in test_filter_patterns.items():
        test_skip_snippet_data += f'{key}: "{value}"\n'

    with mock.patch(
        "logdetective.utils.open", mock.mock_open(read_data=test_skip_snippet_data)
    ):
        prompts_config = load_skip_snippet_patterns("/valid/filters.yml")

    assert isinstance(prompts_config, SkipSnippets)

    assert len(prompts_config.snippet_patterns) == len(test_filter_patterns)


def test_load_skip_snippet_patterns_invalid_syntax():
    """Test behavior for case when the syntax of patterns
    is incorrect. This must trigger an error."""

    test_skip_snippet_data = "this_is_not_a_regex: $**.^.*\n"

    with mock.patch(
        "logdetective.utils.open", mock.mock_open(read_data=test_skip_snippet_data)
    ):
        with pytest.raises(ValueError, match="Invalid pattern"):
            load_skip_snippet_patterns("/there/is/nothing/to/read.yml")


@pytest.mark.parametrize("max_chunk_len", [10, 20, 100])
def test_get_chunks_max_length(simple_log, max_chunk_len):
    """Test that maximum length of chunks is properly enforced
    and that no text is lost"""
    log = "".join(simple_log)
    chunks = list(get_chunks(log, max_chunk_len=max_chunk_len))
    reconstructed_text = ""
    for c in chunks:
        assert len(c[1]) <= max_chunk_len
        assert c[1] in log
        reconstructed_text += c[1]

    for _, line in enumerate(simple_log):
        if len(line) > 0:
            assert line.strip() in reconstructed_text
