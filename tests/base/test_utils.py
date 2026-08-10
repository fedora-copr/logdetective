from unittest import mock

import aiohttp
import aioresponses
import pytest
from jinja2.exceptions import TemplateNotFound

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
from logdetective.constants import DEFAULT_MAXIMUM_ARTIFACT_MIB, PROMPT_PATH, TRUNCATED
from logdetective.remote_log import RemoteLog
from logdetective.exceptions import (
    RemoteLogAccessError,
    RemoteLogHeaderError,
    RemoteLogRequestError,
    RemoteLogTooLargeError,
)
from logdetective.models import SkipSnippets
from logdetective.prompts import PromptManager

from tests.base.test_helpers import (
    test_snippets,
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

    with pytest.raises(TemplateNotFound):
        load_prompts("/there/is/nothing/to/read.yml")


def test_load_prompts_correct_path():
    """Test behavior for case when the path is correct."""

    prompts_config = load_prompts(template_path=PROMPT_PATH)

    assert isinstance(prompts_config, PromptManager)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, mock_header, mock_body, limit_bytes, exc_type, exc_match",
    [
        ("http://example.com/build.log", {"Content-Length": "3"}, "123", None, None, None),
        (
            "http://example.com/build.log",
            {"Content-Length": "test"},
            "123",
            None,
            RemoteLogHeaderError,
            "Content-Length header is invalid",
        ),
        (
            "http://example.com/build.log",
            {"Content-Length": f"{mib_to_bytes(DEFAULT_MAXIMUM_ARTIFACT_MIB) + 1}"},
            "123",
            None,
            RemoteLogTooLargeError,
            "Content-Length is over the limit",
        ),
        ("not-a-valid-url", {}, "123", None, RemoteLogRequestError, "Invalid log URL"),
        # No Content-Length: small body succeeds
        ("http://example.com/build.log", {}, "123", None, None, None),
        # No Content-Length: body over limit is rejected while reading
        (
            "http://example.com/build.log",
            {},
            "x" * 20,
            10,
            RemoteLogTooLargeError,
            "exceeds the limit",
        ),
        # Lying Content-Length: declared within limit, actual body over limit
        (
            "http://example.com/build.log",
            {"Content-Length": "3"},
            "x" * 20,
            10,
            RemoteLogTooLargeError,
            "exceeds the limit",
        ),
    ],
    indirect=False,
)
# pylint: disable=too-many-arguments,too-many-positional-arguments
async def test_get_url_content(
    url,
    mock_header,
    mock_body,
    limit_bytes,
    exc_type,
    exc_match
):
    """Test various URL requests and correct Exceptions during RemoteLog access."""
    with aioresponses.aioresponses() as mock:
        mock.head(url, status=200, headers=mock_header)
        mock.get(url, status=200, body=mock_body)
        async with aiohttp.ClientSession() as http:
            kwargs = {"limit_bytes": limit_bytes} if limit_bytes is not None else {}
            if exc_type:
                with pytest.raises(exc_type, match=exc_match):
                    remote_log = RemoteLog(url, http, **kwargs)
                    await remote_log.get_url_content()
            else:
                remote_log = RemoteLog(url, http, **kwargs)
                url_output = await remote_log.get_url_content()
                assert url_output == mock_body


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
        result = filter_snippet_patterns(snippet[0], skip_snippets=skip_snippets)
        assert result == snippet[1]


def _to_toml(patterns: dict) -> bytes:
    """Serialize a patterns dict to TOML bytes for use in tests."""
    lines = []
    for name, entry in patterns.items():
        lines.append(f"[{name}]")
        lines.append(f"pattern = '{entry['pattern']}'")
        if "files" in entry:
            files_str = ", ".join(f"'{f}'" for f in entry["files"])
            lines.append(f"files = [{files_str}]")
        lines.append("")
    return "\n".join(lines).encode()


def test_load_skip_snippet_patterns_wrong_path():
    """Test behavior for case when the path doesn't lead to a any file."""

    default_skip_pattern = load_skip_snippet_patterns("/there/is/nothing/to/read.toml")

    assert default_skip_pattern is None


def test_load_skip_snippet_patterns_no_path():
    """Test behavior for case when no path is provided."""

    assert load_skip_snippet_patterns(None) is None


def test_load_skip_snippet_patterns_empty_file():
    """Test behavior for case when the file exists but is empty."""

    with mock.patch("logdetective.utils.open", mock.mock_open(read_data=b"")):
        result = load_skip_snippet_patterns("/valid/but/empty.toml")

    assert result is None


def test_load_skip_snippet_patterns_only_comments():
    """Test behavior for case when the file is functionally empty (only comments)."""

    with mock.patch(
        "logdetective.utils.open",
        mock.mock_open(read_data=b"# commented out stuff\n# no patterns here\n"),
    ):
        result = load_skip_snippet_patterns("/valid/but/empty.toml")

    assert result is None


def test_load_skip_snippet_patterns_correct_path():
    """Test behavior for case when the path is correct.
    All patterns must be parsed successfully and match
    those from original source."""

    with mock.patch(
        "logdetective.utils.open", mock.mock_open(read_data=_to_toml(test_filter_patterns))
    ):
        prompts_config = load_skip_snippet_patterns("/valid/filters.toml")

    assert isinstance(prompts_config, SkipSnippets)
    assert len(prompts_config.snippet_patterns) == len(test_filter_patterns)


def test_load_skip_snippet_patterns_invalid_syntax():
    """Test behavior for case when the pattern is not a valid regular expression."""

    test_skip_snippet_data = b"[bad_regex]\npattern = '$**.^.*'\n"

    with mock.patch(
        "logdetective.utils.open", mock.mock_open(read_data=test_skip_snippet_data)
    ):
        with pytest.raises(ValueError, match="Invalid pattern"):
            load_skip_snippet_patterns("/valid/filters.toml")


def test_load_skip_snippet_patterns_missing_pattern_key():
    """Test behavior when an entry is missing the required 'pattern' key."""

    test_skip_snippet_data = b"[bad_entry]\nnot_pattern = 'just_a_string'\n"

    with mock.patch(
        "logdetective.utils.open", mock.mock_open(read_data=test_skip_snippet_data)
    ):
        with pytest.raises(ValueError, match="must be a mapping"):
            load_skip_snippet_patterns("/valid/filters.toml")


def test_snippet_filtering_file_scoped():
    """File-scoped patterns apply only to exact filename matches."""
    skip_snippets = SkipSnippets({
        "global_pattern": {"pattern": "^GLOBAL"},
        "scoped_pattern": {"pattern": "^SCOPED", "files": ["backend.log", "app.log"]},
    })

    # Global pattern fires regardless of filename
    assert filter_snippet_patterns("GLOBAL line", "build.log", skip_snippets) is True
    assert filter_snippet_patterns("GLOBAL line", "other.log", skip_snippets) is True

    # Scoped pattern fires only for exact filename matches
    assert filter_snippet_patterns("SCOPED line", "build.log", skip_snippets) is False
    assert filter_snippet_patterns("SCOPED line", "backend.log", skip_snippets) is True
    assert filter_snippet_patterns("SCOPED line", "app.log", skip_snippets) is True
    assert filter_snippet_patterns("SCOPED line", "unknown.log", skip_snippets) is False


def test_get_patterns_for_file_exact_matching():
    """get_patterns_for_file uses exact filename matching."""
    skip_snippets = SkipSnippets({
        "a": {"pattern": "^A", "files": ["backend.log", "app.log"]},
        "b": {"pattern": "^B", "files": ["build.log"]},
        "c": {"pattern": "^C"},
    })

    patterns_build = skip_snippets.get_patterns_for_file("build.log")
    assert set(patterns_build) == {"b", "c"}

    patterns_backend = skip_snippets.get_patterns_for_file("backend.log")
    assert set(patterns_backend) == {"a", "c"}

    patterns_unknown = skip_snippets.get_patterns_for_file("unknown.txt")
    assert set(patterns_unknown) == {"c"}


@pytest.mark.parametrize("max_chunk_len", [100, 120, 150])
def test_get_chunks_max_length(simple_log, max_chunk_len):
    """Test that maximum length of chunks is properly enforced"""
    log = "".join(simple_log)
    chunks = list(get_chunks(log, max_chunk_len=max_chunk_len))

    # Number of chunks must be <= number of original lines
    assert len(chunks) <= len(simple_log)

    # All chunks must obey contraints and exist in the original text
    for c in chunks:
        assert len(c[1]) <= max_chunk_len
        assert c[1][:-len(TRUNCATED)] in log

    # All chunks must have unique lines
    lines = set(c[0] for c in chunks)
    assert len(lines) == len(chunks)

    # Last chunk must not be empty
    assert len(chunks[-1][1]) > 0


@pytest.mark.parametrize("max_chunk_len", [20, 50, 70])
def test_get_chunks_raises_on_too_short(simple_log, max_chunk_len):

    log = "".join(simple_log)
    with pytest.raises(ValueError):
        list(get_chunks(log, max_chunk_len=max_chunk_len))


def test_empty_log_creates_no_chunks():
    log = ""
    chunks = list(get_chunks(log))

    assert len(chunks) == 0


def test_leading_whitespace_chunks(simple_log):

    log = " ".join(simple_log)

    chunks = list(get_chunks(log))

    # Number of chunks must be <= number of original lines
    assert len(chunks) <= len(simple_log)

    for chunk in chunks[1:]:
        assert chunk[1].startswith(" ")

    # All chunks must have unique lines
    lines = set(c[0] for c in chunks)
    assert len(lines) == len(chunks)

    # Last chunk must not be empty
    assert len(chunks[-1][1]) > 0
