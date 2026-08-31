from unittest import mock

import aiohttp
import aioresponses
import pytest
from jinja2.exceptions import TemplateNotFound

from logdetective.constants import DEFAULT_MAXIMUM_ARTIFACT_MIB, PROMPT_PATH
from logdetective.exceptions import (
    RemoteLogAccessError,
    RemoteLogHeaderError,
    RemoteLogRequestError,
    RemoteLogTooLargeError,
)
from logdetective.models import SkipSnippets
from logdetective.prompts import PromptManager
from logdetective.remote_log import RemoteLog
from logdetective.utils import (
    load_prompts,
    filter_snippet_patterns,
    load_skip_snippet_patterns,
)

from tests.base.test_helpers import (
    test_filter_patterns,
    test_snippets_filtering,
    simple_log,
)


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
            {"Content-Length": f"{(DEFAULT_MAXIMUM_ARTIFACT_MIB) * 1024**2 + 1}"},
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
