from unittest.mock import patch, mock_open

import pytest
from pydantic import ValidationError

from logdetective.config import load_skip_snippet_patterns
from logdetective.models import SkipSnippets

from tests.test_snippets import test_filter_patterns


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

    with patch("logdetective.config.open", mock_open(read_data=b"")):
        result = load_skip_snippet_patterns("/valid/but/empty.toml")

    assert result is None


def test_load_skip_snippet_patterns_only_comments():
    """Test behavior for case when the file is functionally empty (only comments)."""

    with patch(
        "logdetective.config.open",
        mock_open(read_data=b"# commented out stuff\n# no patterns here\n"),
    ):
        result = load_skip_snippet_patterns("/valid/but/empty.toml")

    assert result is None


def test_load_skip_snippet_patterns_correct_path():
    """Test behavior for case when the path is correct.
    All patterns must be parsed successfully and match
    those from original source."""

    with patch(
        "logdetective.config.open", mock_open(read_data=_to_toml(test_filter_patterns))
    ):
        prompts_config = load_skip_snippet_patterns("/valid/filters.toml")

    assert isinstance(prompts_config, SkipSnippets)
    assert len(prompts_config.snippet_patterns) == len(test_filter_patterns)


def test_load_skip_snippet_patterns_invalid_syntax():
    """Test behavior for case when the pattern is not a valid regular expression."""

    test_skip_snippet_data = b"[bad_regex]\npattern = '$**.^.*'\n"

    with patch(
        "logdetective.config.open", mock_open(read_data=test_skip_snippet_data)
    ):
        with pytest.raises(ValidationError, match="regular expression"):
            load_skip_snippet_patterns("/valid/filters.toml")


def test_load_skip_snippet_patterns_missing_pattern_key():
    """Test behavior when an entry is missing the required 'pattern' key."""

    test_skip_snippet_data = b"[bad_entry]\nnot_pattern = 'just_a_string'\n"

    with patch(
        "logdetective.config.open", mock_open(read_data=test_skip_snippet_data)
    ):
        with pytest.raises(ValidationError):
            load_skip_snippet_patterns("/valid/filters.toml")
