import subprocess as sp
from unittest.mock import MagicMock
import pytest

from logdetective.models import SkipSnippets, CSGrepOutput, CSGrepDefect, CSGrepEvent
from logdetective.extractors import DrainExtractor, CSGrepExtractor, Extractor


@pytest.fixture
def simple_log():
    """Provides a simple log for testing."""
    return """This is a test log.
This is another test log.
An error occurred: file not found.
An error occurred: permission denied.
Another line.
    """


@pytest.fixture
def csgrep_output():
    """Provides a sample csgrep JSON output using the new data structures."""
    return CSGrepOutput(
        defects=[
            CSGrepDefect(
                checker="some-checker",
                language="C",
                tool="gcc",
                key_event_idx=0,
                events=[
                    CSGrepEvent(
                        file_name="test.c",
                        line=3,
                        event="error",
                        message="An error occurred: file not found.",
                        verbosity_level=1,
                    )
                ],
            ),
            CSGrepDefect(
                checker="another-checker",
                language="C++",
                tool="g++",
                key_event_idx=0,
                events=[
                    CSGrepEvent(
                        file_name="test.cpp",
                        line=4,
                        event="error",
                        message="An error occurred: permission denied.",
                        verbosity_level=1,
                    )
                ],
            ),
        ]
    ).model_dump_json()


# --- Tests for Extractor base ---


def test_filter_snippet_patterns():
    """Tests that the filter_snippet_patterns method correctly filters out chunks
    that match the provided skip patterns.
    """
    skip_snippets = SkipSnippets(
        data={"dummy filter": ".*test.*"},
    )
    extractor = Extractor(skip_snippets=skip_snippets)
    chunks = [
        (1, "This is a test log."),
        (2, "This is another line."),
        (3, "Another test line."),
    ]
    filtered_chunks = extractor.filter_snippet_patterns(chunks)
    assert len(filtered_chunks) == 1
    assert filtered_chunks[0] == (2, "This is another line.")


# --- Tests for DrainExtractor ---


def test_drain_extractor_call(simple_log):
    """Tests the basic functionality of the DrainExtractor to ensure it
    clusters and extracts log messages correctly.
    """
    extractor = DrainExtractor(max_clusters=2)
    result = extractor(simple_log)
    # The extractor should identify the two unique "An error occurred" lines
    assert len(result) > 0
    assert "An error occurred: file not found" in result[0][1]
    assert "An error occurred: permission denied." in result[1][1]


# --- Tests for CSGrepExtractor ---


def test_csgrep_extractor_call_success(monkeypatch, simple_log, csgrep_output):
    """Tests a successful run of the CSGrepExtractor, ensuring it correctly
    parses the JSON output from a mocked csgrep process.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stdout=csgrep_output, stderr="")
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    result = extractor(simple_log)

    assert len(result) == 2
    assert result[0] == (3, "An error occurred: file not found.")
    assert result[1] == (4, "An error occurred: permission denied.")
    mock_run.assert_called_once()


def test_csgrep_extractor_call_no_output(monkeypatch, simple_log):
    """Tests the CSGrepExtractor's behavior when the csgrep command
    produces no standard output.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    result = extractor(simple_log)

    assert len(result) == 0
    mock_run.assert_called_once()


def test_csgrep_extractor_call_error(monkeypatch, simple_log, caplog):
    """Tests the CSGrepExtractor's handling of a non-zero return code from
    the csgrep command. It should log a warning and return no messages.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="csgrep error")
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    result = extractor(simple_log)

    assert len(result) == 0
    assert "csgrep call resulted in an error" in caplog.text
    mock_run.assert_called_once()


def test_csgrep_extractor_timeout(monkeypatch, simple_log):
    """Tests that the CSGrepExtractor correctly handles a TimeoutExpired
    exception from the subprocess run command.
    """
    mock_run = MagicMock(side_effect=sp.TimeoutExpired(cmd="csgrep", timeout=1.0))
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    with pytest.raises(sp.TimeoutExpired):
        extractor(simple_log)
    mock_run.assert_called_once()


def test_csgrep_extractor_invalid_json(monkeypatch, simple_log):
    """Tests that the CSGrepExtractor raises a ValidationError when it
    receives malformed JSON from the csgrep command.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stdout="{invalid json}", stderr="")
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    with pytest.raises(Exception):
        extractor(simple_log)
    mock_run.assert_called_once()
