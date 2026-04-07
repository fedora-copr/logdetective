import subprocess as sp
from unittest.mock import MagicMock
import pytest

from logdetective.models import SkipSnippets
from logdetective.extractors import DrainExtractor, CSGrepExtractor, Extractor

from tests.base.test_helpers import (
    simple_log,
    csgrep_output_simple,
    siril_log_snippet,
    csgrep_output_siril,
    dolphin_emu_log_snippet,
    csgrep_output_dolphin_emu,
    package_unavailable_log,
    DNF_PACKAGE_UNAVAILABLE_EXPECTED_SNIPPETS,
)


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


def test_drain_extractor_call(simple_log: list[str]):
    """Tests the basic functionality of the DrainExtractor to ensure it
    clusters and extracts log messages correctly.
    """
    extractor = DrainExtractor(max_clusters=2)
    result = extractor("".join(simple_log))
    # The extractor should identify the two unique "An error occurred" lines
    assert len(result) == 2
    messages = [msg for _, msg in result]
    # Two of the longest messages must be included in the results
    assert simple_log[6].strip() in messages
    assert simple_log[7].strip() in messages


def test_drain_extractor_package_unavailable_log(package_unavailable_log):
    """Tests DrainExtractor to ensure with DNF package unavailable message."""
    extractor = DrainExtractor(max_clusters=4)
    result = extractor(package_unavailable_log)
    # The extractor should identify the two unique "An error occurred" lines
    assert len(result) == 4
    messages = [e[1] for e in result]
    for chunk in DNF_PACKAGE_UNAVAILABLE_EXPECTED_SNIPPETS:
        assert chunk in messages


# --- Tests for CSGrepExtractor ---


def test_csgrep_extractor_call_success(monkeypatch, simple_log: list[str], csgrep_output_simple):
    """Tests a successful run of the CSGrepExtractor, ensuring it correctly
    parses the JSON output from a mocked csgrep process.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(returncode=0, stdout=csgrep_output_simple, stderr="")
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    snippets = extractor("".join(simple_log))

    assert len(snippets) == 2
    assert snippets[0] == (3, "An error occurred: file not found.")
    assert snippets[1] == (4, "An error occurred: permission denied.")
    mock_run.assert_called_once()


def test_csgrep_extractor_call_success_model_output(
    monkeypatch,
    siril_log_snippet,
    csgrep_output_siril
):
    """
    Test a snippet from a failed build log file, mocking csgrep result as as validated model.
    Note: Actually running csgrep would require us to run these tests
    in a container setup, so we mock the results.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=csgrep_output_siril,
        stderr=""
    )
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    snippets = extractor(siril_log_snippet)

    assert len(snippets) == 1
    line, text = snippets[0]
    assert line == 2
    assert "ld returned 1 exit status" in text
    mock_run.assert_called_once()


def test_csgrep_extractor_call_success_raw_output(
    monkeypatch,
    dolphin_emu_log_snippet,
    csgrep_output_dolphin_emu
):
    """
    Test a snippet from a failed build log file, mocking csgrep result as raw string.
    Note: Actually running csgrep would require us to run these tests
    in a container setup, so we mock the results.
    """
    mock_run = MagicMock()
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=csgrep_output_dolphin_emu,
        stderr=""
    )
    monkeypatch.setattr(sp, "run", mock_run)

    extractor = CSGrepExtractor()
    snippets = extractor(dolphin_emu_log_snippet)

    assert len(snippets) == 1
    line, text = snippets[0]
    assert line == 1
    assert "expected primary-expression before > token" in text
    assert "static_assert(fmt::detail::is_compile_string<S>::value)" in text
    mock_run.assert_called_once()


def _get_csgrep_version() -> tuple[int]:
    try:
        csgrep_ver = tuple(
            int(x) for x in
            sp.run(
                [
                    "csgrep",
                    "--version",
                ],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.split()[-1].split(".")
        )
    except (FileNotFoundError, IndexError, ValueError):
        csgrep_ver = (0, 0, 0)
    return csgrep_ver


@pytest.mark.skipif(_get_csgrep_version() < (3, 5, 7), reason="requires csgrep >= 3.5.7")
def test_csgrep_extractor_no_mock(dolphin_emu_log_snippet, siril_log_snippet):
    extractor = CSGrepExtractor()
    snippets = extractor(siril_log_snippet)

    assert len(snippets) == 1
    line, text = snippets[0]
    assert line == 2
    assert "ld returned 1 exit status" in text

    extractor = CSGrepExtractor()
    snippets = extractor(dolphin_emu_log_snippet)

    assert len(snippets) == 1
    line, text = snippets[0]
    assert line == 1
    assert "expected primary-expression before > token" in text
    assert "static_assert(fmt::detail::is_compile_string<S>::value)" in text


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
