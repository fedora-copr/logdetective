import subprocess as sp
from unittest.mock import MagicMock
import pytest

from logdetective.constants import TRUNCATED
from logdetective.extractors import (
    Extractor,
    DrainExtractor,
    CSGrepExtractor,
    PythonTracebackExtractor,
)
from logdetective.models import SkipSnippets

from tests.base.test_helpers import (
    test_filter_patterns,
    test_snippets_filtering,
    simple_log,
    csgrep_output_simple,
    siril_log_snippet,
    csgrep_output_siril,
    dolphin_emu_log_snippet,
    csgrep_output_dolphin_emu,
    package_unavailable_log,
    DNF_PACKAGE_UNAVAILABLE_EXPECTED_SNIPPETS,
    SIMPLE_TRACEBACK_LOG,
    CHAINED_TRACEBACK_LOG,
    LONGER_TRACEBACK_LOG,
    LONG_CHAIN_TRACEBACK_LOG,
    PYTHON_SIMPLE_TB,
    PYTHON_SIMPLE_CHAINED_TB,
    PYTHON_LONGER_TB,
    PYTHON_LONG_CHAIN_TB,
)


# --- Tests for Extractor base ---


def test_filter_snippet_patterns_simple():
    """Tests that the filter_snippet_patterns method correctly filters out chunks
    that match the provided skip patterns.
    """
    skip_snippets = SkipSnippets(
        snippet_patterns={"dummy filter": {"pattern": ".*test.*"}},
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


def test_filter_snippet_patterns_complex():
    """Test snippet filtering via Extractor."""
    extractor = Extractor(skip_snippets=SkipSnippets(snippet_patterns=test_filter_patterns))
    chunks = [(idx, text) for idx, (text, _) in enumerate(test_snippets_filtering)]
    result = extractor.filter_snippet_patterns(chunks)
    kept_texts = [text for _, text in result]

    for snippet, skip in test_snippets_filtering:
        if skip:
            assert snippet not in kept_texts
        else:
            assert snippet in kept_texts


def test_filter_snippet_patterns_scoped_simple():
    """File-scoped patterns apply only to exact filename matches."""
    skip_snippets = SkipSnippets(snippet_patterns={
        "global_pattern": {"pattern": "^GLOBAL"},
        "scoped_pattern": {"pattern": "^SCOPED", "files": ["backend.log", "app.log"]},
    })

    extractor = Extractor(skip_snippets=skip_snippets)

    # Global pattern fires regardless of filename
    assert not extractor.filter_snippet_patterns([(1, "GLOBAL line")], "build.log")
    assert not extractor.filter_snippet_patterns([(1, "GLOBAL line")], "other.log")

    # Scoped pattern fires only for exact filename matches (filtered out -> returns empty list)
    assert extractor.filter_snippet_patterns([(1, "SCOPED line")], "build.log")
    assert not extractor.filter_snippet_patterns([(1, "SCOPED line")], "backend.log")
    assert not extractor.filter_snippet_patterns([(1, "SCOPED line")], "app.log")
    assert extractor.filter_snippet_patterns([(1, "SCOPED line")], "unknown.log")


def test_filter_snippet_patterns_scoped_complex():
    """File-scoped patterns apply only to exact filename matches."""
    skip_snippets = SkipSnippets(
        snippet_patterns={
            "mock_output_specific": {
                "pattern": "INFO:",
                "files": ["mock_output.log"],
            },
            "build_log_specific": {
                "pattern": ".*Child return code was: 0",
                "files": ["build.log"],
            },
            "applied_everywhere": {
                "pattern": ".* OK "
            },
        }
    )
    extractor = Extractor(skip_snippets=skip_snippets)
    mock_chunks = [
        (1, "INFO: We want to skip this line in mock_output only"),
        (2, "Skip this in build log only: Child return code was: 0"),
        (3, "Build log passed - OK - this line is not interesting and is skipped everywhere."),
        (4, "this line is interesting and contains the explanation")
    ]

    build_chunks = [
        (1, "INFO: We want to skip this line in mock_output only"),
        (2, "Skip this in build log only: Child return code was: 0"),
        (3, "Build log passed - OK - this line is not interesting and is skipped everywhere."),
    ]

    filtered_mock_chunks = extractor.filter_snippet_patterns(mock_chunks, "mock_output.log")
    filtered_build_chunks = extractor.filter_snippet_patterns(build_chunks, "build.log")

    assert len(filtered_mock_chunks) == 2
    assert filtered_mock_chunks[0] == (
        2, "Skip this in build log only: Child return code was: 0"
    )
    assert filtered_mock_chunks[1] == (
        4, "this line is interesting and contains the explanation"
    )

    assert len(filtered_build_chunks) == 1
    assert filtered_build_chunks[0] == (
        1, "INFO: We want to skip this line in mock_output only"
    )


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


@pytest.mark.parametrize("max_chunk_len", [100, 120, 150])
def test_get_chunks_max_length(simple_log, max_chunk_len):
    """Test that maximum length of chunks is properly enforced"""
    log = "".join(simple_log)
    extractor = DrainExtractor(max_snippet_len=max_chunk_len)
    chunks = list(extractor._get_chunks(log))

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
    extractor = DrainExtractor(max_snippet_len=max_chunk_len)
    with pytest.raises(ValueError):
        list(extractor._get_chunks(log))


def test_empty_log_creates_no_chunks():

    log = ""
    extractor = DrainExtractor()
    chunks = list(extractor._get_chunks(log))

    assert len(chunks) == 0


def test_leading_whitespace_chunks(simple_log):

    log = " ".join(simple_log)
    extractor = DrainExtractor()
    chunks = list(extractor._get_chunks(log))

    # Number of chunks must be <= number of original lines
    assert len(chunks) <= len(simple_log)

    for chunk in chunks[1:]:
        assert chunk[1].startswith(" ")

    # All chunks must have unique lines
    lines = set(c[0] for c in chunks)
    assert len(lines) == len(chunks)

    # Last chunk must not be empty
    assert len(chunks[-1][1]) > 0


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


# --- Tests for PythonTracebackExtractor ---


def test_python_tb_single():
    """Single traceback embedded in surrounding log output."""
    extractor = PythonTracebackExtractor()
    results = extractor(SIMPLE_TRACEBACK_LOG)

    assert len(results) == 1
    line_no, text = results[0]
    assert line_no == 3
    assert PYTHON_SIMPLE_TB in text
    assert "\n\n" not in text


def test_python_tb_chained():
    """Chained tracebacks are coalesced into a single snippet."""
    extractor = PythonTracebackExtractor()
    results = extractor(CHAINED_TRACEBACK_LOG)

    assert len(results) == 1
    line_no, text = results[0]
    assert line_no == 13
    assert not text.startswith("\n")
    assert not text.endswith("\n")
    assert PYTHON_SIMPLE_CHAINED_TB.replace("\n\n", "\n") in text
    assert "\n\n" not in text


def test_python_tb_multiple_independent():
    """Two separate tracebacks produce two independent snippets."""
    log = SIMPLE_TRACEBACK_LOG + "\n" + SIMPLE_TRACEBACK_LOG
    extractor = PythonTracebackExtractor()
    results = extractor(log)

    assert len(results) == 2
    assert results[0][0] == 3
    assert results[1][0] == 15
    assert all(PYTHON_SIMPLE_TB in snip for _, snip in results)
    assert all("\n\n" not in snip for _, snip in results)


def test_python_tb_empty():
    """Log with no tracebacks returns empty list."""
    extractor = PythonTracebackExtractor()
    results = extractor("[INFO] All good\n[INFO] Done\n")

    assert not results


def test_python_tb_max_snippet_len():
    """Traceback exceeding max_snippet_len is truncated with ...<truncated>..."""
    extractor = PythonTracebackExtractor(max_snippet_len=50)
    results = extractor(SIMPLE_TRACEBACK_LOG)

    assert len(results) == 1
    line_no, text = results[0]
    assert line_no == 3
    assert "...<truncated>..." in text
    assert "\n\n" not in text


def test_python_tb_skip_patterns():
    """Skip pattern matching removes tracebacks whose text matches the pattern."""
    skip = SkipSnippets(
        snippet_patterns={
            "skip_file_not_found": {"pattern": ".*FileNotFoundError.*"}
        }
    )
    extractor = PythonTracebackExtractor(skip_snippets=skip)
    results = extractor(SIMPLE_TRACEBACK_LOG)

    assert len(results) == 0


def test_python_tb_line_numbers():
    """Line numbers correspond to the 1-indexed position of the Traceback header in the log."""
    log = "line1\nline2\n" + CHAINED_TRACEBACK_LOG
    extractor = PythonTracebackExtractor()
    results = extractor(log)

    assert len(results) == 1
    line_no, text = results[0]
    assert line_no == 15
    assert "\n\n" not in text


def test_python_tb_long_stack():
    """Long call stack (5+ frames) is captured completely."""
    extractor = PythonTracebackExtractor()
    results = extractor(LONGER_TRACEBACK_LOG)

    assert len(results) == 1
    line_no, text = results[0]
    assert line_no == 4
    assert PYTHON_LONGER_TB in text
    assert "\n\n" not in text


def test_python_tb_chained_3():
    """Three-level exception chain is captured as single snippet."""
    extractor = PythonTracebackExtractor()
    results = extractor(LONG_CHAIN_TRACEBACK_LOG)

    assert len(results) == 1
    line_no, text = results[0]
    assert line_no == 5
    assert not text.startswith("\n")
    assert not text.endswith("\n")
    assert PYTHON_LONG_CHAIN_TB.replace("\n\n", "\n") in text


def test_python_tb_all():
    """Four concatenated logs, each with 1 separate traceback."""
    long_log: str = "\n".join([
        SIMPLE_TRACEBACK_LOG,
        LONGER_TRACEBACK_LOG,
        CHAINED_TRACEBACK_LOG,
        LONG_CHAIN_TRACEBACK_LOG,
    ])
    extractor = PythonTracebackExtractor()
    results = extractor(long_log)

    assert len(results) == 4
    assert all("\n\n" not in snip for _, snip in results)

    assert results[0][0] == 3
    assert PYTHON_SIMPLE_TB in results[0][1]

    assert results[1][0] == 16
    assert PYTHON_LONGER_TB in results[1][1]

    assert results[2][0] == 44
    assert PYTHON_SIMPLE_CHAINED_TB.replace("\n\n", "\n") in results[2][1]

    assert results[3][0] == 64
    assert PYTHON_LONG_CHAIN_TB.replace("\n\n", "\n") in results[3][1]
