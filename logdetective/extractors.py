import os
import logging
import subprocess as sp
from typing import Tuple

from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from pydantic import ValidationError

from logdetective.utils import get_chunks, filter_snippet_patterns
from logdetective.models import SkipSnippets, CSGrepOutput

LOG = logging.getLogger("logdetective")


class Extractor:
    """Base extractor class."""

    def __init__(
        self,
        verbose: bool = False,
        skip_snippets: SkipSnippets = SkipSnippets({}),
        max_snippet_len: int = 2000,
    ):
        self.verbose = verbose
        self.skip_snippets = skip_snippets
        self.max_snippet_len = max_snippet_len

        if self.verbose:
            LOG.setLevel(logging.DEBUG)

    def __call__(self, log: str) -> list[Tuple[int, str]]:
        raise NotImplementedError

    def filter_snippet_patterns(
        self, chunks: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """Keep only chunks that don't match any of the excluded patterns"""
        chunks = [
            (_, chunk)
            for _, chunk in chunks
            if not filter_snippet_patterns(chunk, self.skip_snippets)
        ]
        return chunks


class DrainExtractor(Extractor):
    """A class that extracts information from logs using a template miner algorithm."""

    _clusters: list

    def __init__(
        self,
        verbose: bool = False,
        skip_snippets: SkipSnippets = SkipSnippets({}),
        max_snippet_len: int = 2000,
        max_clusters: int = 8,
    ):
        super().__init__(verbose, skip_snippets, max_snippet_len)
        config = TemplateMinerConfig()
        config.load(f"{os.path.dirname(__file__)}/drain3.ini")
        config.profiling_enabled = verbose
        config.drain_max_clusters = max_clusters
        self.miner = TemplateMiner(config=config)

    def __call__(self, log: str) -> list[Tuple[int, str]]:
        # Create chunks
        chunks = list(get_chunks(log, self.max_snippet_len))

        chunks = self.filter_snippet_patterns(chunks)

        # First pass to create clusters
        self._create_clusters(chunks=chunks)

        # Second pass, only matching lines with clusters,
        # to recover original text
        snippets = self._extract_messages(chunks=chunks)
        return snippets

    def _create_clusters(self, chunks: list[tuple[int, str]]):
        """First pass to create clusters"""
        for _, chunk in chunks:
            processed_chunk = self.miner.add_log_message(chunk)
            LOG.debug(processed_chunk)
        self._clusters = list(self.miner.drain.clusters)

    def _extract_messages(self, chunks: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Second pass with drain using patterns from the first,
        to extract matching lines and their numbers."""
        out = []

        for chunk_start, chunk in chunks:
            cluster = self.miner.match(chunk, "always")
            if cluster in self._clusters:
                out.append((chunk_start, chunk))
                self._clusters.remove(cluster)
        return out


class CSGrepExtractor(DrainExtractor):
    """Extract messages using csgrep
    This extractor is only effective at retrieving messages from GCC
    compiler and associated utilities, it is not capable of safely
    extracting other messages from the logs. Therefore, it must only
    be used together with the Drain based extractor."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        verbose: bool = False,
        skip_snippets: SkipSnippets = SkipSnippets({}),
        max_snippet_len: int = 2000,
        max_clusters: int = 8,
        csgrep_timeout: float = 1.0,
    ):
        super().__init__(verbose, skip_snippets, max_snippet_len, max_clusters)
        self.csgrep_timeout = csgrep_timeout

    def __call__(self, log: str) -> list[Tuple[int, str]]:
        """Extract error messages from log using csgrep"""
        chunks = []
        try:
            # We are not running binary in check mode, since csgrep
            # can produce many errors due to log file syntax
            result = sp.run(
                [
                    "csgrep",
                    "--event=error",
                    "--record-input-locations",
                    "--remove-duplicates",
                    "--mode=json",
                    "--quiet",
                ],
                input=log,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.csgrep_timeout,
            )
        except sp.TimeoutExpired as ex:
            LOG.exception("Exception encountered while parsing log with csgrep %s", ex)
            raise ex
        if result.returncode != 0:
            # This can happen even if `csgrep` managed to extract useful info.
            # Most commonly, when it encountered unexpected syntax in the log.
            LOG.warning("csgrep call resulted in an error")
            LOG.debug("csgrep error: `%s`", result.stderr)
        if not result.stdout:
            return []

        # Parse JSON output from csgrep
        try:
            report = CSGrepOutput.model_validate_json(result.stdout)
        except ValidationError as ex:
            LOG.exception("Exception encountered while parsing csgrep output %s", ex)
            raise ex
        # Single original error message can be split across multiple events.
        # Before returning, we will turn them back into single string.
        # We must also extract the original line number,
        # i.e. NOT the location of the issue in the source code ("event.line"),
        # but the location of message in the log ("event.input_line").
        chunks = [
            (
                d.events[0].input_line,
                "\n".join([e.message for e in d.events])
            )
            for d in report.defects if d.events  # skipping potential 0-event defects
        ]

        chunks = self.filter_snippet_patterns(chunks)
        LOG.info("Total %d messages extracted with csgrep", len(chunks))
        self._create_clusters(chunks=chunks)
        snippets = self._extract_messages(chunks=chunks)

        return snippets


class PythonTracebackExtractor(Extractor):
    """Extract Python exception tracebacks from logs using a line-scanning state machine."""

    _TB_START = "Traceback (most recent call last):"
    _CHAIN_CONT = (
        "During handling of the above exception, another exception occurred:",
        "The above exception was the direct cause of the following exception:",
    )
    _TRUNCATE_STR = "\n...<truncated>...\n"

    def __call__(self, log: str) -> list[Tuple[int, str]]:
        lines = log.splitlines()
        chunks = []
        current_idx = 0
        while current_idx < len(lines):
            if lines[current_idx] == self._TB_START:
                snippet_lines, next_idx = self._collect_traceback(lines, current_idx)
                text = "\n".join(snippet_lines)
                chunks.append((current_idx + 1, text))  # 1-indexed
                current_idx = next_idx
            else:
                current_idx += 1
        filtered_chunks = self.filter_snippet_patterns(chunks)
        truncated_chunks = list(map(self._truncate_long_traceback, filtered_chunks))
        LOG.info("Total %d python tracebacks messages", len(truncated_chunks))
        return truncated_chunks

    def _truncate_long_traceback(self, snippet: tuple[int, str]) -> list[tuple[int, str]]:
        """Shorten a snippet with text longer than `max_snippet_len`"""
        line_no, text = snippet
        if len(text) <= self.max_snippet_len:
            return snippet
        border = (self.max_snippet_len - len(self._TRUNCATE_STR)) // 2
        return (line_no, f"{text[:border]}{self._TRUNCATE_STR}{text[-border:]}")

    # In the following, by chaining, we mean:
    #   |Traceback ...
    #   |...
    #   |<blank line>
    #   |During handling of the above ...
    #   |<blank line>
    #   |Traceback ...
    #   |...

    # And by frames, we mean file-code references:
    #   |Traceback ...
    #   |  File "module1.py", line 42, in <module>  <- frame
    #   |    foo()
    #   |  File "module2.py" ...  <- another frame
    #   |    bar()
    #   |  ...
    #   |Exception: details of exception

    def _is_frame_line(self, line: str) -> bool:
        """Check if line is an indented traceback frame (File/code reference)."""
        return bool(line.startswith((" ", "\t")) and line.strip())

    def _is_chain_marker(self, line: str) -> bool:
        """Check if line marks a chained exception or new traceback."""
        return bool(line in self._CHAIN_CONT or line == self._TB_START)

    def _find_next_non_blank(self, lines: list[str], start_idx: int) -> int:
        """Find index of next non-blank line. Returns len(lines) if none found."""
        idx = start_idx
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    def _has_chain_continuation(self, lines: list[str], from_idx: int) -> int:
        """Check if a chain continuation follows after blank lines.

        Returns:
            Non-negative index of chain marker if found, otherwise -1
        """
        next_non_blank = self._find_next_non_blank(lines, from_idx)
        if next_non_blank < len(lines) and self._is_chain_marker(lines[next_non_blank]):
            return next_non_blank
        return -1

    def _collect_traceback(self, lines: list[str], start_idx: int) -> tuple[list[str], int]:
        """Collect all lines belonging to a traceback, including chained exceptions.

        Handles the state machine for parsing Python tracebacks:
        1. Indented frame lines (File/code references)
        2. Blank lines (may separate chained tracebacks)
        3. Chain continuation markers
        4. Exception type lines (non-indented, non-blank)

        Args:
            lines: All log lines
            start_idx: Index of "Traceback (most recent call last):" line

        Returns:
            Tuple of (collected lines, index after last collected line)
        """
        collected = [lines[start_idx]]
        current_idx = start_idx + 1

        while current_idx < len(lines):
            line = lines[current_idx]
            line = line.rstrip()

            # frame line (File/code reference)
            if self._is_frame_line(line):
                collected.append(line)
                current_idx += 1
                continue

            # blank line -> check if chain continues
            if not line.strip():
                chain_idx = self._has_chain_continuation(lines, current_idx + 1)
                if chain_idx >= 0:
                    current_idx = chain_idx
                    continue
                break

            # chain marker / new traceback
            if self._is_chain_marker(line):
                collected.append(line)
                current_idx += 1
                continue

            # exception type (non-indented, non-blank)
            collected.append(line)
            current_idx += 1

            # check if another exception follows after
            chain_idx = self._has_chain_continuation(lines, current_idx)
            if chain_idx >= 0:
                current_idx = chain_idx
                continue
            break

        return collected, current_idx
