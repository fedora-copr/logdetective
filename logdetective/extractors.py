import os
import logging
import subprocess as sp
from typing import Tuple

import drain3
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
        self.miner = drain3.TemplateMiner(config=config)

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

    def __init__(
        self,
        verbose: bool = False,
        skip_snippets: SkipSnippets = SkipSnippets({}),
        max_snippet_len: int = 2000,
        max_clusters: int = 8,
    ):
        super().__init__(verbose, skip_snippets, max_snippet_len, max_clusters)

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
                    "--remove-duplicates",
                    "--mode=json",
                    "--quiet",
                ],
                input=log,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
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
            LOG.exception("Exception encountered while parsing csgrpe output %s", ex)
            raise ex
        for defect in report.defects:
            # Single original error message can be split across multiple events
            # before returning, we will turn them back into single string.
            # We must also extract the original line number.
            # Line number is NOT location of message in the log, but location of
            # the issue in source, we can't really mix the two, so we'll set it to `0`.

            chunks.append((0, "\n".join([event.message for event in defect.events])))

        chunks = self.filter_snippet_patterns(chunks)
        LOG.info("Total %d messages extracted with csgrep", len(chunks))
        self._create_clusters(chunks=chunks)
        snippets = self._extract_messages(chunks=chunks)

        return snippets
