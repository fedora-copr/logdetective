import os
import logging
from typing import Tuple

import drain3
from drain3.template_miner_config import TemplateMinerConfig

from logdetective.utils import get_chunks, filter_snippet_patterns
from logdetective.models import SkipSnippets

LOG = logging.getLogger("logdetective")


class DrainExtractor:
    """A class that extracts information from logs using a template miner algorithm."""

    def __init__(
        self,
        verbose: bool = False,
        context: bool = False,
        max_clusters=8,
        skip_snippets: SkipSnippets = SkipSnippets({}),
    ):
        config = TemplateMinerConfig()
        config.load(f"{os.path.dirname(__file__)}/drain3.ini")
        config.profiling_enabled = verbose
        config.drain_max_clusters = max_clusters
        self.miner = drain3.TemplateMiner(config=config)
        self.verbose = verbose
        self.context = context
        self.skip_snippets = skip_snippets

    def __call__(self, log: str) -> list[Tuple[int, str]]:
        out = []
        # Create chunks
        chunks = list(get_chunks(log))
        # Keep only chunks that don't match any of the excluded patterns
        chunks = [
            (_, chunk)
            for _, chunk in chunks
            if not filter_snippet_patterns(chunk, self.skip_snippets)
        ]
        # First pass create clusters
        for _, chunk in chunks:
            processed_chunk = self.miner.add_log_message(chunk)
            LOG.debug(processed_chunk)
        clusters = list(self.miner.drain.clusters)
        # Second pass, only matching lines with clusters,
        # to recover original text
        for chunk_start, chunk in chunks:
            cluster = self.miner.match(chunk, "always")
            if cluster in clusters:
                out.append((chunk_start, chunk))
                clusters.remove(cluster)
        return out
