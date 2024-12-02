import os
import logging

import drain3
from drain3.template_miner_config import TemplateMinerConfig
from llama_cpp import Llama, LlamaGrammar

from logdetective.constants import SUMMARIZE_PROMPT_TEMPLATE
from logdetective.utils import get_chunks

LOG = logging.getLogger("logdetective")


class LLMExtractor:
    """
    A class that extracts relevant information from logs using a language model.
    """
    def __init__(self, model: Llama, n_lines: int = 2):
        self.model = model
        self.n_lines = n_lines
        self.grammar = LlamaGrammar.from_string(
            "root ::= (\"Yes\" | \"No\")", verbose=False)

    def __call__(self, log: str, n_lines: int = 2, neighbors: bool = False) -> list[str]:
        chunks = self.rate_chunks(log)
        out = self.create_extract(chunks, neighbors)
        return out

    def rate_chunks(self, log: str) -> list[tuple]:
        """Scan log by the model and store results.

        :param log: log file content
        """
        results = []
        log_lines = log.split("\n")

        for i in range(0, len(log_lines), self.n_lines):
            block = '\n'.join(log_lines[i:i + self.n_lines])
            prompt = SUMMARIZE_PROMPT_TEMPLATE.format(log)
            out = self.model(prompt, max_tokens=7, grammar=self.grammar)
            out = f"{out['choices'][0]['text']}\n"
            results.append((block, out))

        return results

    def create_extract(self, chunks: list[tuple], neighbors: bool = False) -> list[str]:
        """Extract interesting chunks from the model processing.
        """
        interesting = []
        summary = []
        # pylint: disable=consider-using-enumerate
        for i in range(len(chunks)):
            if chunks[i][1].startswith("Yes"):
                interesting.append(i)
                if neighbors:
                    interesting.extend([max(i - 1, 0), min(i + 1, len(chunks) - 1)])

        interesting = set(interesting)

        for i in interesting:
            summary.append(chunks[i][0])

        return summary


class DrainExtractor:
    """A class that extracts information from logs using a template miner algorithm.
    """
    def __init__(self, verbose: bool = False, context: bool = False, max_clusters=8):
        config = TemplateMinerConfig()
        config.load(f"{os.path.dirname(__file__)}/drain3.ini")
        config.profiling_enabled = verbose
        config.drain_max_clusters = max_clusters
        self.miner = drain3.TemplateMiner(config=config)
        self.verbose = verbose
        self.context = context

    def __call__(self, log: str) -> list[str]:
        out = []
        for chunk in get_chunks(log):
            processed_line = self.miner.add_log_message(chunk)
            LOG.debug(processed_line)
        sorted_clusters = sorted(self.miner.drain.clusters, key=lambda it: it.size, reverse=True)
        for chunk in get_chunks(log):
            cluster = self.miner.match(chunk, "always")
            if cluster in sorted_clusters:
                out.append(chunk)
                sorted_clusters.remove(cluster)
        return out
