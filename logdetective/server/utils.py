from typing import List

import aiohttp
from fastapi import HTTPException

from logdetective.constants import SNIPPET_DELIMITER
from logdetective.server.config import LOG
from logdetective.server.exceptions import LogDetectiveConnectionError
from logdetective.server.models import AnalyzedSnippet, RatedSnippetAnalysis


def format_analyzed_snippets(snippets: list[AnalyzedSnippet]) -> str:
    """Format snippets for submission into staged prompt."""
    summary = f"\n{SNIPPET_DELIMITER}\n".join(
        [f"[{e.text}] at line [{e.line_number}]: [{e.explanation}]" for e in snippets]
    )
    return summary


def connection_error_giveup(details: dict) -> None:
    """Too many connection errors, give up.
    """
    LOG.error("Too many connection errors, giving up. %s", details["exception"])
    raise LogDetectiveConnectionError() from details["exception"]


def should_we_giveup(exc: aiohttp.ClientResponseError) -> bool:
    """From backoff's docs:

    > a function which accepts the exception and returns
    > a truthy value if the exception should not be retried
    """
    LOG.info("Should we give up on retrying error %s", exc)
    return exc.status < 400


def we_give_up(details: dict):
    """Retries didn't work (or we got a different exc)
    we give up and raise proper 500 for our API endpoint
    """
    LOG.error("Last exception: %s", details["exception"])
    LOG.error("Inference error: %s", details["args"])
    raise HTTPException(500, "Request to the inference API failed")


def select_relevance(snippet: AnalyzedSnippet) -> float:
    """Retrieve relevance value from structure, if there is one."""
    if not isinstance(snippet.explanation, RatedSnippetAnalysis):
        LOG.exception("Only rated snippets can be ordered by relevance.")
        raise ValueError
    return snippet.explanation.relevance


def select_line_number(explanation: AnalyzedSnippet) -> int:
    """Returns line number of original snippet."""
    return explanation.line_number


def filter_snippets(
    processed_snippets: List[AnalyzedSnippet], top_k: int
) -> List[AnalyzedSnippet]:
    """Filter snippets according to criteria in config while keeping them ordered by line number.
    If all snippets recieved the same score, return them all.
    AnalyzedSnippet objects must have `explanation` attribute set to `RatedSnippetAnalysis`,
    otherwise raise `ValueError`."""

    if top_k >= len(processed_snippets):
        LOG.warning(
            "The `top-k` parameter >= number of original snippets, skipping filtering."
        )
        return processed_snippets

    # Sorting invokes `select_relevance` which also tests if objects actually
    # have the score assigned. Otherwise it raises exception.
    processed_snippets = sorted(processed_snippets, key=select_relevance, reverse=True)

    # Check for failure mode when all snippets have
    # the same relevance. In such cases there is no point in filtering
    # and all snippets are returned.
    max_relevance = processed_snippets[0].explanation.relevance
    min_relevance = processed_snippets[-1].explanation.relevance

    LOG.info(
        "Analyzed snippets sorted. Max relevance: %d Min relevance: %e",
        max_relevance,
        min_relevance,
    )
    if max_relevance == min_relevance:
        LOG.warning("All snippets recieved the same rating. Filtering disabled.")
        return processed_snippets

    processed_snippets = processed_snippets[:top_k]

    # Re-sorting snippets by line number
    processed_snippets = sorted(processed_snippets, key=select_line_number)

    return processed_snippets


def construct_final_prompt(formatted_snippets: str, prompt_template: str) -> str:
    """Create final prompt from processed snippets and csgrep output, if it is available."""

    final_prompt = prompt_template.format(formatted_snippets)
    return final_prompt
