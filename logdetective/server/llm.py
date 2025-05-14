import os
import asyncio
import json
import random
from typing import List, Tuple, Dict, Any, Union

import backoff
from aiohttp import StreamReader
from fastapi import HTTPException

import aiohttp

from logdetective.constants import SNIPPET_DELIMITER
from logdetective.extractors import DrainExtractor
from logdetective.utils import (
    compute_certainty,
)
from logdetective.server.config import LOG, SERVER_CONFIG, PROMPT_CONFIG
from logdetective.server.models import (
    StagedResponse,
    Explanation,
    AnalyzedSnippet,
)


LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)


def format_analyzed_snippets(snippets: list[AnalyzedSnippet]) -> str:
    """Format snippets for submission into staged prompt."""
    summary = f"\n{SNIPPET_DELIMITER}\n".join(
        [
            f"[{e.text}] at line [{e.line_number}]: [{e.explanation.text}]"
            for e in snippets
        ]
    )
    return summary


def mine_logs(log: str) -> List[Tuple[int, str]]:
    """Extract snippets from log text"""
    extractor = DrainExtractor(
        verbose=True, context=True, max_clusters=SERVER_CONFIG.extractor.max_clusters
    )

    LOG.info("Getting summary")
    log_summary = extractor(log)

    ratio = len(log_summary) / len(log.split("\n"))
    LOG.debug("Log summary: \n %s", log_summary)
    LOG.info("Compression ratio: %s", ratio)

    return log_summary


async def submit_to_llm_endpoint(
    http: aiohttp.ClientSession,
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    stream: bool,
) -> Any:
    """Send request to selected API endpoint. Verifying successful request unless
    the using the stream response.

    url:
    data:
    headers:
    stream:
    """
    async with SERVER_CONFIG.inference.get_limiter():
        LOG.debug("async request %s headers=%s data=%s", url, headers, data)
        response = await http.post(
            url,
            headers=headers,
            # we need to use the `json=` parameter here and let aiohttp
            # handle the json-encoding
            json=data,
            timeout=int(LLM_CPP_SERVER_TIMEOUT),
            # Docs says chunked takes int, but:
            #   DeprecationWarning: Chunk size is deprecated #1615
            # So let's make sure we either put True or None here
            chunked=True if stream else None,
            raise_for_status=True,
        )
        if stream:
            return response
        try:
            return json.loads(await response.text())
        except UnicodeDecodeError as ex:
            LOG.error("Error encountered while parsing llama server response: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't parse the response.\nError: {ex}\nData: {response.text}",
            ) from ex


def should_we_giveup(exc: aiohttp.ClientResponseError) -> bool:
    """
    From backoff's docs:

    > a function which accepts the exception and returns
    > a truthy value if the exception should not be retried
    """
    LOG.info("Should we give up on retrying error %s", exc)
    return exc.status < 400


def we_give_up(details: backoff._typing.Details):
    """
    retries didn't work (or we got a different exc)
    we give up and raise proper 500 for our API endpoint
    """
    LOG.error("Last exception: %s", details["exception"])
    LOG.error("Inference error: %s", details["args"])
    raise HTTPException(500, "Request to the inference API failed")


@backoff.on_exception(
    lambda: backoff.constant([10, 30, 120]),
    aiohttp.ClientResponseError,
    max_tries=4,  # 4 tries and 3 retries
    jitter=lambda wait_gen_value: random.uniform(wait_gen_value, wait_gen_value + 30),
    giveup=should_we_giveup,
    raise_on_giveup=False,
    on_giveup=we_give_up,
)
async def submit_text(  # pylint: disable=R0913,R0917
    http: aiohttp.ClientSession,
    text: str,
    max_tokens: int = -1,
    log_probs: int = 1,
    stream: bool = False,
    model: str = "default-model",
) -> Explanation:
    """Submit prompt to LLM using a selected endpoint.
    max_tokens: number of tokens to be produces, 0 indicates run until encountering EOS
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Analyzing the text")

    headers = {"Content-Type": "application/json"}

    if SERVER_CONFIG.inference.api_token:
        headers["Authorization"] = f"Bearer {SERVER_CONFIG.inference.api_token}"

    if SERVER_CONFIG.inference.api_endpoint == "/chat/completions":
        return await submit_text_chat_completions(
            http, text, headers, max_tokens, log_probs > 0, stream, model
        )
    return await submit_text_completions(
        http, text, headers, max_tokens, log_probs, stream, model
    )


async def submit_text_completions(  # pylint: disable=R0913,R0917
    http: aiohttp.ClientSession,
    text: str,
    headers: dict,
    max_tokens: int = -1,
    log_probs: int = 1,
    stream: bool = False,
    model: str = "default-model",
) -> Explanation:
    """Submit prompt to OpenAI API completions endpoint.
    max_tokens: number of tokens to be produces, 0 indicates run until encountering EOS
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Submitting to /v1/completions endpoint")
    data = {
        "prompt": text,
        "max_tokens": max_tokens,
        "logprobs": log_probs,
        "stream": stream,
        "model": model,
        "temperature": SERVER_CONFIG.inference.temperature,
    }

    response = await submit_to_llm_endpoint(
        http,
        f"{SERVER_CONFIG.inference.url}/v1/completions",
        data,
        headers,
        stream,
    )

    return Explanation(
        text=response["choices"][0]["text"], logprobs=response["choices"][0]["logprobs"]
    )


async def submit_text_chat_completions(  # pylint: disable=R0913,R0917
    http: aiohttp.ClientSession,
    text: str,
    headers: dict,
    max_tokens: int = -1,
    log_probs: int = 1,
    stream: bool = False,
    model: str = "default-model",
) -> Union[Explanation, StreamReader]:
    """Submit prompt to OpenAI API /chat/completions endpoint.
    max_tokens: number of tokens to be produces, 0 indicates run until encountering EOS
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Submitting to /v1/chat/completions endpoint")

    data = {
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ],
        "max_tokens": max_tokens,
        "logprobs": log_probs,
        "stream": stream,
        "model": model,
        "temperature": SERVER_CONFIG.inference.temperature,
    }

    response = await submit_to_llm_endpoint(
        http,
        f"{SERVER_CONFIG.inference.url}/v1/chat/completions",
        data,
        headers,
        stream,
    )

    if stream:
        return response
    return Explanation(
        text=response["choices"][0]["message"]["content"],
        logprobs=response["choices"][0]["logprobs"]["content"],
    )


async def perform_staged_analysis(
    http: aiohttp.ClientSession, log_text: str
) -> StagedResponse:
    """Submit the log file snippets to the LLM and retrieve their results"""
    log_summary = mine_logs(log_text)

    # Process snippets asynchronously
    awaitables = [
        submit_text(
            http,
            PROMPT_CONFIG.snippet_prompt_template.format(s),
            model=SERVER_CONFIG.inference.model,
            max_tokens=SERVER_CONFIG.inference.max_tokens,
        )
        for s in log_summary
    ]
    analyzed_snippets = await asyncio.gather(*awaitables)

    analyzed_snippets = [
        AnalyzedSnippet(line_number=e[0][0], text=e[0][1], explanation=e[1])
        for e in zip(log_summary, analyzed_snippets)
    ]
    final_prompt = PROMPT_CONFIG.prompt_template_staged.format(
        format_analyzed_snippets(analyzed_snippets)
    )

    final_analysis = await submit_text(
        http,
        final_prompt,
        model=SERVER_CONFIG.inference.model,
        max_tokens=SERVER_CONFIG.inference.max_tokens,
    )

    certainty = 0

    if final_analysis.logprobs:
        try:
            certainty = compute_certainty(final_analysis.logprobs)
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n"
                f"{final_analysis.logprobs}",
            ) from ex

    return StagedResponse(
        explanation=final_analysis,
        snippets=analyzed_snippets,
        response_certainty=certainty,
    )
