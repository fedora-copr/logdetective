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
    AnalyzedSnippet,
    InferenceConfig,
    Explanation,
    StagedResponse,
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
    url_path: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    stream: bool,
    inference_cfg: InferenceConfig = SERVER_CONFIG.inference,
) -> Any:
    """Send request to an API endpoint. Verifying successful request unless
    the using the stream response.

    url_path: The endpoint path to query. (e.g. "/v1/chat/completions"). It should
    not include the scheme and netloc of the URL, which is stored in the
    InferenceConfig.
    data:
    headers:
    stream:
    inference_cfg: An InferenceConfig object containing the URL, max_tokens
    and other relevant configuration for talking to an inference server.
    """
    async with inference_cfg.get_limiter():
        LOG.debug("async request %s headers=%s data=%s", url_path, headers, data)
        session = inference_cfg.get_http_session()

        if inference_cfg.api_token:
            headers["Authorization"] = f"Bearer {inference_cfg.api_token}"

        response = await session.post(
            url_path,
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
            LOG.error(
                "Error encountered while parsing llama server response: %s", ex
            )
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
async def submit_text(
    text: str,
    inference_cfg: InferenceConfig,
    stream: bool = False,
) -> Union[Explanation, StreamReader]:
    """Submit prompt to LLM.
    inference_cfg: The configuration section from the config.json representing
    the relevant inference server for this request.
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Analyzing the text")

    headers = {"Content-Type": "application/json"}

    if SERVER_CONFIG.inference.api_token:
        headers["Authorization"] = f"Bearer {SERVER_CONFIG.inference.api_token}"

    LOG.info("Submitting to /v1/chat/completions endpoint")

    data = {
        "messages": [
            {
                "role": "user",
                "content": text,
            }
        ],
        "max_tokens": inference_cfg.max_tokens,
        "logprobs": inference_cfg.log_probs,
        "stream": stream,
        "model": inference_cfg.model,
        "temperature": inference_cfg.temperature,
    }

    response = await submit_to_llm_endpoint(
        "/v1/chat/completions",
        data,
        headers,
        inference_cfg=inference_cfg,
        stream=stream,
    )

    if stream:
        return response
    return Explanation(
        text=response["choices"][0]["message"]["content"],
        logprobs=response["choices"][0]["logprobs"]["content"],
    )


async def perform_staged_analysis(log_text: str) -> StagedResponse:
    """Submit the log file snippets to the LLM and retrieve their results"""
    log_summary = mine_logs(log_text)

    # Process snippets asynchronously
    awaitables = [
        submit_text(
            PROMPT_CONFIG.snippet_prompt_template.format(s),
            inference_cfg=SERVER_CONFIG.snippet_inference,
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
        final_prompt,
        inference_cfg=SERVER_CONFIG.inference,
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
