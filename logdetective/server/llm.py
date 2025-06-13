import os
import asyncio
import random
from typing import List, Tuple, Union, Dict

import backoff
from fastapi import HTTPException

import aiohttp
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from logdetective.constants import SNIPPET_DELIMITER
from logdetective.extractors import DrainExtractor
from logdetective.utils import (
    compute_certainty,
    prompt_to_messages,
)
from logdetective.server.config import LOG, SERVER_CONFIG, PROMPT_CONFIG, CLIENT
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
    messages: List[Dict[str, str]],
    inference_cfg: InferenceConfig,
    stream: bool = False,
) -> Union[Explanation, AsyncStream[ChatCompletionChunk]]:
    """Submit prompt to LLM.
    inference_cfg: The configuration section from the config.json representing
    the relevant inference server for this request.
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Analyzing the text")

    LOG.info("Submitting to /v1/chat/completions endpoint")

    async with inference_cfg.get_limiter():
        response = await CLIENT.chat.completions.create(
            messages=messages,
            max_tokens=inference_cfg.max_tokens,
            logprobs=inference_cfg.log_probs,
            stream=stream,
            model=inference_cfg.model,
            temperature=inference_cfg.temperature,
        )

    if isinstance(response, AsyncStream):
        return response
    if not response.choices[0].message.content:
        LOG.error("No response content recieved from %s", inference_cfg.url)
        raise RuntimeError()
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        logprobs = [e.to_dict() for e in response.choices[0].logprobs.content]
    else:
        logprobs = None

    return Explanation(
        text=response.choices[0].message.content,
        logprobs=logprobs,
    )


async def perform_staged_analysis(log_text: str) -> StagedResponse:
    """Submit the log file snippets to the LLM and retrieve their results"""
    log_summary = mine_logs(log_text)

    # Process snippets asynchronously
    awaitables = [
        submit_text(
            prompt_to_messages(
                PROMPT_CONFIG.snippet_prompt_template.format(s),
                PROMPT_CONFIG.snippet_system_prompt,
                SERVER_CONFIG.inference.system_role,
                SERVER_CONFIG.inference.user_role,
            ),
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
    messages = prompt_to_messages(
        final_prompt,
        PROMPT_CONFIG.staged_system_prompt,
        SERVER_CONFIG.inference.system_role,
        SERVER_CONFIG.inference.user_role,
    )
    final_analysis = await submit_text(
        messages,
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
