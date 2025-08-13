import os
import asyncio
import random
import time
from typing import List, Tuple, Dict

import backoff
from fastapi import HTTPException
from pydantic import ValidationError

import aiohttp
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from logdetective.utils import (
    compute_certainty,
    prompt_to_messages,
    format_snippets,
    mine_logs,
)
from logdetective.server.config import (
    LOG,
    SERVER_CONFIG,
    PROMPT_CONFIG,
    CLIENT,
)
from logdetective.server.models import (
    AnalyzedSnippet,
    InferenceConfig,
    Explanation,
    StagedResponse,
    SnippetAnalysis,
    RatedSnippetAnalysis,
    Response,
)
from logdetective.server.utils import (
    format_analyzed_snippets,
    should_we_giveup,
    we_give_up,
    filter_snippets,
    construct_final_prompt,
)


LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)


@backoff.on_exception(
    lambda: backoff.constant([10, 30, 120]),
    aiohttp.ClientResponseError,
    max_tries=4,  # 4 tries and 3 retries
    jitter=lambda wait_gen_value: random.uniform(wait_gen_value, wait_gen_value + 30),
    giveup=should_we_giveup,
    raise_on_giveup=False,
    on_giveup=we_give_up,
)
async def call_llm(
    messages: List[Dict[str, str]],
    inference_cfg: InferenceConfig,
    stream: bool = False,
    structured_output: dict | None = None,
) -> Explanation:
    """Submit prompt to LLM.
    inference_cfg: The configuration section from the config.json representing
    the relevant inference server for this request.
    """
    LOG.info("Analyzing the text")

    LOG.info("Submitting to /v1/chat/completions endpoint")

    kwargs = {}

    # OpenAI API does not guarantee that the behavior for parameter set to `None`
    # and parameter not given at all is the same.
    # We build a dictionary of parameters based on the configuration.
    if inference_cfg.log_probs:
        LOG.info("Requesting log probabilities from LLM")
        kwargs["logprobs"] = inference_cfg.log_probs
    if structured_output:
        LOG.info("Requesting structured output from LLM")
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "rated-snippet-analysis",
                "schema": structured_output,
            },
        }
        kwargs["response_format"] = response_format

    async with inference_cfg.get_limiter():
        response = await CLIENT.chat.completions.create(
            messages=messages,
            max_tokens=inference_cfg.max_tokens,
            stream=stream,
            model=inference_cfg.model,
            temperature=inference_cfg.temperature,
            **kwargs,
        )

    if not response.choices[0].message.content:
        LOG.error("No response content recieved from %s", inference_cfg.url)
        raise RuntimeError()

    message_content = response.choices[0].message.content

    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        logprobs = [e.to_dict() for e in response.choices[0].logprobs.content]
    else:
        logprobs = None

    return Explanation(
        text=message_content,
        logprobs=logprobs,
    )


@backoff.on_exception(
    lambda: backoff.constant([10, 30, 120]),
    aiohttp.ClientResponseError,
    max_tries=4,  # 4 tries and 3 retries
    jitter=lambda wait_gen_value: random.uniform(wait_gen_value, wait_gen_value + 30),
    giveup=should_we_giveup,
    raise_on_giveup=False,
    on_giveup=we_give_up,
)
async def call_llm_stream(
    messages: List[Dict[str, str]],
    inference_cfg: InferenceConfig,
    stream: bool = False,
) -> AsyncStream[ChatCompletionChunk]:
    """Submit prompt to LLM and recieve stream of tokens as a result.
    inference_cfg: The configuration section from the config.json representing
    the relevant inference server for this request.
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

    return response


async def analyze_snippets(
    log_summary: List[Tuple[int, str]], structured_output: dict | None = None
) -> List[SnippetAnalysis | RatedSnippetAnalysis]:
    """Submit log file snippets to the LLM and gather results"""
    # Process snippets asynchronously
    awaitables = [
        call_llm(
            prompt_to_messages(
                PROMPT_CONFIG.snippet_prompt_template.format(s),
                PROMPT_CONFIG.snippet_system_prompt,
                SERVER_CONFIG.inference.system_role,
                SERVER_CONFIG.inference.user_role,
            ),
            inference_cfg=SERVER_CONFIG.snippet_inference,
            structured_output=structured_output,
        )
        for s in log_summary
    ]
    gathered_responses = await asyncio.gather(*awaitables)
    analyzed_snippets = []

    for response in gathered_responses:
        if structured_output:
            try:
                snippet = RatedSnippetAnalysis.model_validate_json(response.text)
            except ValidationError as ex:
                LOG.error("Invalid data structure returned `%s`", response.text)
                raise ex
        else:
            snippet = SnippetAnalysis(text=response.text)
        analyzed_snippets.append(snippet)

    return analyzed_snippets


async def perfrom_analysis(log_text: str) -> Response:
    """Sumbit log file snippets in aggregate to LLM and retrieve results"""
    log_summary = mine_logs(log_text, SERVER_CONFIG.extractor.get_extractors())
    log_summary = format_snippets(log_summary)

    final_prompt = construct_final_prompt(log_summary, PROMPT_CONFIG.prompt_template)

    messages = prompt_to_messages(
        final_prompt,
        PROMPT_CONFIG.default_system_prompt,
        SERVER_CONFIG.inference.system_role,
        SERVER_CONFIG.inference.user_role,
    )
    response = await call_llm(
        messages,
        inference_cfg=SERVER_CONFIG.inference,
    )
    certainty = 0

    if response.logprobs is not None:
        try:
            certainty = compute_certainty(response.logprobs)
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n{response.logprobs}",
            ) from ex

    return Response(explanation=response, response_certainty=certainty)


async def perform_analyis_stream(log_text: str) -> AsyncStream:
    """Submit log file snippets in aggregate and return a stream of tokens"""
    log_summary = mine_logs(log_text, SERVER_CONFIG.extractor.get_extractors())
    log_summary = format_snippets(log_summary)

    final_prompt = construct_final_prompt(log_summary, PROMPT_CONFIG.prompt_template)

    messages = prompt_to_messages(
        final_prompt,
        PROMPT_CONFIG.default_system_prompt,
        SERVER_CONFIG.inference.system_role,
        SERVER_CONFIG.inference.user_role,
    )

    stream = call_llm_stream(
        messages,
        inference_cfg=SERVER_CONFIG.inference,
    )

    # we need to figure out a better response here, this is how it looks rn:
    # b'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],
    #   "created":1744818071,"id":"chatcmpl-c9geTxNcQO7M9wR...
    return stream


async def perform_staged_analysis(log_text: str) -> StagedResponse:
    """Submit the log file snippets to the LLM and retrieve their results"""
    log_summary = mine_logs(log_text, SERVER_CONFIG.extractor.get_extractors())
    start = time.time()
    if SERVER_CONFIG.general.top_k_snippets:
        rated_snippets = await analyze_snippets(
            log_summary=log_summary,
            structured_output=RatedSnippetAnalysis.model_json_schema(),
        )

        # Extract original text and line number from `log_summary`
        processed_snippets = [
            AnalyzedSnippet(line_number=e[0][0], text=e[0][1], explanation=e[1])
            for e in zip(log_summary, rated_snippets)
        ]
        processed_snippets = filter_snippets(
            processed_snippets=processed_snippets,
            top_k=SERVER_CONFIG.general.top_k_snippets,
        )
        LOG.info(
            "Keeping %d of original %d snippets",
            len(processed_snippets),
            len(rated_snippets),
        )
    else:
        processed_snippets = await analyze_snippets(log_summary=log_summary)

        # Extract original text and line number from `log_summary`
        processed_snippets = [
            AnalyzedSnippet(line_number=e[0][0], text=e[0][1], explanation=e[1])
            for e in zip(log_summary, processed_snippets)
        ]
    delta = time.time() - start
    LOG.info("Snippet analysis performed in %f s", delta)
    log_summary = format_analyzed_snippets(processed_snippets)
    final_prompt = construct_final_prompt(log_summary, PROMPT_CONFIG.prompt_template_staged)

    messages = prompt_to_messages(
        final_prompt,
        PROMPT_CONFIG.staged_system_prompt,
        SERVER_CONFIG.inference.system_role,
        SERVER_CONFIG.inference.user_role,
    )
    final_analysis = await call_llm(
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
        snippets=processed_snippets,
        response_certainty=certainty,
    )
