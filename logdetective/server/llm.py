import os
import asyncio
import random
from typing import List, Tuple, Dict

import backoff
from fastapi import HTTPException
from pydantic import ValidationError

import aiohttp
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from logdetective.constants import SNIPPET_DELIMITER
from logdetective.extractors import DrainExtractor
from logdetective.utils import (
    compute_certainty,
    prompt_to_messages,
)
from logdetective.server.config import (
    LOG,
    SERVER_CONFIG,
    PROMPT_CONFIG,
    CLIENT,
    SKIP_SNIPPETS_CONFIG,
)
from logdetective.server.models import (
    AnalyzedSnippet,
    InferenceConfig,
    Explanation,
    StagedResponse,
    SnippetAnalysis,
    RatedSnippetAnalysis,
)


LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)


def format_analyzed_snippets(snippets: list[AnalyzedSnippet]) -> str:
    """Format snippets for submission into staged prompt."""
    summary = f"\n{SNIPPET_DELIMITER}\n".join(
        [f"[{e.text}] at line [{e.line_number}]: [{e.explanation}]" for e in snippets]
    )
    return summary


def mine_logs(log: str) -> List[Tuple[int, str]]:
    """Extract snippets from log text"""
    extractor = DrainExtractor(
        verbose=True,
        context=True,
        max_clusters=SERVER_CONFIG.extractor.max_clusters,
        skip_snippets=SKIP_SNIPPETS_CONFIG,
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

    # OpenAI API does not guarantee that the behavior for parameter set to `None`
    # and parameter not given at all is the same.
    # Therefore we must branch on the way we call the API.
    if structured_output:
        LOG.info("Requesting structured output from LLM")
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "rated-snippet-analysis",
                "schema": structured_output,
            },
        }

        async with inference_cfg.get_limiter():
            response = await CLIENT.chat.completions.create(
                messages=messages,
                max_tokens=inference_cfg.max_tokens,
                logprobs=inference_cfg.log_probs,
                stream=stream,
                model=inference_cfg.model,
                temperature=inference_cfg.temperature,
                response_format=response_format,
            )
    else:
        async with inference_cfg.get_limiter():
            response = await CLIENT.chat.completions.create(
                messages=messages,
                max_tokens=inference_cfg.max_tokens,
                logprobs=inference_cfg.log_probs,
                stream=stream,
                model=inference_cfg.model,
                temperature=inference_cfg.temperature,
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


async def perform_staged_analysis(log_text: str) -> StagedResponse:
    """Submit the log file snippets to the LLM and retrieve their results"""
    log_summary = mine_logs(log_text)

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

    final_prompt = PROMPT_CONFIG.prompt_template_staged.format(
        format_analyzed_snippets(processed_snippets)
    )
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
