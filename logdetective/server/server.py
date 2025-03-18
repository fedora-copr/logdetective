import asyncio
import json
import logging
import os
from typing import List, Annotated, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
import requests

from logdetective.constants import (
    PROMPT_TEMPLATE,
    SNIPPET_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_STAGED,
    SNIPPET_DELIMITER,
)
from logdetective.extractors import DrainExtractor
from logdetective.utils import validate_url, compute_certainty
from logdetective.server.models import BuildLog, Response, StagedResponse, Explanation
from logdetective.server.utils import load_server_config

LOG = logging.getLogger("logdetective")

LLM_CPP_HOST = os.environ.get("LLAMA_CPP_HOST", "localhost")
LLM_CPP_SERVER_ADDRESS = f"http://{LLM_CPP_HOST}"
LLM_CPP_SERVER_PORT = os.environ.get("LLAMA_CPP_SERVER_PORT", 8000)
LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)
LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)
API_TOKEN = os.environ.get("LOGDETECTIVE_TOKEN", None)
SERVER_CONFIG_PATH = os.environ.get("LOGDETECTIVE_SERVER_CONF", None)
LLM_API_TOKEN = os.environ.get("LLM_API_TOKEN", None)

SERVER_CONFIG = load_server_config(SERVER_CONFIG_PATH)


def requires_token_when_set(authentication: Annotated[str | None, Header()] = None):
    """
    FastAPI Depend function that expects a header named Authentication

    If LOGDETECTIVE_TOKEN env var is set, validate the client-supplied token
    otherwise ignore it
    """
    if not API_TOKEN:
        LOG.info("LOGDETECTIVE_TOKEN env var not set, authentication disabled")
        # no token required, means local dev environment
        return
    token = None
    if authentication:
        try:
            token = authentication.split(" ", 1)[1]
        except (ValueError, IndexError):
            LOG.warning(
                "Authentication header has invalid structure (%s), it should be 'Bearer TOKEN'",
                authentication,
            )
            # eat the exception and raise 401 below
            token = None
        if token == API_TOKEN:
            return
    LOG.info(
        "LOGDETECTIVE_TOKEN env var is set (%s), clien token = %s", API_TOKEN, token
    )
    raise HTTPException(status_code=401, detail=f"Token {token} not valid.")


app = FastAPI(dependencies=[Depends(requires_token_when_set)])


def process_url(url: str) -> str:
    """Validate log URL and return log text."""
    if validate_url(url=url):
        try:
            log_request = requests.get(url, timeout=int(LOG_SOURCE_REQUEST_TIMEOUT))
        except requests.RequestException as ex:
            raise HTTPException(
                status_code=400, detail=f"We couldn't obtain the logs: {ex}"
            ) from ex

        if not log_request.ok:
            raise HTTPException(
                status_code=400,
                detail="Something went wrong while getting the logs: "
                f"[{log_request.status_code}] {log_request.text}",
            )
    else:
        LOG.error("Invalid URL received ")
        raise HTTPException(status_code=400, detail=f"Invalid log URL: {url}")

    return log_request.text


def mine_logs(log: str) -> List[str]:
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
    url: str, data: Dict[str, Any], headers: Dict[str, str], stream: bool
) -> Any:
    """Send request to selected API endpoint. Verifying successful request unless
    the using the stream response.

    url:
    data:
    headers:
    stream:
    """
    try:
        # Expects llama-cpp server to run on LLM_CPP_SERVER_ADDRESS:LLM_CPP_SERVER_PORT
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data),
            timeout=int(LLM_CPP_SERVER_TIMEOUT),
            stream=stream,
        )
    except requests.RequestException as ex:
        raise HTTPException(
            status_code=400, detail=f"Llama-cpp query failed: {ex}"
        ) from ex
    if not stream:
        if not response.ok:
            raise HTTPException(
                status_code=400,
                detail="Something went wrong while getting a response from the llama server: "
                f"[{response.status_code}] {response.text}",
            )
        try:
            response = json.loads(response.text)
        except UnicodeDecodeError as ex:
            LOG.error("Error encountered while parsing llama server response: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't parse the response.\nError: {ex}\nData: {response.text}",
            ) from ex

    return response


async def submit_text(
    text: str,
    max_tokens: int = -1,
    log_probs: int = 1,
    stream: bool = False,
    model: str = "default-model",
    api_endpoint: str = "/chat/completions",
) -> Explanation:
    """Submit prompt to LLM using a selected endpoint.
    max_tokens: number of tokens to be produces, 0 indicates run until encountering EOS
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Analyzing the text")

    headers = {"Content-Type": "application/json"}

    if LLM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_API_TOKEN}"

    if api_endpoint == "/chat/completions":
        return await submit_text_chat_completions(
            text, headers, max_tokens, log_probs > 0, stream, model
        )
    return await submit_text_completions(
        text, headers, max_tokens, log_probs, stream, model
    )


async def submit_text_completions(
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
    }

    response = await submit_to_llm_endpoint(
        f"{LLM_CPP_SERVER_ADDRESS}:{LLM_CPP_SERVER_PORT}/v1/completions",
        data,
        headers,
        stream,
    )

    return Explanation(
        text=response["choices"][0]["text"], logprobs=response["choices"][0]["logprobs"]
    )


async def submit_text_chat_completions(
    text: str,
    headers: dict,
    max_tokens: int = -1,
    log_probs: int = 1,
    stream: bool = False,
    model: str = "default-model",
) -> Explanation:
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
    }

    response = await submit_to_llm_endpoint(
        f"{LLM_CPP_SERVER_ADDRESS}:{LLM_CPP_SERVER_PORT}/v1/chat/completions",
        data,
        headers,
        stream,
    )

    if stream:
        return Explanation(
            text=response["choices"][0]["delta"]["content"],
            logprobs=response["choices"][0]["logprobs"]["content"],
        )
    return Explanation(
        text=response["choices"][0]["message"]["content"],
        logprobs=response["choices"][0]["logprobs"]["content"],
    )


@app.post("/analyze", response_model=Response)
async def analyze_log(build_log: BuildLog):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    log_text = process_url(build_log.url)
    log_summary = mine_logs(log_text)
    response = await submit_text(
        PROMPT_TEMPLATE.format(log_summary),
        api_endpoint=SERVER_CONFIG.inference.api_endpoint,
    )
    certainty = 0

    if response.logprobs is not None:
        try:
            certainty = compute_certainty(response.logprobs)
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n"
                f"{response.logprobs}",
            ) from ex

    return Response(explanation=response, response_certainty=certainty)


@app.post("/analyze/staged", response_model=StagedResponse)
async def analyze_log_staged(build_log: BuildLog):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    log_text = process_url(build_log.url)
    log_summary = mine_logs(log_text)

    # Process snippets asynchronously
    analyzed_snippets = await asyncio.gather(
        *[
            submit_text(
                SNIPPET_PROMPT_TEMPLATE.format(s),
                api_endpoint=SERVER_CONFIG.inference.api_endpoint,
            )
            for s in log_summary
        ]
    )

    analyzed_snippets = [
        {"snippet": e[0], "comment": e[1]} for e in zip(log_summary, analyzed_snippets)
    ]

    final_prompt = PROMPT_TEMPLATE_STAGED.format(
        f"\n{SNIPPET_DELIMITER}\n".join(
            [f"[{e["snippet"]}] : [{e["comment"].text}]" for e in analyzed_snippets]
        )
    )

    final_analysis = await submit_text(
        final_prompt, api_endpoint=SERVER_CONFIG.inference.api_endpoint
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


@app.post("/analyze/stream", response_class=StreamingResponse)
async def analyze_log_stream(build_log: BuildLog):
    """Stream response endpoint for Logdetective.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    log_text = process_url(build_log.url)
    log_summary = mine_logs(log_text)
    headers = {"Content-Type": "application/json"}

    if LLM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_API_TOKEN}"

    stream = await submit_text_chat_completions(
        PROMPT_TEMPLATE.format(log_summary), stream=True, headers=headers
    )

    return StreamingResponse(stream)
