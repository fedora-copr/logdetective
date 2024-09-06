import json
import logging
import os
from typing import List

from llama_cpp import CreateCompletionResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests

from logdetective.constants import PROMPT_TEMPLATE, SNIPPET_PROMPT_TEMPLATE
from logdetective.extractors import DrainExtractor
from logdetective.utils import validate_url, compute_certainty

class BuildLog(BaseModel):
    """Model of data submitted to API.
    """
    url: str


class Response(BaseModel):
    """Model of data returned by Log Detective API

    explanation: CreateCompletionResponse
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama_types.CreateCompletionResponse
    response_certainty: float
    """
    explanation: CreateCompletionResponse
    response_certainty: float


class StagedResponse(Response):
    """Model of data returned by Log Detective API when called when staged response
    is requested. Contains list of reponses to prompts for individual snippets.

    explanation: CreateCompletionResponse
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama_types.CreateCompletionResponse
    response_certainty: float
    snippets: list of CreateCompletionResponse
    """
    snippets: List[CreateCompletionResponse]


LOG = logging.getLogger("logdetective")

app = FastAPI()

LLM_CPP_HOST = os.environ.get("LLAMA_CPP_HOST", "localhost")
LLM_CPP_SERVER_ADDRESS = f"http://{LLM_CPP_HOST}"
LLM_CPP_SERVER_PORT = os.environ.get("LLAMA_CPP_SERVER_PORT", 8000)
LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)
LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)


def process_url(url: str) -> str:
    """Validate log URL and return log text.
    """
    if validate_url(url=url):
        try:
            log_request = requests.get(url, timeout=int(LOG_SOURCE_REQUEST_TIMEOUT))
        except requests.RequestException as ex:
            raise HTTPException(
                status_code=400,
                detail=f"We couldn't obtain the logs: {ex}") from ex

        if not log_request.ok:
            raise HTTPException(status_code=400,
                                detail="Something went wrong while getting the logs: "
                                    f"[{log_request.status_code}] {log_request.text}")
    else:
        LOG.error("Invalid URL received ")
        raise HTTPException(status_code=400,
                            detail=f"Invalid log URL: {url}")

    return log_request.text


def mine_logs(log: str) -> List[str]:
    """Extract snippets from log text
    """
    extractor = DrainExtractor(verbose=True, context=True, max_clusters=8)

    LOG.info("Getting summary")
    log_summary = extractor(log)

    ratio = len(log_summary) / len(log.split('\n'))
    LOG.debug("Log summary: \n %s", log_summary)
    LOG.info("Compression ratio: %s", ratio)


    return log_summary

def submit_text(text: str, max_tokens: int = 0, log_probs: int = 1):
    """Submit prompt to LLM.
    max_tokens: number of tokens to be produces, 0 indicates run until encountering EOS
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Analyzing the text")
    data = {
            "prompt": text,
            "max_tokens": str(max_tokens),
            "logprobs": str(log_probs)}

    try:
        # Expects llama-cpp server to run on LLM_CPP_SERVER_ADDRESS:LLM_CPP_SERVER_PORT
        response = requests.post(
            f"{LLM_CPP_SERVER_ADDRESS}:{LLM_CPP_SERVER_PORT}/v1/completions",
            headers={"Content-Type":"application/json"},
            data=json.dumps(data),
            timeout=int(LLM_CPP_SERVER_TIMEOUT))
    except requests.RequestException as ex:
        raise HTTPException(
            status_code=400,
            detail=f"Llama-cpp query failed: {ex}") from ex

    if not response.ok:
        raise HTTPException(
            status_code=400,
            detail="Something went wrong while getting a response from the llama server: "
                f"[{response.status_code}] {response.text}")
    try:
        response = json.loads(response.text)
    except UnicodeDecodeError as ex:
        LOG.error("Error encountered while parsing llama server response: %s", ex)
        raise HTTPException(
            status_code=400,
            detail=f"Couldn't parse the response.\nError: {ex}\nData: {response.text}") from ex

    return CreateCompletionResponse(response)


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
    response = submit_text(PROMPT_TEMPLATE.format(log_summary))

    if "logprobs" in response["choices"][0]:
        try:
            certainty = compute_certainty(
                response["choices"][0]["logprobs"]["top_logprobs"])
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n"
                f"{response["choices"][0]["logprobs"]["top_logprobs"]}") from ex

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

    analyzed_snippets = []

    for snippet in log_summary:
        response = submit_text(SNIPPET_PROMPT_TEMPLATE.format(snippet))
        analyzed_snippets.append(response)

    final_analysis = submit_text(
        PROMPT_TEMPLATE.format([e["choices"][0]["text"] for e in analyzed_snippets]))

    certainty = 0
    if "logprobs" in final_analysis["choices"][0]:
        try:
            certainty = compute_certainty(
                final_analysis["choices"][0]["logprobs"]["top_logprobs"])
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n"
                f"{final_analysis["choices"][0]["logprobs"]["top_logprobs"]}") from ex

    return StagedResponse(
        explanation=final_analysis, snippets=analyzed_snippets, response_certainty=certainty)
