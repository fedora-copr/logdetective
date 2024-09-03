import logging
import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests

from logdetective.constants import PROMPT_TEMPLATE
from logdetective.extractors import DrainExtractor
from logdetective.utils import validate_url

class BuildLog(BaseModel):
    """Model of data submitted to API.
    """
    url: str

LOG = logging.getLogger("logdetective")

app = FastAPI()

LLM_CPP_HOST = os.environ.get("LLAMA_CPP_HOST", "localhost")
LLM_CPP_SERVER_ADDRESS = f"http://{LLM_CPP_HOST}"
LLM_CPP_SERVER_PORT = os.environ.get("LLAMA_CPP_SERVER_PORT", 8000)
LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)
LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)

@app.post("/analyze", )
async def analyze_log(build_log: BuildLog):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    extractor = DrainExtractor(verbose=True, context=True, max_clusters=8)

    LOG.info("Getting summary")
    # Perform basic validation of the URL
    if validate_url(url=build_log.url):
        try:
            log_request = requests.get(build_log.url, timeout=int(LOG_SOURCE_REQUEST_TIMEOUT))
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
                            detail=f"Invalid log URL: {build_log.url}")

    log = log_request.text
    log_summary = extractor(log)

    ratio = len(log_summary) / len(log.split('\n'))
    LOG.debug("Log summary: \n %s", log_summary)
    LOG.info("Compression ratio: %s", ratio)

    LOG.info("Analyzing the text")
    data = {
            "prompt": PROMPT_TEMPLATE.format(log_summary),
            "max_tokens": "0"}

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

    if not log_request.ok:
        raise HTTPException(
            status_code=400,
            detail="Something went wrong while getting a response from the llama server: "
                   f"[{log_request.status_code}] {log_request.text}")
    return response.text
