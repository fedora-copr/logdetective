import logging
import os
import json

from fastapi import FastAPI
from pydantic import BaseModel

import requests

from logdetective.constants import PROMPT_TEMPLATE
from logdetective.extractors import DrainExtractor


class BuildLog(BaseModel):
    """Model of data submitted to API.
    """
    url: str

LOG = logging.getLogger("logdetective")

app = FastAPI()

LLM_CPP_SERVER_ADDRESS = os.environ.get("LLAMA_CPP_SERVER", " http://localhost")
LLM_CPP_SERVER_PORT = os.environ.get("LLAMA_CPP_SERVER_PORT", 8000)
LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)
LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)

@app.post("/analyze", )
async def analyze_log(build_log: BuildLog):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    """
    extractor = DrainExtractor(verbose=True, context=True, max_clusters=8)

    LOG.info("Getting summary")

    log = requests.get(build_log.url, timeout=int(LOG_SOURCE_REQUEST_TIMEOUT)).text
    log_summary = extractor(log)

    ratio = len(log_summary) / len(log.split('\n'))
    LOG.debug("Log summary: \n %s", log_summary)
    LOG.info("Compression ratio: %s", ratio)

    LOG.info("Analyzing the text")
    data = {
            "prompt": PROMPT_TEMPLATE.format(log_summary),
            "max_tokens": "0"}

    # Expects llama-cpp server to run on LLM_CPP_SERVER_ADDRESS:LLM_CPP_SERVER_PORT
    response = requests.post(
        f"{LLM_CPP_SERVER_ADDRESS}:{LLM_CPP_SERVER_PORT}/v1/completions",
        headers={"Content-Type":"application/json"},
        data=json.dumps(data),
        timeout=int(LLM_CPP_SERVER_TIMEOUT))

    return response.text
