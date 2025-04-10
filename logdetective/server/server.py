import asyncio
import json
import os
import re
from tempfile import TemporaryFile
from typing import List, Annotated, Tuple, Dict, Any
from io import BytesIO


import matplotlib
import matplotlib.pyplot
from fastapi import FastAPI, HTTPException, Depends, Header

from fastapi.responses import StreamingResponse
from fastapi.responses import Response as BasicResponse
import requests

from logdetective.extractors import DrainExtractor
from logdetective.utils import (
    validate_url,
    compute_certainty,
    format_snippets,
    load_prompts,
)
from logdetective.server.utils import (
    load_server_config,
    get_log,
    format_analyzed_snippets,
)
from logdetective.server.metric import track_request
from logdetective.server.models import (
    BuildLog,
    JobHook,
    Response,
    StagedResponse,
    Explanation,
    AnalyzedSnippet,
    TimePeriod,
)
from logdetective.server import plot
from logdetective.server.database.models import EndpointType

LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)
LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)
API_TOKEN = os.environ.get("LOGDETECTIVE_TOKEN", None)
SERVER_CONFIG_PATH = os.environ.get("LOGDETECTIVE_SERVER_CONF", None)
SERVER_PROMPT_PATH = os.environ.get("LOGDETECTIVE_PROMPTS", None)

SERVER_CONFIG = load_server_config(SERVER_CONFIG_PATH)
PROMPT_CONFIG = load_prompts(SERVER_PROMPT_PATH)

MR_REGEX = re.compile(r"refs/merge-requests/(\d+)/.*$")
FAILURE_LOG_REGEX = re.compile(r"(\w*\.log)")

LOG = get_log(SERVER_CONFIG)


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
        LOG.error("Llama-cpp query failed: %s", ex)
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


async def submit_text(  # pylint: disable=R0913,R0917
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
            text, headers, max_tokens, log_probs > 0, stream, model
        )
    return await submit_text_completions(
        text, headers, max_tokens, log_probs, stream, model
    )


async def submit_text_completions(  # pylint: disable=R0913,R0917
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
        f"{SERVER_CONFIG.inference.url}/v1/completions",
        data,
        headers,
        stream,
    )

    return Explanation(
        text=response["choices"][0]["text"], logprobs=response["choices"][0]["logprobs"]
    )


async def submit_text_chat_completions(  # pylint: disable=R0913,R0917
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
        "temperature": SERVER_CONFIG.inference.temperature,
    }

    response = await submit_to_llm_endpoint(
        f"{SERVER_CONFIG.inference.url}/v1/chat/completions",
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
@track_request()
async def analyze_log(build_log: BuildLog):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    log_text = process_url(build_log.url)
    log_summary = mine_logs(log_text)
    log_summary = format_snippets(log_summary)
    response = await submit_text(
        PROMPT_CONFIG.prompt_template.format(log_summary),
        model=SERVER_CONFIG.inference.model,
        max_tokens=SERVER_CONFIG.inference.max_tokens,
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


@app.post("/analyze/staged", response_model=StagedResponse)
@track_request()
async def analyze_log_staged(build_log: BuildLog):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    log_text = process_url(build_log.url)

    return await perform_staged_analysis(log_text=log_text)


async def perform_staged_analysis(log_text: str) -> StagedResponse:
    """Submit the log file snippets to the LLM and retrieve their results"""
    log_summary = mine_logs(log_text)

    # Process snippets asynchronously
    analyzed_snippets = await asyncio.gather(
        *[
            submit_text(
                PROMPT_CONFIG.snippet_prompt_template.format(s),
                model=SERVER_CONFIG.inference.model,
                max_tokens=SERVER_CONFIG.inference.max_tokens,
            )
            for s in log_summary
        ]
    )

    analyzed_snippets = [
        AnalyzedSnippet(line_number=e[0][0], text=e[0][1], explanation=e[1])
        for e in zip(log_summary, analyzed_snippets)
    ]
    final_prompt = PROMPT_CONFIG.prompt_template_staged.format(
        format_analyzed_snippets(analyzed_snippets)
    )

    final_analysis = await submit_text(
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


@app.post("/analyze/stream", response_class=StreamingResponse)
@track_request()
async def analyze_log_stream(build_log: BuildLog):
    """Stream response endpoint for Logdetective.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    log_text = process_url(build_log.url)
    log_summary = mine_logs(log_text)
    log_summary = format_snippets(log_summary)
    headers = {"Content-Type": "application/json"}

    if SERVER_CONFIG.inference.api_token:
        headers["Authorization"] = f"Bearer {SERVER_CONFIG.inference.api_token}"

    stream = await submit_text_chat_completions(
        PROMPT_CONFIG.prompt_template.format(log_summary), stream=True, headers=headers,
        model=SERVER_CONFIG.inference.model,
        max_tokens=SERVER_CONFIG.inference.max_tokens,
    )

    return StreamingResponse(stream)


def _svg_figure_response(fig: matplotlib.figure.Figure):
    """Create a response with the given svg figure."""
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    matplotlib.pyplot.close(fig)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/svg+xml",
        headers={"Content-Disposition": "inline; filename=plot.svg"},
    )


def _multiple_svg_figures_response(figures: list[matplotlib.figure.Figure]):
    """Create a response with multiple svg figures."""
    svg_contents = []
    for i, fig in enumerate(figures):
        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        matplotlib.pyplot.close(fig)
        buf.seek(0)
        svg_contents.append(buf.read().decode("utf-8"))

    html_content = "<html><body>\n"
    for i, svg in enumerate(svg_contents):
        html_content += f"<div id='figure-{i}'>\n{svg}\n</div>\n"
    html_content += "</body></html>"

    return BasicResponse(content=html_content, media_type="text/html")


@app.get("/metrics/analyze", response_class=StreamingResponse)
async def show_analyze_metrics(period_since_now: TimePeriod = Depends(TimePeriod)):
    """Show statistics for requests and responses in the given period of time
    for the /analyze API endpoint."""
    fig_requests = plot.requests_per_time(period_since_now, EndpointType.ANALYZE)
    fig_responses = plot.average_time_per_responses(
        period_since_now, EndpointType.ANALYZE
    )
    return _multiple_svg_figures_response([fig_requests, fig_responses])


@app.get("/metrics/analyze/requests", response_class=StreamingResponse)
async def show_analyze_requests(period_since_now: TimePeriod = Depends(TimePeriod)):
    """Show statistics for the requests received in the given period of time
    for the /analyze API endpoint."""
    fig = plot.requests_per_time(period_since_now, EndpointType.ANALYZE)
    return _svg_figure_response(fig)


@app.get("/metrics/analyze/responses", response_class=StreamingResponse)
async def show_analyze_responses(period_since_now: TimePeriod = Depends(TimePeriod)):
    """Show statistics for responses given in the specified period of time
    for the /analyze API endpoint."""
    fig = plot.average_time_per_responses(period_since_now, EndpointType.ANALYZE)
    return _svg_figure_response(fig)


@app.get("/metrics/analyze/staged", response_class=StreamingResponse)
async def show_analyze_staged_metrics(
    period_since_now: TimePeriod = Depends(TimePeriod),
):
    """Show statistics for requests and responses in the given period of time
    for the /analyze/staged API endpoint."""
    fig_requests = plot.requests_per_time(period_since_now, EndpointType.ANALYZE_STAGED)
    fig_responses = plot.average_time_per_responses(
        period_since_now, EndpointType.ANALYZE_STAGED
    )
    return _multiple_svg_figures_response([fig_requests, fig_responses])


@app.get("/metrics/analyze/staged/requests", response_class=StreamingResponse)
async def show_analyze_staged_requests(
    period_since_now: TimePeriod = Depends(TimePeriod),
):
    """Show statistics for the requests received in the given period of time
    for the /analyze/staged API endpoint."""
    fig = plot.requests_per_time(period_since_now, EndpointType.ANALYZE_STAGED)
    return _svg_figure_response(fig)


@app.get("/metrics/analyze/staged/responses", response_class=StreamingResponse)
async def show_analyze_staged_responses(
    period_since_now: TimePeriod = Depends(TimePeriod),
):
    """Show statistics for responses given in the specified period of time
    for the /analyze/staged API endpoint."""
    fig = plot.average_time_per_responses(period_since_now, EndpointType.ANALYZE_STAGED)
    return _svg_figure_response(fig)
