import asyncio
import json
import os
import re
import zipfile
from pathlib import PurePath
from tempfile import TemporaryFile
from typing import List, Annotated, Tuple


from llama_cpp import CreateCompletionResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.responses import Response as BasicResponse
import gitlab
import requests

from logdetective.constants import (
    PROMPT_TEMPLATE,
    SNIPPET_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_STAGED,
)
from logdetective.extractors import DrainExtractor
from logdetective.utils import (
    validate_url,
    compute_certainty,
    format_snippets,
    format_analyzed_snippets,
)
from logdetective.server.models import BuildLog, JobHook, Response, StagedResponse
from logdetective.server.utils import load_server_config, get_log


LLM_CPP_HOST = os.environ.get("LLAMA_CPP_HOST", "localhost")
LLM_CPP_SERVER_ADDRESS = f"http://{LLM_CPP_HOST}"
LLM_CPP_SERVER_PORT = os.environ.get("LLAMA_CPP_SERVER_PORT", 8000)
LLM_CPP_SERVER_TIMEOUT = os.environ.get("LLAMA_CPP_SERVER_TIMEOUT", 600)
LOG_SOURCE_REQUEST_TIMEOUT = os.environ.get("LOG_SOURCE_REQUEST_TIMEOUT", 60)
API_TOKEN = os.environ.get("LOGDETECTIVE_TOKEN", None)
SERVER_CONFIG_PATH = os.environ.get("LOGDETECTIVE_SERVER_CONF", None)
LLM_API_TOKEN = os.environ.get("LLM_API_TOKEN", None)

SERVER_CONFIG = load_server_config(SERVER_CONFIG_PATH)

MR_REGEX = re.compile(r"refs/merge-requests/(\d+)/merge")
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
app.gitlab_conn = gitlab.Gitlab(
    url=SERVER_CONFIG.gitlab.url, private_token=SERVER_CONFIG.gitlab.api_token
)


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


async def submit_text(
    text: str,
    max_tokens: int = -1,
    log_probs: int = 1,
    stream: bool = False,
    model: str = "default-model",
):
    """Submit prompt to LLM.
    max_tokens: number of tokens to be produces, 0 indicates run until encountering EOS
    log_probs: number of token choices to produce log probs for
    """
    LOG.info("Analyzing the text")
    data = {
        "prompt": text,
        "max_tokens": max_tokens,
        "logprobs": log_probs,
        "stream": stream,
        "model": model,
    }

    headers = {"Content-Type": "application/json"}

    if LLM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_API_TOKEN}"

    try:
        # Expects llama-cpp server to run on LLM_CPP_SERVER_ADDRESS:LLM_CPP_SERVER_PORT
        response = requests.post(
            f"{LLM_CPP_SERVER_ADDRESS}:{LLM_CPP_SERVER_PORT}/v1/completions",
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
    else:
        return response

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
    log_summary = format_snippets(log_summary)
    response = await submit_text(PROMPT_TEMPLATE.format(log_summary))
    certainty = 0

    if "logprobs" in response["choices"][0]:
        try:
            certainty = compute_certainty(
                response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            )
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n"
                f"{response['choices'][0]['logprobs']['content'][0]['top_logprobs']}",
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
        *[submit_text(SNIPPET_PROMPT_TEMPLATE.format(s[1])) for s in log_summary]
    )

    analyzed_snippets = [
        {"snippet": e[0][1], "line_number": e[0][0], "comment": e[1]}
        for e in zip(log_summary, analyzed_snippets)
    ]
    final_prompt = PROMPT_TEMPLATE_STAGED.format(
        format_analyzed_snippets(analyzed_snippets)
    )

    final_analysis = await submit_text(final_prompt)
    print(final_analysis)
    certainty = 0

    if "logprobs" in final_analysis["choices"][0]:
        try:
            certainty = compute_certainty(
                final_analysis["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            )
        except ValueError as ex:
            LOG.error("Error encountered while computing certainty: %s", ex)
            raise HTTPException(
                status_code=400,
                detail=f"Couldn't compute certainty with data:\n"
                f"{final_analysis['choices'][0]['logprobs']['content'][0]['top_logprobs']}",
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
    log_summary = format_snippets(log_summary)
    stream = await submit_text(PROMPT_TEMPLATE.format(log_summary), stream=True)

    return StreamingResponse(stream)


@app.post("/webhook/gitlab/job_events")
async def receive_gitlab_job_event_webhook(
    job_hook: JobHook, background_tasks: BackgroundTasks
):
    """Webhook endpoint for receiving job_events notifications from GitLab
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    lists the full specification for the messages sent for job events."""

    # Handle the message in the background so we can return 200 immediately
    background_tasks.add_task(process_gitlab_job_event, job_hook)

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)


async def process_gitlab_job_event(job_hook):
    """Handle a received job_event webhook from GitLab"""
    LOG.debug("Received webhook message:\n%s", job_hook)

    # Look up the project this job belongs to
    project = await asyncio.to_thread(app.gitlab_conn.projects.get, job_hook.project_id)

    # check if this project is on the opt-in list
    if project.name not in SERVER_CONFIG.general.packages:
        LOG.info("Ignoring unrecognized package %s", project.name)
        return
    LOG.info("Processing failed job for %s", project.name)

    # Retrieve data about the job from the GitLab API
    job = await asyncio.to_thread(project.jobs.get, job_hook.build_id)

    # Retrieve the pipeline that started this job
    pipeline = await asyncio.to_thread(project.pipelines.get, job_hook.pipeline_id)

    # Verify this is a merge request
    if pipeline.source != "merge_request_event":
        LOG.info("Not a merge request pipeline. Ignoring.")
        return

    # Extract the merge-request ID from the job
    match = MR_REGEX.search(pipeline.ref)
    if not match:
        LOG.error(
            "Pipeline source is merge_request_event but no merge request ID was provided."
        )
        return
    merge_request_id = int(match.group(1))

    LOG.debug("Retrieving log artifacts")
    # Retrieve the build logs from the merge request artifacts and preprocess them
    try:
        preprocessed_log = await retrieve_and_preprocess_koji_logs(job)
    except LogsTooLargeError:
        LOG.error("Could not retrieve logs. Too large.")
        raise

    # Submit log to Log Detective and await the results.
    response = await submit_log_to_llm(preprocessed_log)
    preprocessed_log.close()

    # Add the Log Detective response as a comment to the merge request
    await comment_on_mr(merge_request_id, response)


class LogsTooLargeError(RuntimeError):
    """The log archive exceeds the configured maximum size"""


async def retrieve_and_preprocess_koji_logs(job):
    """Download logs from the merge request artifacts

    This function will retrieve the build logs and do some minimal
    preprocessing to determine which log is relevant for analysis.

    returns: An open, file-like object containing the log contents to be sent
    for processing by Log Detective. The calling function is responsible for
    closing this object."""

    # Make sure the file isn't too large to process.
    if not await check_artifacts_file_size(job):
        raise LogsTooLargeError(
            f"Oversized logs for job {job.id} in project {job.project_id}"
        )

    # Create a temporary file to store the downloaded log zipfile.
    # This will be automatically deleted when the last reference into it
    # (returned by this function) is closed.
    tempfile = TemporaryFile(mode="w+b")
    await asyncio.to_thread(job.artifacts, streamed=True, action=tempfile.write)
    tempfile.seek(0)

    failed_arches = {}
    artifacts_zip = zipfile.ZipFile(tempfile, mode="r")
    for zipinfo in artifacts_zip.infolist():
        if zipinfo.filename.endswith("task_failed.log"):
            # The koji logs store this file in two places: 1) in the
            # directory with the failed architecture and 2) in the parent
            # directory. We actually want to ignore the one in the parent
            # directory, since the rest of the information is in the
            # specific task directory.
            # The paths look like `kojilogs/noarch-XXXXXX/task_failed.log`
            # or `kojilogs/noarch-XXXXXX/x86_64-XXXXXX/task_failed.log`
            path = PurePath(zipinfo.filename)
            if len(path.parts) <= 3:
                continue

            # Extract the architecture from the immediate parent path
            architecture = path.parent.parts[-1].split("-")[0]

            # Open this file and read which log failed.
            # The string in this log has the format
            # `see <log> for more information`.
            # Note: it may sometimes say
            # `see build.log or root.log for more information`, but in
            # that situation, we only want to handle build.log (for now),
            # which means accepting only the first match for the regular
            # expression.
            with artifacts_zip.open(zipinfo.filename) as task_failed_log:
                contents = task_failed_log.read().decode("utf-8")
                match = FAILURE_LOG_REGEX.search(contents)
                if not match:
                    LOG.error(
                        "task_failed.log does not indicate which log contains the failure."
                    )
                    raise SyntaxError(
                        "task_failed.log does not indicate which log contains the failure."
                    )
                failure_log_name = match.group(1)

            failed_arches[architecture] = PurePath(path.parent, failure_log_name)

    if not failed_arches:
        # No failed task found?
        raise FileNotFoundError("Could not detect failed architecture.")

    # First check if we only found one failed architecture
    if len(failed_arches) == 1:
        failed_arch = list(failed_arches.keys())[0]

    else:
        # We only want to handle one arch, so we'll check them in order of
        # "most to least likely for the maintainer to have access to hardware"
        # This means: x86_64 > aarch64 > ppc64le > s390x
        if "x86_64" in failed_arches:
            failed_arch = "x86_64"
        elif "aarch64" in failed_arches:
            failed_arch = "aarch64"
        elif "ppc64le" in failed_arches:
            failed_arch = "ppc64le"
        elif "s390x" in failed_arches:
            failed_arch = "s390x"
        else:
            # It should be impossible for us to get "noarch" here, since
            # the only way that should happen is for a single architecture
            # build.
            raise FileNotFoundError("No failed architecture detected.")

    LOG.debug("Failed architecture: %s", failed_arch)

    log_path = failed_arches[failed_arch]
    LOG.debug("Returning contents of %s", log_path)

    # Return the log as a file-like object with .read() function
    return artifacts_zip.open(log_path.as_posix())


async def check_artifacts_file_size(job):
    """Method to determine if the artifacts are too large to process"""
    # First, make sure that the artifacts are of a reasonable size. The
    # zipped artifact collection will be stored in memory below. The
    # python-gitlab library doesn't expose a way to check this value directly,
    # so we need to interact with directly with the headers.
    artifacts_url = f"{SERVER_CONFIG.gitlab.api_url}/projects/{job.project_id}/jobs/{job.id}/artifacts"  # pylint: disable=line-too-long
    header_resp = await asyncio.to_thread(
        requests.head,
        artifacts_url,
        allow_redirects=True,
        headers={"Authorization": f"Bearer {SERVER_CONFIG.gitlab.api_token}"},
        timeout=(3.07, 5),
    )
    content_length = int(header_resp.headers.get("content-length"))
    LOG.debug(
        "URL: %s, content-length: %d, max length: %d",
        artifacts_url,
        content_length,
        SERVER_CONFIG.gitlab.max_artifact_size,
    )
    return content_length <= SERVER_CONFIG.gitlab.max_artifact_size


async def submit_log_to_llm(log):
    """Stream the log to the LLM for processing"""
    # TODO: query the LLM with the log contents  # pylint: disable=fixme
    # This function will be implemented later; right now it does nothing.
    LOG.debug("Log contents:\n%s", log.read())
    return ""


async def comment_on_mr(merge_request_id: int, response: str):  # pylint: disable=unused-argument
    """Add the Log Detective response as a comment to the merge request"""
    # TODO: Implement this  # pylint: disable=fixme
    pass  # pylint: disable=unnecessary-pass
