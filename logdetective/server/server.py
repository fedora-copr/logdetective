import asyncio
import json
import os
import re
import zipfile
from enum import Enum
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from tempfile import TemporaryFile
from typing import List, Annotated, Tuple, Dict, Any, Union
from io import BytesIO

import backoff
import matplotlib
import matplotlib.pyplot
from aiohttp import StreamReader
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request

from fastapi.responses import StreamingResponse
from fastapi.responses import Response as BasicResponse
import gitlab
import gitlab.v4
import gitlab.v4.objects
import jinja2
import aiohttp
import sqlalchemy
import sentry_sdk

import logdetective.server.database.base

from logdetective.extractors import DrainExtractor
from logdetective.utils import (
    compute_certainty,
    format_snippets,
    load_prompts,
)
from logdetective.server.utils import (
    load_server_config,
    get_log,
    format_analyzed_snippets,
)
from logdetective.server.metric import track_request, add_new_metrics, update_metrics
from logdetective.server.models import (
    BuildLog,
    JobHook,
    Response,
    StagedResponse,
    Explanation,
    AnalyzedSnippet,
    TimePeriod,
)
from logdetective.server import plot as plot_engine
from logdetective.server.remote_log import RemoteLog
from logdetective.server.database.models import (
    Comments,
    EndpointType,
    Forge,
)
from logdetective.server.database.models import AnalyzeRequestMetrics

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

if sentry_dsn := SERVER_CONFIG.general.sentry_dsn:
    sentry_sdk.init(dsn=str(sentry_dsn), traces_sample_rate=1.0)


@asynccontextmanager
async def lifespan(fapp: FastAPI):
    """
    Establish one HTTP session
    """
    fapp.http = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=int(LOG_SOURCE_REQUEST_TIMEOUT), connect=3.07
        )
    )

    # Ensure that the database is initialized.
    logdetective.server.database.base.init()

    yield
    await fapp.http.close()


async def get_http_session(request: Request) -> aiohttp.ClientSession:
    """
    Return the single aiohttp ClientSession for this app
    """
    return request.app.http


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


app = FastAPI(dependencies=[Depends(requires_token_when_set)], lifespan=lifespan)
app.gitlab_conn = gitlab.Gitlab(
    url=SERVER_CONFIG.gitlab.url, private_token=SERVER_CONFIG.gitlab.api_token
)


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
    http: aiohttp.ClientSession,
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    stream: bool,
) -> Any:
    """Send request to selected API endpoint. Verifying successful request unless
    the using the stream response.

    url:
    data:
    headers:
    stream:
    """
    LOG.debug("async request %s headers=%s data=%s", url, headers, data)
    response = await http.post(
        url,
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
        LOG.error("Error encountered while parsing llama server response: %s", ex)
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
    return exc.status < 500


def we_give_up(details: backoff._typing.Details):
    """
    retries didn't work (or we got a different exc)
    we give up and raise proper 500 for our API endpoint
    """
    LOG.error("Inference error: %s", details["args"])
    raise HTTPException(500, "Request to the inference API failed")


@backoff.on_exception(
    backoff.expo,
    aiohttp.ClientResponseError,
    max_tries=3,
    giveup=should_we_giveup,
    raise_on_giveup=False,
    on_giveup=we_give_up,
)
async def submit_text(  # pylint: disable=R0913,R0917
    http: aiohttp.ClientSession,
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
            http, text, headers, max_tokens, log_probs > 0, stream, model
        )
    return await submit_text_completions(
        http, text, headers, max_tokens, log_probs, stream, model
    )


async def submit_text_completions(  # pylint: disable=R0913,R0917
    http: aiohttp.ClientSession,
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
        http,
        f"{SERVER_CONFIG.inference.url}/v1/completions",
        data,
        headers,
        stream,
    )

    return Explanation(
        text=response["choices"][0]["text"], logprobs=response["choices"][0]["logprobs"]
    )


async def submit_text_chat_completions(  # pylint: disable=R0913,R0917
    http: aiohttp.ClientSession,
    text: str,
    headers: dict,
    max_tokens: int = -1,
    log_probs: int = 1,
    stream: bool = False,
    model: str = "default-model",
) -> Union[Explanation, StreamReader]:
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
        http,
        f"{SERVER_CONFIG.inference.url}/v1/chat/completions",
        data,
        headers,
        stream,
    )

    if stream:
        return response
    return Explanation(
        text=response["choices"][0]["message"]["content"],
        logprobs=response["choices"][0]["logprobs"]["content"],
    )


@app.post("/analyze", response_model=Response)
@track_request()
async def analyze_log(
    build_log: BuildLog, http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    remote_log = RemoteLog(build_log.url, http_session)
    log_text = await remote_log.process_url()
    log_summary = mine_logs(log_text)
    log_summary = format_snippets(log_summary)
    response = await submit_text(
        http_session,
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


@track_request()
@app.post("/analyze/staged", response_model=StagedResponse)
async def analyze_log_staged(
    build_log: BuildLog, http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Provide endpoint for log file submission and analysis.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    remote_log = RemoteLog(build_log.url, http_session)
    log_text = await remote_log.process_url()

    return await perform_staged_analysis(http_session, log_text=log_text)


async def perform_staged_analysis(
    http: aiohttp.ClientSession, log_text: str
) -> StagedResponse:
    """Submit the log file snippets to the LLM and retrieve their results"""
    log_summary = mine_logs(log_text)

    # Process snippets asynchronously
    analyzed_snippets = await asyncio.gather(
        *[
            submit_text(
                http,
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
        http,
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
async def analyze_log_stream(
    build_log: BuildLog, http_session: aiohttp.ClientSession = Depends(get_http_session)
):
    """Stream response endpoint for Logdetective.
    Request must be in form {"url":"<YOUR_URL_HERE>"}.
    URL must be valid for the request to be passed to the LLM server.
    Meaning that it must contain appropriate scheme, path and netloc,
    while lacking  result, params or query fields.
    """
    remote_log = RemoteLog(build_log.url, http_session)
    log_text = await remote_log.process_url()
    log_summary = mine_logs(log_text)
    log_summary = format_snippets(log_summary)
    headers = {"Content-Type": "application/json"}

    if SERVER_CONFIG.inference.api_token:
        headers["Authorization"] = f"Bearer {SERVER_CONFIG.inference.api_token}"

    try:
        stream = await submit_text_chat_completions(
            http_session,
            PROMPT_CONFIG.prompt_template.format(log_summary),
            stream=True,
            headers=headers,
            model=SERVER_CONFIG.inference.model,
            max_tokens=SERVER_CONFIG.inference.max_tokens,
        )
    except aiohttp.ClientResponseError as ex:
        raise HTTPException(
            status_code=400,
            detail="HTTP Error while getting response from inference server "
            f"[{ex.status}] {ex.message}",
        ) from ex

    # we need to figure out a better response here, this is how it looks rn:
    # b'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],
    #   "created":1744818071,"id":"chatcmpl-c9geTxNcQO7M9wR...
    return StreamingResponse(stream)


@app.post("/webhook/gitlab/job_events")
async def receive_gitlab_job_event_webhook(
    x_gitlab_instance: Annotated[str | None, Header()],
    job_hook: JobHook,
    background_tasks: BackgroundTasks,
    http: aiohttp.ClientSession = Depends(get_http_session),
):
    """Webhook endpoint for receiving job_events notifications from GitLab
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    lists the full specification for the messages sent for job events."""

    try:
        forge = Forge(x_gitlab_instance)
    except ValueError:
        LOG.critical("%s is not a recognized forge. Ignoring.", x_gitlab_instance)
        return BasicResponse(status_code=400)

    # Handle the message in the background so we can return 200 immediately
    background_tasks.add_task(process_gitlab_job_event, http, forge, job_hook)

    # No return value or body is required for a webhook.
    # 204: No Content
    return BasicResponse(status_code=204)


async def process_gitlab_job_event(
    http: aiohttp.ClientSession,
    forge: Forge,
    job_hook: JobHook,
):
    """Handle a received job_event webhook from GitLab"""
    LOG.debug("Received webhook message from %s:\n%s", forge.value, job_hook)

    # Look up the project this job belongs to
    project = await asyncio.to_thread(app.gitlab_conn.projects.get, job_hook.project_id)
    LOG.info("Processing failed job for %s", project.name)

    # Retrieve data about the job from the GitLab API
    job = await asyncio.to_thread(project.jobs.get, job_hook.build_id)

    # For easy retrieval later, we'll add project_name and project_url to the
    # job object
    job.project_name = project.name
    job.project_url = project.web_url

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
    merge_request_iid = int(match.group(1))

    LOG.debug("Retrieving log artifacts")
    # Retrieve the build logs from the merge request artifacts and preprocess them
    try:
        log_url, preprocessed_log = await retrieve_and_preprocess_koji_logs(http, job)
    except LogsTooLargeError:
        LOG.error("Could not retrieve logs. Too large.")
        raise

    # Submit log to Log Detective and await the results.
    log_text = preprocessed_log.read().decode(encoding="utf-8")
    metrics_id = await add_new_metrics(
        api_name=EndpointType.ANALYZE_GITLAB_JOB,
        url=log_url,
        http_session=http,
        compressed_log_content=RemoteLog.zip_text(log_text),
    )
    staged_response = await perform_staged_analysis(http, log_text=log_text)
    update_metrics(metrics_id, staged_response)
    preprocessed_log.close()

    # check if this project is on the opt-in list for posting comments.
    if project.name not in SERVER_CONFIG.general.packages:
        LOG.info("Not publishing comment for unrecognized package %s", project.name)
        return

    # Add the Log Detective response as a comment to the merge request
    await comment_on_mr(
        forge,
        project,
        merge_request_iid,
        job,
        log_url,
        staged_response,
        metrics_id,
    )

    return staged_response


class LogsTooLargeError(RuntimeError):
    """The log archive exceeds the configured maximum size"""


async def retrieve_and_preprocess_koji_logs(
    http: aiohttp.ClientSession, job: gitlab.v4.objects.ProjectJob
):  # pylint: disable=too-many-branches
    """Download logs from the merge request artifacts

    This function will retrieve the build logs and do some minimal
    preprocessing to determine which log is relevant for analysis.

    returns: The URL pointing to the selected log file and an open, file-like
    object containing the log contents to be sent for processing by Log
    Detective. The calling function is responsible for closing this object."""

    # Make sure the file isn't too large to process.
    if not await check_artifacts_file_size(http, job):
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
    artifacts_zip = zipfile.ZipFile(tempfile, mode="r")  # pylint: disable=consider-using-with
    for zipinfo in artifacts_zip.infolist():
        if zipinfo.filename.endswith("task_failed.log"):
            # The koji logs store this file in two places: 1) in the
            # directory with the failed architecture and 2) in the parent
            # directory. Most of the time, we want to ignore the one in the
            # parent directory, since the rest of the information is in the
            # specific task directory. However, there are some situations
            # where non-build failures (such as "Target build already exists")
            # may be presented only at the top level.
            # The paths look like `kojilogs/noarch-XXXXXX/task_failed.log`
            # or `kojilogs/noarch-XXXXXX/x86_64-XXXXXX/task_failed.log`
            path = PurePath(zipinfo.filename)
            if len(path.parts) <= 3:
                failed_arches["toplevel"] = path
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
        # No failed task found in the sub-tasks.
        raise FileNotFoundError("Could not detect failed architecture.")

    # We only want to handle one arch, so we'll check them in order of
    # "most to least likely for the maintainer to have access to hardware"
    # This means: x86_64 > aarch64 > riscv > ppc64le > s390x
    if "x86_64" in failed_arches:
        failed_arch = "x86_64"
    elif "aarch64" in failed_arches:
        failed_arch = "aarch64"
    elif "riscv" in failed_arches:
        failed_arch = "riscv"
    elif "ppc64le" in failed_arches:
        failed_arch = "ppc64le"
    elif "s390x" in failed_arches:
        failed_arch = "s390x"
    elif "noarch" in failed_arches:
        # May have failed during BuildSRPMFromSCM phase
        failed_arch = "noarch"
    elif "toplevel" in failed_arches:
        # Probably a Koji-specific error, not a build error
        failed_arch = "toplevel"
    else:
        # We have one or more architectures that we don't know about? Just
        # pick the first alphabetically.
        failed_arch = sorted(list(failed_arches.keys()))[0]

    LOG.debug("Failed architecture: %s", failed_arch)

    log_path = failed_arches[failed_arch].as_posix()

    log_url = f"{SERVER_CONFIG.gitlab.api_url}/projects/{job.project_id}/jobs/{job.id}/artifacts/{log_path}"  # pylint: disable=line-too-long
    LOG.debug("Returning contents of %s", log_url)

    # Return the log as a file-like object with .read() function
    return log_url, artifacts_zip.open(log_path)


async def check_artifacts_file_size(
    http: aiohttp.ClientSession,
    job: gitlab.v4.objects.ProjectJob,
):
    """Method to determine if the artifacts are too large to process"""
    # First, make sure that the artifacts are of a reasonable size. The
    # zipped artifact collection will be stored in memory below. The
    # python-gitlab library doesn't expose a way to check this value directly,
    # so we need to interact with directly with the headers.
    artifacts_url = f"{SERVER_CONFIG.gitlab.api_url}/projects/{job.project_id}/jobs/{job.id}/artifacts"  # pylint: disable=line-too-long
    LOG.debug("checking artifact URL %s", artifacts_url)
    try:
        head_response = await http.head(
            artifacts_url,
            allow_redirects=True,
            headers={"Authorization": f"Bearer {SERVER_CONFIG.gitlab.api_token}"},
            timeout=5,
            raise_for_status=True,
        )
    except aiohttp.ClientResponseError as ex:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to check artifact URL: [{ex.status}] {ex.message}",
        ) from ex
    content_length = int(head_response.headers.get("content-length"))
    LOG.debug(
        "URL: %s, content-length: %d, max length: %d",
        artifacts_url,
        content_length,
        SERVER_CONFIG.gitlab.max_artifact_size,
    )
    return content_length <= SERVER_CONFIG.gitlab.max_artifact_size


async def comment_on_mr(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
    forge: Forge,
    project: gitlab.v4.objects.Project,
    merge_request_iid: int,
    job: gitlab.v4.objects.ProjectJob,
    log_url: str,
    response: StagedResponse,
    metrics_id: int,
):
    """Add the Log Detective response as a comment to the merge request"""
    LOG.debug(
        "Primary Explanation for %s MR %d: %s",
        project.name,
        merge_request_iid,
        response.explanation.text,
    )

    # First, we'll see if there's an existing comment on this Merge Request
    # and wrap it in <details></details> to reduce noise.
    await suppress_latest_comment(forge, project, merge_request_iid)

    # Get the formatted short comment.
    short_comment = await generate_mr_comment(job, log_url, response, full=False)

    # Look up the merge request
    merge_request = await asyncio.to_thread(
        project.mergerequests.get, merge_request_iid
    )

    # Submit a new comment to the Merge Request using the Gitlab API
    discussion = await asyncio.to_thread(
        merge_request.discussions.create, {"body": short_comment}
    )

    # Get the ID of the first note
    note_id = discussion.attributes["notes"][0]["id"]
    note = discussion.notes.get(note_id)

    # Update the comment with the full details
    # We do this in a second step so we don't bombard the user's email
    # notifications with a massive message. Gitlab doesn't send email for
    # comment edits.
    full_comment = await generate_mr_comment(job, log_url, response, full=True)
    note.body = full_comment

    # Pause for five seconds before sending the snippet data, otherwise
    # Gitlab may bundle the edited message together with the creation
    # message in email.
    await asyncio.sleep(5)
    await asyncio.to_thread(note.save)

    # Save the new comment to the database
    try:
        metrics = AnalyzeRequestMetrics.get_metric_by_id(metrics_id)
        Comments.create(
            forge,
            project.id,
            merge_request_iid,
            job.id,
            discussion.id,
            metrics,
        )
    except sqlalchemy.exc.IntegrityError:
        # We most likely attempted to save a new comment for the same
        # build job. This is somewhat common during development when we're
        # submitting requests manually. It shouldn't really happen in
        # production.
        if not SERVER_CONFIG.general.devmode:
            raise


async def suppress_latest_comment(
    gitlab_instance: str,
    project: gitlab.v4.objects.Project,
    merge_request_iid: int,
) -> None:
    """Look up the latest comment on this Merge Request, if any, and wrap it
    in a <details></details> block with a comment indicating that it has been
    superseded by a new push."""

    # Ask the database for the last known comment for this MR
    previous_comment = Comments.get_latest_comment(
        gitlab_instance, project.id, merge_request_iid
    )

    if previous_comment is None:
        # No existing comment, so nothing to do.
        return

    # Retrieve its content from the Gitlab API

    # Look up the merge request
    merge_request = await asyncio.to_thread(
        project.mergerequests.get, merge_request_iid
    )

    # Find the discussion matching the latest comment ID
    discussion = await asyncio.to_thread(
        merge_request.discussions.get, previous_comment.comment_id
    )

    # Get the ID of the first note
    note_id = discussion.attributes["notes"][0]["id"]
    note = discussion.notes.get(note_id)

    # Wrap the note in <details>, indicating why.
    note.body = (
        "This comment has been superseded by a newer "
        f"Log Detective analysis.\n<details>\n{note.body}\n</details>"
    )
    await asyncio.to_thread(note.save)


async def generate_mr_comment(
    job: gitlab.v4.objects.ProjectJob,
    log_url: str,
    response: StagedResponse,
    full: bool = True,
) -> str:
    """Use a template to generate a comment string to submit to Gitlab"""

    # Locate and load the comment template
    script_path = Path(__file__).resolve().parent
    template_path = Path(script_path, "templates")
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_path))

    if full:
        tpl = jinja_env.get_template("gitlab_full_comment.md.j2")
    else:
        tpl = jinja_env.get_template("gitlab_short_comment.md.j2")

    artifacts_url = f"{job.project_url}/-/jobs/{job.id}/artifacts/download"

    if response.response_certainty >= 90:
        emoji_face = ":slight_smile:"
    elif response.response_certainty >= 70:
        emoji_face = ":neutral_face:"
    else:
        emoji_face = ":frowning2:"

    # Generate the comment from the template
    content = tpl.render(
        package=job.project_name,
        explanation=response.explanation.text,
        certainty=f"{response.response_certainty:.2f}",
        emoji_face=emoji_face,
        snippets=response.snippets,
        log_url=log_url,
        artifacts_url=artifacts_url,
    )

    return content


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


class MetricRoute(str, Enum):
    """Routes for metrics"""

    ANALYZE = "analyze"
    ANALYZE_STAGED = "analyze-staged"
    ANALYZE_GITLAB_JOB = "analyze-gitlab"


class Plot(str, Enum):
    """Type of served plots"""

    REQUESTS = "requests"
    RESPONSES = "responses"
    BOTH = ""


ROUTE_TO_ENDPOINT_TYPES = {
    MetricRoute.ANALYZE: EndpointType.ANALYZE,
    MetricRoute.ANALYZE_STAGED: EndpointType.ANALYZE_STAGED,
    MetricRoute.ANALYZE_GITLAB_JOB: EndpointType.ANALYZE_GITLAB_JOB,
}


@app.get("/metrics/{route}/", response_class=StreamingResponse)
@app.get("/metrics/{route}/{plot}", response_class=StreamingResponse)
async def get_metrics(
    route: MetricRoute,
    plot: Plot = Plot.BOTH,
    period_since_now: TimePeriod = Depends(TimePeriod),
):
    """Get an handler for visualize statistics for the specified endpoint and plot."""
    endpoint_type = ROUTE_TO_ENDPOINT_TYPES[route]

    async def handler():
        """Show statistics for the specified endpoint and plot."""
        if plot == Plot.REQUESTS:
            fig = plot_engine.requests_per_time(period_since_now, endpoint_type)
            return _svg_figure_response(fig)
        if plot == Plot.RESPONSES:
            fig = plot_engine.average_time_per_responses(
                period_since_now, endpoint_type
            )
            return _svg_figure_response(fig)
        # BOTH
        fig_requests = plot_engine.requests_per_time(period_since_now, endpoint_type)
        fig_responses = plot_engine.average_time_per_responses(
            period_since_now, endpoint_type
        )
        return _multiple_svg_figures_response([fig_requests, fig_responses])

    descriptions = {
        Plot.REQUESTS: (
            "Show statistics for the requests received in the given period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
        Plot.RESPONSES: (
            "Show statistics for responses given in the specified period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
        Plot.BOTH: (
            "Show statistics for requests and responses in the given period of time "
            f"for the /{endpoint_type.value} API endpoint."
        ),
    }
    handler.__doc__ = descriptions[plot]

    return await handler()
