import re
import asyncio
import zipfile
from io import BufferedRandom, BytesIO
from pathlib import Path, PurePath
from tempfile import TemporaryFile

import gitlab
import gitlab.v4
import gitlab.v4.objects
import jinja2
import aiohttp
from beeai_framework.backend import ChatModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_before_delay,
    wait_exponential_jitter,
)

from logdetective.utils import (
    run_blocking,
    sanitize_artifact,
    ContentSizeCheck,
    check_content_size,
)
from logdetective.config import SERVER_CONFIG, LOG
from logdetective.exceptions import (
    LogsTooLargeError,
    LogDetectiveConnectionError,
    LogDetectiveArtifactsMissingError,
    LogDetectiveInferenceError,
)
from logdetective.agent.agent import analyze_artifacts
from logdetective.metric import add_new_metrics, update_metrics
from logdetective.models import (
    GitLabInstanceConfig,
    JobHook,
    APIResponse,
    BuildMetadata,
)
from logdetective.database.models import (
    AnalyzeRequestMetrics,
    Comments,
    EndpointType,
    Forge,
    GitlabMergeRequestJobs,
)
from logdetective.utils import connection_error_giveup

MR_REGEX = re.compile(r"refs/merge-requests/(\d+)/.*$")
FAILURE_LOG_REGEX = re.compile(r"(\w*\.log)")

# see https://ansi.tools/lookup for reference
# this is to remove 'terminal' color tags, cursor movement, clean screen,
# and similar escape sequences from the trace log for drain to work properly
ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# The following regex follows the trace log structure for packages
# from redhat/centos-stream/rpms - based on data collected from Sep-Nov 2026.
# Example: https://gitlab.com/api/v4/projects/23664088/jobs/17969065313/trace
# In cases where kojilogs/ was empty, we can still find some relevant log file
# for analysis if we find "StepFailed" in the trace logs.
# The structure of the trace log is not in our control and may change,
# so this regex might need to be revisited for LD to be able to select the correct
# log from the trace and not fallback to trace log itself for analysis.
# Note that we use this regex AFTER the trace was stripped of the ANSI
# control/escape sequences.

# "step-name    StepFailed   https://gitlab.../build_logs/step-name.log\n"
STEP_FAILED_LOG_REGEX = re.compile(
    r"(?m)^[\w.-]+[ \t]+StepFailed[ \t]+(https?://\S+\.log)[ \t]*$"
)
ZIP_ENTRY_CHUNK_SIZE = 65536


def read_zip_entry_chunked(
    zip_file: zipfile.ZipFile, filename: str, limit_bytes: int
) -> bytes:
    """Read a zip entry in chunks, aborting if size exceeds limit_bytes."""
    content = BytesIO()
    with zip_file.open(filename) as entry:
        while True:
            chunk = entry.read(ZIP_ENTRY_CHUNK_SIZE)
            if not chunk:
                break
            if content.tell() + len(chunk) > limit_bytes:
                raise LogsTooLargeError(
                    f"Decompressing {filename} exceeded {limit_bytes} bytes"
                )
            content.write(chunk)
    return content.getvalue()


def create_job_context(job: gitlab.v4.objects.ProjectJob):
    """Custom gitlab job identifier for logging."""
    return f"project {job.project_id}, job {job.id}, MR {job.project_name}!{job.mr_iid}"


def create_job_url(gitlab_cfg: GitLabInstanceConfig, job: gitlab.v4.objects.ProjectJob) -> str:
    """Construct URL for the gitlab job."""
    return (
        f"{gitlab_cfg.url.strip('/')}/"
        f"{gitlab_cfg.api_path.strip('/')}/"
        f"projects/{job.project_id}/"
        f"jobs/{job.id}"
    )


def select_log_from_trace(
    gitlab_cfg: GitLabInstanceConfig,
    job: gitlab.v4.objects.ProjectJob,
    tempfile: BufferedRandom,
    trace: str,
) -> tuple[str, str]:
    """Identify failed pipeline step in the log trace.
    Return URL and contents of the log (from artifacts.zip loaded in tempfile or the trace itself).
    """
    job_ctx = create_job_context(job)
    artifact_prefix = f"{create_job_url(gitlab_cfg, job)}/artifacts/"
    tempfile.seek(0)
    with zipfile.ZipFile(tempfile, mode="r") as artifacts_zip:
        for match in STEP_FAILED_LOG_REGEX.finditer(trace):
            log_url = match.group(1)
            if not log_url.startswith(artifact_prefix):
                continue
            log_path = log_url[len(artifact_prefix):]
            if not log_path.startswith(("build_logs/", "kojilogs/")):
                continue
            try:
                zipinfo = artifacts_zip.getinfo(log_path)
            except KeyError:
                LOG.warning("%s (StepFailed) not in artifacts.zip - %s", log_path, job_ctx)
                continue
            if zipinfo.is_dir() or not zipinfo.file_size:
                continue
            if zipinfo.file_size > gitlab_cfg.max_artifact_size:
                LOG.warning(
                    "%s (StepFailed) is larger than %d bytes - %s",
                    log_path,
                    gitlab_cfg.max_artifact_size,
                    job_ctx
                )
                continue
            try:
                log_content = read_zip_entry_chunked(
                    artifacts_zip, log_path, gitlab_cfg.max_artifact_size
                ).decode("utf-8", errors="replace")
            except LogsTooLargeError:
                LOG.warning(
                    "%s (StepFailed) is larger than %d bytes (during read) - %s",
                    log_path,
                    gitlab_cfg.max_artifact_size,
                    job_ctx,
                )
                continue
            LOG.info("Selected log %s from trace - %s", log_path, job_ctx)
            return log_url, log_content

    LOG.info("Selecting trace log for analysis - %s", job_ctx)
    return f"{create_job_url(gitlab_cfg, job)}/trace", trace


# pylint: disable=too-many-locals
def select_koji_log(
    gitlab_cfg: GitLabInstanceConfig,
    job: gitlab.v4.objects.ProjectJob,
    tempfile: BufferedRandom,
) -> tuple[str, str] | None:
    """Select and open the relevant log from a downloaded artifact archive."""
    failed_arches = {}
    job_ctx = create_job_context(job)
    with zipfile.ZipFile(tempfile, mode="r") as artifacts_zip:
        size_sum = 0
        for zipinfo in artifacts_zip.infolist():
            if not zipinfo.filename.endswith("task_failed.log"):
                continue
            # The koji logs store this file in two places: 1) in the
            # directory with the failed architecture and 2) in the parent
            # directory. Most of the time, we want to ignore the one in the
            # parent directory, since the rest of the information is in the
            # specific task directory. However, there are some situations
            # where non-build failures (such as "Target build already exists")
            # may be presented only at the top level.
            # The paths look like `kojilogs/noarch-XXXXXX/task_failed.log`
            # or `kojilogs/noarch-XXXXXX/x86_64-XXXXXX/task_failed.log`
            # We prefix "toplevel" with '~' so that later when we sort the
            # keys to see if there are any unrecognized arches, it will always
            # sort last.
            path = PurePath(zipinfo.filename)
            if len(path.parts) <= 3:
                failed_arches["~toplevel"] = path
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
            contents = read_zip_entry_chunked(
                artifacts_zip, zipinfo.filename, gitlab_cfg.max_artifact_size - size_sum
            )
            size_sum += len(contents)
            decoded_contents = contents.decode("utf-8")
            match = FAILURE_LOG_REGEX.search(decoded_contents)
            if match:
                failed_arches[architecture] = PurePath(path.parent, match.group(1))
            else:
                # The best thing we can do at this point is return the
                # task_failed.log, since it will probably contain the most
                # relevant information
                LOG.info(
                    "task_failed.log does not indicate which log contains the failure - %s",
                    job_ctx,
                )
                failed_arches[architecture] = path

        if not failed_arches:
            LOG.warning("Could not detect failed architecture - %s", job_ctx)
            return None

        # We only want to handle one arch, so we'll check them in order of
        # "most to least likely for the maintainer to have access to hardware"
        # This means: x86_64 > aarch64 > riscv > ppc64le > s390x
        # If no architecture matches `preferred_arches` list just
        # pick the first alphabetically. If the issue was a Koji error
        # rather than a build failure, this will fall back to ~toplevel as
        # the lowest-sorting possibility.
        preferred_arches = (
            "x86_64",
            "aarch64",
            "riscv",
            "ppc64le",
            "s390x",
            "noarch",
        )
        failed_arch = next(
            (arch for arch in preferred_arches if arch in failed_arches),
            sorted(failed_arches)[0],
        )
        log_path = failed_arches[failed_arch].as_posix()
        if failed_arch not in preferred_arches and failed_arch != "~toplevel":
            LOG.info(
                "No preferred failed architecture found; selecting %s for - %s ",
                failed_arch,
                job_ctx,
            )
        LOG.info("Selected failed log %s - %s", log_path, job_ctx)
        log_url = f"{create_job_url(gitlab_cfg, job)}/artifacts/{log_path}"
        log_content = read_zip_entry_chunked(
            artifacts_zip, log_path, gitlab_cfg.max_artifact_size - size_sum
        ).decode("utf-8")
        LOG.debug("Returning contents of %s%s", gitlab_cfg.url, log_url)
        return log_url, log_content


def construct_build_metadata(log_url: str) -> BuildMetadata:
    """Based on the log, construct a commentary for the LD request."""
    if "/artifacts/build_logs/" in log_url:
        build_commentary = (
            f"This log is from a failed pipeline step - {log_url.split('/')[-1]}. "
            "If the log's name does not start with 'rpmbuild', "
            "the failure occurred before the RPM build."
        )
    elif log_url.endswith("/trace"):
        build_commentary = (
            "This is a trace log of the GitLab job. "
            "It is submitted when something else than an RPM build failed, "
            "and it was also not possible to detect which pipeline step failed."
        )
    else:
        build_commentary = (
            "The package was built in Koji, using Mock. "
            "Mock provides a temporary chroot for the build environment. "
            "This environment can't be directly modified by the packager."
        )
    return BuildMetadata(commentary=build_commentary)


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-return-statements
async def process_gitlab_job_event(
    gitlab_cfg: GitLabInstanceConfig,
    gitlab_connection: gitlab.Gitlab,
    http_session: aiohttp.ClientSession,
    forge: Forge,
    job_hook: JobHook,
    chat_model: ChatModel,
    api_token_name: str | None = None,
    metrics_id: int | None = None,
) -> APIResponse | None:
    """Handle a received job_event webhook from GitLab"""
    LOG.debug("Received webhook message from %s:\n%s", forge.value, job_hook)

    # Look up the project this job belongs to
    project = await run_blocking(gitlab_connection.projects.get, job_hook.project_id)

    # check if this project is on the opt-in list for analysis.
    if not is_eligible_package(project.name):
        LOG.info("Not analysing failed build of unrecognized package %s", project.name)
        return

    # Retrieve data about the job from the GitLab API
    job = await run_blocking(project.jobs.get, job_hook.build_id)
    LOG.info("Processing failed GitLab job %s", job.web_url)

    # Retrieve the pipeline that started this job
    pipeline = await run_blocking(project.pipelines.get, job_hook.pipeline_id)

    # Verify this is a merge request
    if pipeline.source != "merge_request_event":
        LOG.info("Not a merge request pipeline. Ignoring.")
        return

    # Extract the merge-request IID from the job
    mr_number_match = MR_REGEX.search(pipeline.ref)
    if not mr_number_match:
        LOG.error(
            "Pipeline source is merge_request_event but no merge request IID was provided."
        )
        return

    # For easy retrieval later, add some project details used in logs and comments.
    job.project_name = project.name
    job.project_id = job_hook.project_id
    job.project_url = project.web_url
    job.mr_iid = int(mr_number_match.group(1))
    job_ctx = create_job_context(job)

    # Check if this is a resubmission of an existing, completed job.
    # If it is, we'll exit out here and not waste time retrieving the logs,
    # running a new analysis or trying to submit a new comment.
    mr_job_db = await GitlabMergeRequestJobs.get_by_details(
        forge=forge,
        project_id=project.id,
        mr_iid=job.mr_iid,
        job_id=job_hook.build_id,
    )
    if mr_job_db:
        LOG.info("Resubmission of an existing build. Skipping.")
        return

    LOG.debug("Retrieving log artifacts")
    # Retrieve the build logs from the merge request artifacts and preprocess them
    try:
        log_url, preprocessed_log = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg, job, http_session
        )
    except (
        LogsTooLargeError,
        LogDetectiveConnectionError,
        LogDetectiveArtifactsMissingError,
    ) as ex:
        LOG.error("Could not retrieve logs in %s: %s", job_ctx, ex)
        return

    if metrics_id is None:
        metrics_id = await add_new_metrics(
            api_name=EndpointType.ANALYZE_GITLAB_JOB,
            api_token_name=api_token_name,
        )

    log_name = "job-trace.log" if log_url.endswith("/trace") else PurePath(log_url).name
    log_text = await run_blocking(sanitize_artifact, preprocessed_log)
    build_metadata = construct_build_metadata(log_url)

    try:
        response = await analyze_artifacts(
            artifacts={log_name: log_text},
            chat_model=chat_model,
            build_metadata=build_metadata,
        )
    except LogDetectiveInferenceError as exc:
        LOG.error("%s during %s: %s", type(exc).__name__, job_ctx, exc)
        return

    # Add the Log Detective response as a comment to the merge request
    await comment_on_mr(
        forge,
        project,
        job.mr_iid,
        job,
        log_url,
        response,
        metrics_id,
    )

    await update_metrics(metrics_id, response)

    return response


def is_eligible_package(project_name: str):
    """Check whether the provided package name is eligible for analysis of
    builds of the merge request"""

    # First check the allow-list. If it's not allowed, we deny.
    allowed = False
    for pattern in SERVER_CONFIG.general.packages:
        LOG.debug("include %s", pattern)
        if re.search(pattern, project_name):
            allowed = True
            break
    if not allowed:
        # The project did not match any of the permitted regular expressions
        return False

    # Next, check the deny-list. If it was allowed before, but denied here, we deny.
    for pattern in SERVER_CONFIG.general.excluded_packages:
        LOG.debug("exclude %s", pattern)
        if re.search(pattern, project_name):
            return False

    # It was allowed and not denied, so return True to indicate it is eligible
    return True


def read_job_trace(job: gitlab.v4.objects.ProjectJob, max_size: int) -> str:
    """Stream a job trace without retaining more than the artifact size limit."""
    trace_buffer = BytesIO()
    job_ctx = create_job_context(job)

    trace_chunks = job.trace(
        streamed=True, iterator=True, chunk_size=ZIP_ENTRY_CHUNK_SIZE
    )
    for chunk in trace_chunks:
        if not chunk:
            continue
        if trace_buffer.tell() + len(chunk) > max_size:
            raise LogsTooLargeError(
                f"Trace is larger than allowed {max_size} bytes - {job_ctx}"
            )
        trace_buffer.write(chunk)
    trace_bytes = trace_buffer.getvalue()
    trace_clean = ANSI_ESCAPE_REGEX.sub(
        "", trace_bytes.decode("utf-8", errors="replace")
    ).replace("\r\n", "\n").replace("\r", "\n")

    return trace_clean


# pylint: disable=too-many-branches
@retry(
    stop=stop_before_delay(60),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception_type(ConnectionResetError),
    retry_error_callback=connection_error_giveup,
)
async def retrieve_and_preprocess_koji_logs(
    gitlab_cfg: GitLabInstanceConfig,
    job: gitlab.v4.objects.ProjectJob,
    http_session: aiohttp.ClientSession,
) -> tuple[str, str]:
    """Select the relevant log from job artifacts or the job trace.
    1. Search `kojilogs` for `task_failed.log` to identify the failed Koji log.
    2. If no Koji log is selected, search the trace for a valid `StepFailed` link in `build_logs`.
    3. If no log is selected from artifacts, analyze the job trace itself.

    Return URL and content of the selected log.
    """

    job_ctx = create_job_context(job)

    try:
        artifacts_are_small_enough = await check_artifacts_file_size(
            gitlab_cfg, job, http_session
        )
    except LogDetectiveArtifactsMissingError:
        tempfile = None
    else:
        if not artifacts_are_small_enough:
            raise LogsTooLargeError(f"Oversized logs for {job_ctx}")
        # Keep the archive open until trace selection has checked linked entries.
        tempfile = TemporaryFile(mode="w+b")

    fallback_to_trace = tempfile is None
    try:
        if tempfile:
            try:
                await run_blocking(job.artifacts, streamed=True, action=tempfile.write)
            except gitlab.exceptions.GitlabGetError as exc:
                if exc.response_code != 404:
                    raise
                fallback_to_trace = True
            else:
                archive_size = tempfile.tell()
                LOG.info("Downloaded artifacts.zip (%s bytes) - %s", archive_size, job_ctx)
                selected_log = await run_blocking(select_koji_log, gitlab_cfg, job, tempfile)
                if selected_log:
                    return selected_log

        try:
            trace_text = await run_blocking(read_job_trace, job, gitlab_cfg.max_artifact_size)
        except gitlab.exceptions.GitlabError as exc:
            raise LogDetectiveArtifactsMissingError(
                f"Could not retrieve trace - {job_ctx} - {exc}"
            ) from exc
        if not trace_text:
            raise LogDetectiveArtifactsMissingError(
                f"Empty trace for job {job.id} ({job.project_name}#{job.mr_iid})"
            )
        if fallback_to_trace:
            LOG.info("Using job trace because artifacts are unavailable for %s", job_ctx)
            return f"{create_job_url(gitlab_cfg, job)}/trace", trace_text

        selected_log = await run_blocking(
            select_log_from_trace, gitlab_cfg, job, tempfile, trace_text
        )
        return selected_log
    finally:
        if tempfile:
            await run_blocking(tempfile.close)


async def check_artifacts_file_size(
    gitlab_cfg: GitLabInstanceConfig,
    job: gitlab.v4.objects.ProjectJob,
    http_session: aiohttp.ClientSession,
):
    """Method to determine if the artifacts are too large to process

    First, make sure that the artifacts are of a reasonable size.
    The zipped artifact collection will be stored in memory below.
    The python-gitlab library doesn't expose a way to check this value directly,
    so we need to interact with directly with the headers.

    If the `Content-Length` is not set or is invalid, treat artifacts as too large (reject).

    Raises:
        LogDetectiveArtifactsMissingError in case error status code 404
        LogDetectiveConnectionError in case of other error status codes
    Return:
        True if Content-Length is present, valid, and within the size limit; False otherwise.
    """
    artifacts_path = (
        f"{gitlab_cfg.api_path}/projects/{job.project_id}/jobs/{job.id}/artifacts"
    )
    LOG.debug("checking artifact URL %s%s", gitlab_cfg.url, artifacts_path)
    try:
        head_response = await http_session.head(
            artifacts_path,
            allow_redirects=True,
            raise_for_status=True,
        )
    except aiohttp.ClientResponseError as ex:
        if ex.status == 404:
            raise LogDetectiveArtifactsMissingError(
                f"Requested artifacts are not available - {create_job_context(job)}"
            ) from ex
        raise LogDetectiveConnectionError(
            f"Connection error checking artifacts - {create_job_context(job)}"
        ) from ex
    LOG.debug("Checking content-length for %s", artifacts_path)
    size_check: ContentSizeCheck = check_content_size(
        head_response.headers, gitlab_cfg.max_artifact_size
    )
    LOG.debug(
        "URL: %s, content-length: %s, max length: %d",
        artifacts_path,
        str(size_check.size_in_bytes),
        gitlab_cfg.max_artifact_size,
    )
    return size_check.proceed


async def comment_on_mr(  # pylint: disable=too-many-arguments disable=too-many-positional-arguments
    forge: Forge,
    project: gitlab.v4.objects.Project,
    merge_request_iid: int,
    job: gitlab.v4.objects.ProjectJob,
    log_url: str,
    response: APIResponse,
    metrics_id: int,
):
    """Add the Log Detective response as a comment to the merge request"""
    LOG.debug(
        "Primary Explanation for %s MR %d: %s",
        project.name,
        merge_request_iid,
        response.explanation,
    )

    # First, we'll see if there's an existing comment on this Merge Request
    # and wrap it in <details></details> to reduce noise.
    await suppress_latest_comment(forge, project, merge_request_iid)

    # Get the formatted short comment.
    short_comment = await run_blocking(
        generate_mr_comment,
        job,
        log_url,
        response,
        full=False,
    )

    # Look up the merge request
    merge_request = await run_blocking(project.mergerequests.get, merge_request_iid)

    # Submit a new comment to the Merge Request using the Gitlab API
    discussion = await run_blocking(
        merge_request.discussions.create, {"body": short_comment}
    )

    # Get the ID of the first note
    note_id = discussion.attributes["notes"][0]["id"]
    note = await run_blocking(discussion.notes.get, note_id)

    # Update the comment with the full details
    # We do this in a second step so we don't bombard the user's email
    # notifications with a massive message. Gitlab doesn't send email for
    # comment edits.
    full_comment = await run_blocking(
        generate_mr_comment,
        job,
        log_url,
        response,
        full=True,
    )
    note.body = full_comment

    # Pause for five seconds before sending the snippet data, otherwise
    # Gitlab may bundle the edited message together with the creation
    # message in email.
    await asyncio.sleep(5)
    await run_blocking(note.save)

    # Save the new comment to the database
    metrics = await AnalyzeRequestMetrics.get_metric_by_id(metrics_id)
    await Comments.create(
        forge,
        project.id,
        merge_request_iid,
        job.id,
        str(discussion.id),
        metrics,
    )


async def suppress_latest_comment(
    gitlab_instance: Forge,
    project: gitlab.v4.objects.Project,
    merge_request_iid: int,
) -> None:
    """Look up the latest comment on this Merge Request, if any, and wrap it
    in a <details></details> block with a comment indicating that it has been
    superseded by a new push."""

    # Ask the database for the last known comment for this MR
    previous_comment = await Comments.get_latest_comment(
        gitlab_instance, project.id, merge_request_iid
    )

    if previous_comment is None:
        # No existing comment, so nothing to do.
        return

    # Retrieve its content from the Gitlab API

    # Look up the merge request
    merge_request = await run_blocking(project.mergerequests.get, merge_request_iid)

    # Find the discussion matching the latest comment ID
    discussion = await run_blocking(
        merge_request.discussions.get, previous_comment.comment_id
    )

    # Get the ID of the first note
    note_id = discussion.attributes["notes"][0]["id"]
    note = await run_blocking(discussion.notes.get, note_id)

    # Wrap the note in <details>, indicating why.
    note.body = (
        "This comment has been superseded by a newer "
        f"Log Detective analysis.\n<details>\n{note.body}\n</details>"
    )
    await run_blocking(note.save)


def generate_mr_comment(
    job: gitlab.v4.objects.ProjectJob,
    log_url: str,
    response: APIResponse,
    full: bool = True,
) -> str:
    """Use a template to generate a comment string to submit to Gitlab"""

    # Locate and load the comment template
    script_path = Path(__file__).resolve().parent
    template_path = Path(script_path, "templates")
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path), autoescape=True
    )

    if full:
        tpl = jinja_env.get_template("gitlab_full_comment.md.j2")
    else:
        tpl = jinja_env.get_template("gitlab_short_comment.md.j2")

    artifacts_url = f"{job.project_url}/-/jobs/{job.id}/artifacts/download"

    # Generate the comment from the template
    content = tpl.render(
        package=job.project_name,
        explanation=response.explanation,
        snippets=response.snippets,
        log_url=log_url,
        artifacts_url=artifacts_url,
    )

    return content
