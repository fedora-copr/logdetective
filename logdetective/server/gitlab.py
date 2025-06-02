import re
import asyncio
import zipfile
from pathlib import Path, PurePath
from tempfile import TemporaryFile

from fastapi import HTTPException

import gitlab
import gitlab.v4
import gitlab.v4.objects
import jinja2
import aiohttp

from logdetective.server.config import SERVER_CONFIG, LOG
from logdetective.server.llm import perform_staged_analysis
from logdetective.server.metric import add_new_metrics, update_metrics
from logdetective.server.models import (
    GitLabInstanceConfig,
    JobHook,
    StagedResponse,
)
from logdetective.server.database.models import (
    AnalyzeRequestMetrics,
    Comments,
    EndpointType,
    Forge,
    GitlabMergeRequestJobs,
)
from logdetective.server.compressors import RemoteLogCompressor

MR_REGEX = re.compile(r"refs/merge-requests/(\d+)/.*$")
FAILURE_LOG_REGEX = re.compile(r"(\w*\.log)")


async def process_gitlab_job_event(
    gitlab_cfg: GitLabInstanceConfig,
    forge: Forge,
    job_hook: JobHook,
):  # pylint: disable=too-many-locals
    """Handle a received job_event webhook from GitLab"""
    LOG.debug("Received webhook message from %s:\n%s", forge.value, job_hook)

    # Look up the project this job belongs to
    gitlab_conn = gitlab_cfg.get_connection()
    project = await asyncio.to_thread(gitlab_conn.projects.get, job_hook.project_id)
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

    # Extract the merge-request IID from the job
    match = MR_REGEX.search(pipeline.ref)
    if not match:
        LOG.error(
            "Pipeline source is merge_request_event but no merge request IID was provided."
        )
        return
    merge_request_iid = int(match.group(1))

    # Check if this is a resubmission of an existing, completed job.
    # If it is, we'll exit out here and not waste time retrieving the logs,
    # running a new analysis or trying to submit a new comment.
    mr_job_db = GitlabMergeRequestJobs.get_by_details(
        forge=forge,
        project_id=project.id,
        mr_iid=merge_request_iid,
        job_id=job_hook.build_id,
    )
    if mr_job_db:
        LOG.info("Resubmission of an existing build. Skipping.")
        return

    LOG.debug("Retrieving log artifacts")
    # Retrieve the build logs from the merge request artifacts and preprocess them
    try:
        log_url, preprocessed_log = await retrieve_and_preprocess_koji_logs(
            gitlab_cfg, job
        )
    except LogsTooLargeError:
        LOG.error("Could not retrieve logs. Too large.")
        raise

    # Submit log to Log Detective and await the results.
    log_text = preprocessed_log.read().decode(encoding="utf-8")
    metrics_id = await add_new_metrics(
        api_name=EndpointType.ANALYZE_GITLAB_JOB,
        url=log_url,
        http_session=gitlab_cfg.get_http_session(),
        compressed_log_content=RemoteLogCompressor.zip_text(log_text),
    )
    staged_response = await perform_staged_analysis(log_text=log_text)
    update_metrics(metrics_id, staged_response)
    preprocessed_log.close()

    # check if this project is on the opt-in list for posting comments.
    if not is_eligible_package(project.name):
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


def is_eligible_package(project_name: str):
    """Check whether the provided package name is eligible for posting
    comments to the merge request"""

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


class LogsTooLargeError(RuntimeError):
    """The log archive exceeds the configured maximum size"""


async def retrieve_and_preprocess_koji_logs(
    gitlab_cfg: GitLabInstanceConfig,
    job: gitlab.v4.objects.ProjectJob,
):  # pylint: disable=too-many-branches,too-many-locals
    """Download logs from the merge request artifacts

    This function will retrieve the build logs and do some minimal
    preprocessing to determine which log is relevant for analysis.

    returns: The URL pointing to the selected log file and an open, file-like
    object containing the log contents to be sent for processing by Log
    Detective. The calling function is responsible for closing this object."""

    # Make sure the file isn't too large to process.
    if not await check_artifacts_file_size(gitlab_cfg, job):
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
            with artifacts_zip.open(zipinfo.filename) as task_failed_log:
                contents = task_failed_log.read().decode("utf-8")
                match = FAILURE_LOG_REGEX.search(contents)
                if match:
                    failure_log_name = match.group(1)
                    failed_arches[architecture] = PurePath(
                        path.parent, failure_log_name
                    )
                else:
                    LOG.info(
                        "task_failed.log does not indicate which log contains the failure."
                    )
                    # The best thing we can do at this point is return the
                    # task_failed.log, since it will probably contain the most
                    # relevant information
                    failed_arches[architecture] = path

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
    else:
        # We have one or more architectures that we don't know about? Just
        # pick the first alphabetically. If the issue was a Koji error
        # rather than a build failure, this will fall back to ~toplevel as
        # the lowest-sorting possibility.
        failed_arch = sorted(list(failed_arches.keys()))[0]

    LOG.debug("Failed architecture: %s", failed_arch)

    log_path = failed_arches[failed_arch].as_posix()

    log_url = f"{gitlab_cfg.api_path}/projects/{job.project_id}/jobs/{job.id}/artifacts/{log_path}"  # pylint: disable=line-too-long
    LOG.debug("Returning contents of %s%s", gitlab_cfg.url, log_url)

    # Return the log as a file-like object with .read() function
    return log_url, artifacts_zip.open(log_path)


async def check_artifacts_file_size(
    gitlab_cfg: GitLabInstanceConfig,
    job: gitlab.v4.objects.ProjectJob,
):
    """Method to determine if the artifacts are too large to process"""
    # First, make sure that the artifacts are of a reasonable size. The
    # zipped artifact collection will be stored in memory below. The
    # python-gitlab library doesn't expose a way to check this value directly,
    # so we need to interact with directly with the headers.
    artifacts_path = (
        f"{gitlab_cfg.api_path}/projects/{job.project_id}/jobs/{job.id}/artifacts"
    )
    LOG.debug("checking artifact URL %s%s", gitlab_cfg.url, artifacts_path)
    try:
        head_response = await gitlab_cfg.get_http_session().head(
            artifacts_path,
            allow_redirects=True,
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
        artifacts_path,
        content_length,
        gitlab_cfg.max_artifact_size,
    )
    return content_length <= gitlab_cfg.max_artifact_size


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
    metrics = AnalyzeRequestMetrics.get_metric_by_id(metrics_id)
    Comments.create(
        forge,
        project.id,
        merge_request_iid,
        job.id,
        discussion.id,
        metrics,
    )


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
