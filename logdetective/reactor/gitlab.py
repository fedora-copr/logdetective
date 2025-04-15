import asyncio
import gitlab
import jinja2
import re

from typing import List

from logdetective.reactor.config import MR_REGEX, get_config
from logdetective.reactor.detective import submit_to_log_detective
from logdetective.reactor.koji import retrieve_and_preprocess_koji_logs
from logdetective.reactor.logging import get_log
from logdetective.reactor.models import JobHook
from logdetective.reactor.errors import LogsTooLargeError
from logdetective.server.models import StagedResponse
from logdetective.reactor.dependencies import HttpConnections

FAILURE_LOG_REGEX = re.compile(r"(\w*\.log)")

REACTOR_CONFIG = get_config()
LOG = get_log()


async def process_gitlab_job_event(
    job_hook: JobHook,
    jinja_env: jinja2.Environment,
    http_connections: HttpConnections,
    known_packages: List[str],
) -> None:
    """Handle a received job_event webhook from GitLab"""
    LOG.debug("Received webhook message:\n%s", job_hook)

    # Look up the project this job belongs to
    project = await asyncio.to_thread(
        http_connections.gitlab.projects.get, job_hook.project_id
    )

    # check if this project is on the opt-in list
    if project.name not in known_packages:
        LOG.info("Ignoring unrecognized package %s", project.name)
        return
    LOG.info("Processing RPM build logs for %s", project.name)

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
        log_url = await retrieve_and_preprocess_koji_logs(job, http_connections)
    except LogsTooLargeError:
        LOG.error("Could not retrieve logs. Too large.")
        raise

    # Submit log to Log Detective and await the results.
    staged_response = await submit_to_log_detective(http_connections, log_url)

    # Add the Log Detective response as a comment to the merge request
    await comment_on_mr(
        jinja_env, project, merge_request_iid, job, log_url, staged_response
    )


async def comment_on_mr(
    jinja_env: jinja2.Environment,
    project: gitlab.v4.objects.Project,
    merge_request_iid: int,
    job: gitlab.v4.objects.ProjectJob,
    log_url: str,
    response: StagedResponse,
):
    """Add the Log Detective response as a comment to the merge request"""
    LOG.debug(
        "Primary Explanation for %s MR %d: %s",
        project.name,
        merge_request_iid,
        response.explanation.text,
    )

    # Get the formatted short comment.
    short_comment = await generate_markdown_comment(
        jinja_env, job, log_url, response, full=False
    )

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
    full_comment = await generate_markdown_comment(
        jinja_env, job, log_url, response, full=True
    )
    note.body = full_comment

    # Pause for five seconds before sending the snippet data, otherwise
    # Gitlab may bundle the edited message together with the creation
    # message in email.
    await asyncio.sleep(5)
    await asyncio.to_thread(note.save)


async def generate_markdown_comment(
    jinja_env: jinja2.Environment,
    job: gitlab.v4.objects.ProjectJob,
    log_url: str,
    response: StagedResponse,
    full: bool = True,
) -> str:
    """Use a template to generate a comment string to submit to Gitlab"""

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
