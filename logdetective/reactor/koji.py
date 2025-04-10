import aiohttp
import asyncio
import re
import zipfile


from pathlib import PurePath
from fastapi import FastAPI
from tempfile import TemporaryFile

from gitlab.v4.objects import ProjectJob

from logdetective.reactor.config import get_config
from logdetective.reactor.logging import get_log
from logdetective.reactor.errors import LogsTooLargeError

FAILURE_LOG_REGEX = re.compile(r"(\w*\.log)")

LOG = get_log()
REACTOR_CONFIG = get_config()


async def retrieve_and_preprocess_koji_logs(
    job: ProjectJob,
    app: FastAPI,
):
    """Download logs from the merge request artifacts

    This function will retrieve the build logs and do some minimal
    preprocessing to determine which log is relevant for analysis.

    returns: The URL pointing to the selected log file"""

    # Make sure the file isn't too large to process.
    if not await check_artifacts_file_size(job, app):
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

    log_path = failed_arches[failed_arch].as_posix()

    log_url = f"{REACTOR_CONFIG.gitlab.url}/{REACTOR_CONFIG.gitlab.api_root_path}/projects/{job.project_id}/jobs/{job.id}/artifacts/{log_path}"  # pylint: disable=line-too-long
    LOG.debug("Returning %s", log_url)

    # Return the URL to the identified log
    return log_url


async def check_artifacts_file_size(job: ProjectJob, app: FastAPI):
    """Method to determine if the artifacts are too large to process"""
    # First, make sure that the artifacts are of a reasonable size. The
    # zipped artifact collection will be stored in memory below. The
    # python-gitlab library doesn't expose a way to check this value directly,
    # so we need to interact with directly with the headers.
    artifacts_path = f"{REACTOR_CONFIG.gitlab.api_root_path}/projects/{job.project_id}/jobs/{job.id}/artifacts"  # pylint: disable=line-too-long
    try:
        async with app.gitlab_http.head(
            artifacts_path,
            allow_redirects=True,
        ) as header_resp:
            header_resp.raise_for_status()

            content_length = int(header_resp.headers.get("content-length"))
            LOG.debug(
                "URI: %s, content-length: %d, max length: %d",
                artifacts_path,
                content_length,
                REACTOR_CONFIG.gitlab.max_artifact_size,
            )
            return content_length <= REACTOR_CONFIG.gitlab.max_artifact_size
    except aiohttp.client_exceptions.ClientError as e:
        LOG.exception(e)
        raise
