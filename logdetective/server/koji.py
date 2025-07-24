import asyncio
import re
from typing import Any, Callable, Optional

import backoff
import koji
from logdetective.server.exceptions import (
    KojiInvalidTaskID,
    LogDetectiveConnectionError,
    LogsMissingError,
    LogsTooLargeError,
    UnknownTaskType,
)
from logdetective.server.utils import connection_error_giveup

FAILURE_LOG_REGEX = re.compile(r"(\w*\.log)")


@backoff.on_exception(
    backoff.expo,
    koji.GenericError,
    max_time=60,
    on_giveup=connection_error_giveup,
)
async def call_koji(func: Callable, *args, **kwargs) -> Any:
    """
    Call a Koji function asynchronously.

    Automatically retries on connection errors.
    """
    try:
        result = await asyncio.to_thread(func, *args, **kwargs)
    except koji.ActionNotAllowed as e:
        # User doesn't have permission to do this, don't retry.
        raise LogDetectiveConnectionError(e) from e
    return result


async def get_failed_subtask_info(
    koji_session: koji.ClientSession, task_id: int
) -> dict[str, Any]:
    """
    If the provided task ID represents a task of type "build", this function
    will return the buildArch or rebuildSRPM subtask that failed. If there is
    more than one, it will return the first one found from the following
    ordered list of processor architectures. If none is found among those
    architectures, it will return the first failed architecture after a
    standard sort.
    * x86_64
    * aarch64
    * riscv
    * ppc64le
    * s390x

    If the provided task ID represents a task of type "buildArch" or
    "buildSRPMFromSCM" and has a task state of "FAILED", it will be returned
    directly.

    Any other task type will rase the UnknownTaskType exception.

    If no task or subtask of the provided task is in the task state "FAILED",
    this function will raise a NoFailedSubtask exception.
    """

    # Look up the current task first and check its type.
    taskinfo = await call_koji(koji_session.getTaskInfo, task_id)
    if not taskinfo:
        raise KojiInvalidTaskID(f"Task {task_id} does not exist.")

    # If the parent isn't FAILED, the children probably aren't either.
    # There's one special case where the user may have canceled the
    # overall task when one arch failed, so we should check that situation
    # too.
    if (
        taskinfo["state"] != koji.TASK_STATES["FAILED"]
        and taskinfo["state"] != koji.TASK_STATES["CANCELED"]  # noqa: W503 flake vs lint
    ):
        raise UnknownTaskType(f"The primary task state was {taskinfo['state']}.")

    # If the task is buildArch or buildSRPMFromSCM, we can return it directly.
    if taskinfo["method"] in ["buildArch", "buildSRPMFromSCM"]:
        return taskinfo

    # Look up the subtasks for the task.
    response = await asyncio.to_thread(koji_session.getTaskDescendents, task_id)
    subtasks = response[f"{task_id}"]
    arch_tasks = {}
    for subtask in subtasks:
        if (
            subtask["method"] not in ["buildArch", "buildSRPMFromSCM"]
            or subtask["state"] != koji.TASK_STATES["FAILED"]  # noqa: W503 flake vs lint
        ):
            # Skip over any completed subtasks or non-build types
            continue

        arch_tasks[subtask["arch"]] = subtask

    # Return the first architecture in the order of preference.
    for arch in ["x86_64", "aarch64", "riscv", "ppc64le", "s390x"]:
        if arch in arch_tasks:
            return arch_tasks[arch]

    # If none of those architectures were found, return the first one
    # alphabetically
    return arch_tasks[sorted(arch_tasks.keys())[0]]


async def get_failed_log_from_task(
    koji_session: koji.ClientSession, task_id: int, max_size: int
) -> Optional[tuple[str, str]]:
    """
    Get the failed log from a task.

    If the log is too large, this function will raise a LogsTooLargeError.
    If the log is missing or garbage-collected, this function will raise a
    LogsMissingError.
    """
    taskinfo = await get_failed_subtask_info(koji_session, task_id)

    # Read the failure reason from the task. Note that the taskinfo returned
    # above may not be the same as passed in, so we need to use taskinfo["id"]
    # to look up the correct failure reason.
    result = await call_koji(
        koji_session.getTaskResult, taskinfo["id"], raise_fault=False
    )

    # Examine the result message for the appropriate log file.
    match = FAILURE_LOG_REGEX.search(result["faultString"])
    if match:
        failure_log_name = match.group(1)
    else:
        # The best thing we can do at this point is return the
        # task_failed.log, since it will probably contain the most
        # relevant information
        return result["faultString"]

    # Check that the size of the log file is not enormous
    task_output = await call_koji(
        koji_session.listTaskOutput, taskinfo["id"], stat=True
    )
    if not task_output:
        # If the task has been garbage-collected, the task output will be empty
        raise LogsMissingError(
            "No logs attached to this task. Possibly garbage-collected."
        )

    if failure_log_name not in task_output:
        # This shouldn't be possible, but we'll check anyway.
        raise LogsMissingError(f"{failure_log_name} could not be located")

    if int(task_output[failure_log_name]["st_size"]) > max_size:
        raise LogsTooLargeError(
            f"{task_output[failure_log_name]['st_size']} exceeds max size {max_size}"
        )

    log_contents = await call_koji(
        koji_session.downloadTaskOutput, taskinfo["id"], failure_log_name
    )
    return failure_log_name, log_contents.decode("utf-8")
