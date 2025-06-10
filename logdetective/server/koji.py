import asyncio
from typing import Any, Callable

import backoff
import koji
from logdetective.server.config import LOG


class LogDetectiveKojiException(Exception):
    """Base exception for Koji-related errors."""


class UnknownTaskType(LogDetectiveKojiException):
    """The task type is not supported."""


class NoFailedTask(LogDetectiveKojiException):
    """The task is not in the FAILED state."""


class LogDetectiveConnectionError(LogDetectiveKojiException):
    """A connection error occurred."""


def connection_error_giveup(details: backoff._typing.Details) -> None:
    """
    Too many connection errors, give up.
    """
    LOG.error("Too many connection errors, giving up. %s", details["exception"])
    raise LogDetectiveConnectionError() from details["exception"]


@backoff.on_exception(
    backoff.expo,
    koji.GenericError,
    max_time=60,
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

    # If the parent isn't FAILED, the children probably aren't either.
    # There's one special case where the user may have canceled the
    # overall task when one arch failed, so we should check that situation
    # too.
    if (
        taskinfo["state"] != koji.TASK_STATES["FAILED"]
        and taskinfo["state"] != koji.TASK_STATES["CANCELED"]  # noqa: W503 flake vs lint
    ):
        raise NoFailedTask(f"The primary task state was {taskinfo['state']}.")

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
