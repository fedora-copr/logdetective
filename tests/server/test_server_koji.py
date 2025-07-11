import koji
import pytest

from logdetective.server.models import KojiInstanceConfig, KojiTask, StagedResponse
from logdetective.server.server import analyze_koji_task

from logdetective.server.exceptions import LogsTooLargeError
from logdetective.server.koji import (
    get_failed_subtask_info,
    get_failed_log_from_task,
)
from tests.server.test_helpers import DatabaseFactory, mock_chat_completions


arches = [
    "x86_64",
    "aarch64",
    "ppc64le",
    "riscv",
    "s390x",
]


@pytest.mark.parametrize("arch", arches)
@pytest.mark.asyncio
async def test_koji_get_failed_subtask_info(mocker, arch):
    # Mock the Koji session
    mock_session = mocker.Mock()

    # Mock the parent task info
    mock_session.getTaskInfo.return_value = {
        "state": koji.TASK_STATES["FAILED"],
        "method": "build",
    }

    # Mock the subtasks response
    mock_session.getTaskDescendents.return_value = {
        "133801268": [
            {
                "method": "buildArch",
                "state": koji.TASK_STATES["FAILED"],
                "arch": arch,
            },
            {
                "method": "buildArch",
                "state": koji.TASK_STATES["FAILED"],
                "arch": "x86_64",
            },
        ]
    }

    taskinfo = await get_failed_subtask_info(mock_session, 133801268)

    assert taskinfo is not None

    # Make sure this is a failed task
    assert taskinfo["state"] == koji.TASK_STATES["FAILED"]

    # We passed it a parent, make sure we got the child back
    assert taskinfo["method"] == "buildArch"

    # Several arches failed; make sure we received x86_64, since it's the
    # highest priority in the list.
    assert taskinfo["arch"] == "x86_64"


subtask_arches = [
    (133801422, "x86_64"),
    (133801421, "aarch64"),
    (133801420, "ppc64le"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("taskid,arch", subtask_arches)
async def test_koji_get_failed_subtask_info_arches(mocker, taskid, arch):
    # Mock the Koji session
    mock_session = mocker.Mock()

    mock_session.getTaskInfo.return_value = {
        "state": koji.TASK_STATES["FAILED"],
        "method": "buildArch",
        "arch": arch,
    }

    taskinfo = await get_failed_subtask_info(mock_session, taskid)

    assert taskinfo is not None

    # Make sure this is a failed task
    assert taskinfo["state"] == koji.TASK_STATES["FAILED"]

    # We passed it a parent, make sure we got the child back
    assert taskinfo["method"] == "buildArch"

    # Several arches failed; make sure we received x86_64
    assert taskinfo["arch"] == arch


@pytest.mark.asyncio
async def test_koji_get_failed_subtask_from_canceledtask(mocker):
    # Mock the Koji session
    mock_session = mocker.Mock()

    # Mock the parent task info
    mock_session.getTaskInfo.return_value = {
        "state": koji.TASK_STATES["CANCELED"],
        "method": "build",
    }

    # Mock the subtasks response
    mock_session.getTaskDescendents.return_value = {
        "133858238": [
            {
                "method": "buildArch",
                "state": koji.TASK_STATES["FAILED"],
                "arch": "i686",
            },
            {
                "method": "buildArch",
                "state": koji.TASK_STATES["FAILED"],
                "arch": "anunknownarch",
            },
            {
                "method": "buildArch",
                "state": koji.TASK_STATES["CANCELED"],
                "arch": "x86_64",
            },
        ]
    }

    taskinfo = await get_failed_subtask_info(mock_session, 133858238)

    assert taskinfo is not None

    # Make sure this is a failed task
    assert taskinfo["state"] == koji.TASK_STATES["FAILED"]

    # We passed it a parent, make sure we got the child back
    assert taskinfo["method"] == "buildArch"

    # Several arches failed; Of the remaining arches, we don't recognize
    # either of them, so we return the first one alphabetically.
    assert taskinfo["arch"] == "anunknownarch"


EXAMPLE_TASK_ID = 133858346


def create_mock_koji_session(mocker, task_id):
    mock_session = mocker.Mock()
    mock_session.getTaskInfo.return_value = {
        "id": task_id,
        "state": koji.TASK_STATES["FAILED"],
        "method": "buildArch",
        "arch": "x86_64",
    }

    mock_session.getTaskResult.return_value = {
        "faultString": "BuildError: error building package (arch x86_64), mock exited with status 1; see build.log or root.log for more information"  # pylint: disable=line-too-long
    }

    # Mock the build log response
    mock_session.listTaskOutput.return_value = {
        "build.log": {
            "st_size": "43",
        },
    }

    mock_session.downloadTaskOutput.return_value = (
        b"Error: Build failed\nDetailed error message"
    )
    return mock_session


@pytest.mark.asyncio
async def test_koji_get_failed_log_from_task(mocker):
    task_id = EXAMPLE_TASK_ID

    # Mock the Koji session
    mock_session = create_mock_koji_session(mocker, task_id)

    # Test getting log from a failed task
    log_file, log_contents = await get_failed_log_from_task(
        mock_session, task_id, max_size=1024 * 1024
    )

    # Verify the log content
    assert log_file == "build.log"
    assert log_contents == "Error: Build failed\nDetailed error message"

    # Verify the correct methods were called
    mock_session.getTaskInfo.assert_called_once_with(task_id)
    mock_session.getTaskResult.assert_called_once_with(task_id, raise_fault=False)
    mock_session.listTaskOutput.assert_called_once_with(task_id, stat=True)
    mock_session.downloadTaskOutput.assert_called_once_with(task_id, "build.log")

    # Test getting a log that is too large
    with pytest.raises(LogsTooLargeError):
        await get_failed_log_from_task(mock_session, task_id, max_size=10)


@pytest.mark.parametrize(
    "mock_chat_completions", ["Task analysis completed."], indirect=True
)
@pytest.mark.asyncio
async def test_koji_analyze_koji_task(mocker, mock_chat_completions):
    with DatabaseFactory().make_new_db() as _:
        # Mock the KojiInstanceConfig
        mock_koji_instance_config = mocker.Mock()
        mock_koji_conn = create_mock_koji_session(mocker, EXAMPLE_TASK_ID)
        mock_koji_instance_config.get_connection.return_value = mock_koji_conn
        mock_koji_instance_config.max_artifact_size = 1024 * 1024
        mock_koji_instance_config.name = "fedora"
        mock_koji_instance_config.xmlrpc_url = "https://koji.fedoraproject.org/kojihub"

        koji_task = KojiTask(koji_instance="fedora", task_id=EXAMPLE_TASK_ID)

        response = await analyze_koji_task(koji_task, mock_koji_instance_config)

        assert response is not None

        # Verify the response content
        assert isinstance(response, StagedResponse)
        assert response.explanation.text == "Task analysis completed."
