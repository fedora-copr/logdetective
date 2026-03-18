from unittest.mock import AsyncMock, MagicMock
import koji
import pytest
from beeai_framework.adapters.openai import OpenAIChatModel

from logdetective.utils import mib_to_bytes
from logdetective.server.models import Response, Explanation
from logdetective.server.server import analyze_koji_task, KojiCallbackManager
from logdetective.server.exceptions import LogsTooLargeError, LogsMissingError
from logdetective.server.koji import (
    get_failed_subtask_info,
    get_failed_log_from_task,
)
from tests.server.test_helpers import (
    DatabaseFactory,
    create_mock_koji_session,
    ARCHES,
    SIMPLE_METHODS,
    SUBTASK_ARCHES,
    EXAMPLE_TASK_ID,
)


@pytest.mark.parametrize("arch", ARCHES)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method", SIMPLE_METHODS)
@pytest.mark.parametrize("taskid,arch", SUBTASK_ARCHES)
async def test_koji_get_failed_subtask_info_arches(mocker, taskid, arch, method):
    """Test retrieval of substasks for tasks of type `buildArch` or `buildSRPMFromSCM`.
    These should be returned directly, without going trough architectures."""
    # Mock the Koji session
    mock_session = mocker.Mock()

    mock_session.getTaskInfo.return_value = {
        "state": koji.TASK_STATES["FAILED"],
        "method": method,
        "arch": arch,
    }

    taskinfo = await get_failed_subtask_info(mock_session, taskid)

    assert taskinfo is not None

    # Make sure this is a failed task
    assert taskinfo["state"] == koji.TASK_STATES["FAILED"]

    # We passed it a parent, make sure we got the child back
    assert taskinfo["method"] == method

    # Several arches failed; make sure we received x86_64
    assert taskinfo["arch"] == arch


@pytest.mark.parametrize("method", SIMPLE_METHODS)
@pytest.mark.asyncio
async def test_koji_get_failed_subtask_from_canceledtask(mocker, method):
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
                "method": method,
                "state": koji.TASK_STATES["FAILED"],
                "arch": "i686",
            },
            {
                "method": method,
                "state": koji.TASK_STATES["FAILED"],
                "arch": "anunknownarch",
            },
            {
                "method": method,
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
    assert taskinfo["method"] == method

    # Several arches failed; Of the remaining arches, we don't recognize
    # either of them, so we return the first one alphabetically.
    assert taskinfo["arch"] == "anunknownarch"


@pytest.mark.parametrize("method", SIMPLE_METHODS)
@pytest.mark.asyncio
async def test_koji_get_failed_log_from_task(mocker, method):
    task_id = EXAMPLE_TASK_ID

    # Mock the Koji session
    mock_session = create_mock_koji_session(mocker, task_id, method)

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


@pytest.mark.parametrize("method", SIMPLE_METHODS)
@pytest.mark.asyncio
async def test_koji_get_failed_log_from_task_logs_too_large(mocker, method):
    """Test that attempt to download log larger than a limit raises an exception."""
    task_id = EXAMPLE_TASK_ID

    # Mock the Koji session
    mock_session = create_mock_koji_session(mocker, task_id, method)

    # Test getting a log that is too large
    with pytest.raises(LogsTooLargeError):
        await get_failed_log_from_task(mock_session, task_id, max_size=10)

    # Verify the correct methods were called
    mock_session.getTaskInfo.assert_called_once_with(task_id)
    mock_session.getTaskResult.assert_called_once_with(task_id, raise_fault=False)
    mock_session.listTaskOutput.assert_called_once_with(task_id, stat=True)
    mock_session.downloadTaskOutput.assert_not_called()


@pytest.mark.parametrize("method", SIMPLE_METHODS)
@pytest.mark.asyncio
async def test_koji_get_failed_log_from_task_log_missing(mocker, method):
    """Test that attempt to download log larger than a limit raises an exception."""
    task_id = EXAMPLE_TASK_ID

    # Mock the Koji session
    mock_session = create_mock_koji_session(
        mocker, task_id, method, list_task_output=False
    )

    # Test getting a log that is too large
    with pytest.raises(LogsMissingError):
        await get_failed_log_from_task(mock_session, task_id, max_size=10)

    # Verify the correct methods were called
    mock_session.getTaskInfo.assert_called_once_with(task_id)
    mock_session.getTaskResult.assert_called_once_with(task_id, raise_fault=False)
    mock_session.listTaskOutput.assert_called_once_with(task_id, stat=True)
    mock_session.downloadTaskOutput.assert_not_called()


@pytest.fixture
def mock_analysis(mocker, request):
    """Fixture to mock analyze_artifacts directly at the server level."""
    message = getattr(request, "param", "This is a mock message")
    mock_response = Response(
        explanation=Explanation(text=message),
        response_certainty=0.0,
        snippets=None
    )
    return mocker.patch(
        "logdetective.server.server.analyze_artifacts",
        AsyncMock(return_value=mock_response)
    )


@pytest.mark.parametrize(
    "mock_analysis", ["This is a mock message"], indirect=True
)
@pytest.mark.parametrize("method", SIMPLE_METHODS)
@pytest.mark.asyncio
async def test_koji_analyze_koji_task(mocker, method, mock_analysis):
    async with DatabaseFactory().make_new_db() as _:
        # Mock the KojiInstanceConfig
        mock_koji_instance_config = mocker.Mock()
        mock_koji_conn = create_mock_koji_session(mocker, EXAMPLE_TASK_ID, method)
        mock_koji_instance_config.get_connection.return_value = mock_koji_conn
        mock_koji_instance_config.max_artifact_size = mib_to_bytes(1)
        mock_koji_instance_config.name = "fedora"
        mock_koji_instance_config.xmlrpc_url = "https://koji.fedoraproject.org/kojihub"
        mock_koji_instance_config.get_callbacks.return_value = set()

        mock_chat_model = MagicMock(spec=OpenAIChatModel)

        response = await analyze_koji_task(
            task_id=EXAMPLE_TASK_ID,
            koji_instance_config=mock_koji_instance_config,
            koji_connection=mock_koji_conn,
            koji_callback_manager=KojiCallbackManager(),
            openai_chat_model=mock_chat_model,
        )

        assert response is not None

        # Verify the response content
        assert isinstance(response, Response)
        assert response.explanation.text == "This is a mock message"
