"""HTTP contract tests for asynchronous analysis operations."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from logdetective.compressors import LLMResponseCompressor
from logdetective.config import SERVER_CONFIG
from logdetective.database.models import Forge
from logdetective.database.models.exceptions import TaskConflictError
from logdetective.database.models.tasks import AnalysisState, TaskAnalysis, TaskType
from logdetective.models import APIResponse, GitLabInstanceConfig, JobHook
from logdetective.routes_gitlab import receive_gitlab_job_event_webhook
from logdetective.server import app, task_representation, validate_request_size


def scheduled_task(task_id=None) -> TaskAnalysis:
    """Build a detached application record suitable for route mocks."""
    return TaskAnalysis(
        id=1,
        task_id=task_id or uuid4(),
        owner_token_name=None,
        task_type=TaskType.GENERIC,
        request_hash="0" * 64,
        input_payload={"files": []},
        request_size=10,
        procrastinate_job_id=3,
        state=AnalysisState.SCHEDULED,
        generation=0,
        request_received_at=datetime.now(UTC),
    )


def test_task_representation_uses_task_enums():
    """The public model retains the domain types while JSON uses their values."""
    representation = task_representation(scheduled_task())

    assert representation.task_type is TaskType.GENERIC
    assert representation.status is AnalysisState.SCHEDULED


def test_task_representation_validates_koji_metadata():
    """Persisted JSON is validated with the task-specific metadata model."""
    task = scheduled_task()
    task.task_type = TaskType.KOJI
    task.state = AnalysisState.DONE
    task.response = LLMResponseCompressor(
        APIResponse(explanation="done")
    ).zip_response()
    task.task_metadata = {"task_id": 123, "log_file_name": "build.log"}

    representation = task_representation(task)

    assert representation.result is not None
    assert representation.result.task_id == 123
    assert representation.result.log_file_name == "build.log"


def test_task_representation_rejects_invalid_koji_metadata():
    """Malformed persisted Koji metadata cannot escape into the public API."""
    task = scheduled_task()
    task.task_type = TaskType.KOJI
    task.state = AnalysisState.DONE
    task.response = LLMResponseCompressor(
        APIResponse(explanation="done")
    ).zip_response()
    task.task_metadata = {"task_id": 123}

    with pytest.raises(RuntimeError, match="metadata is invalid"):
        task_representation(task)


@pytest.fixture
def size_override():
    async def fixed_request_size() -> int:
        return 10

    app.dependency_overrides[validate_request_size] = fixed_request_size
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_post_returns_accepted_location_and_stable_envelope(
    mocker, size_override
):
    public_id = uuid4()
    task = scheduled_task(public_id)
    admit = mocker.patch.object(
        TaskAnalysis, "admit", AsyncMock(return_value=(task, True))
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/analyze",
            json={
                "id": str(public_id),
                "files": [{"name": "build.log", "content": "error"}],
            },
        )

    assert response.status_code == 202
    assert response.headers["location"] == f"http://test/tasks/{public_id}"
    assert response.headers["retry-after"] == "5"
    assert response.json() == {
        "id": str(public_id),
        "taskType": "generic",
        "createdAt": task.request_received_at.isoformat().replace("+00:00", "Z"),
        "status": "scheduled",
        "error": None,
        "result": None,
    }
    assert admit.await_args.kwargs["task_id"] == public_id


@pytest.mark.asyncio
async def test_get_active_task_returns_200_with_retry_after(mocker):
    task = scheduled_task()
    mocker.patch.object(TaskAnalysis, "get_owned", AsyncMock(return_value=task))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/tasks/{task.task_id}")

    assert response.status_code == 200
    assert response.headers["retry-after"] == "5"
    assert response.json()["status"] == "scheduled"


@pytest.mark.asyncio
async def test_get_completed_task_returns_plain_text_analysis(mocker):
    task = scheduled_task()
    task.state = AnalysisState.DONE
    task.response = LLMResponseCompressor(
        APIResponse(explanation="Missing dependency", solution="Install libfoo")
    ).zip_response()
    mocker.patch.object(TaskAnalysis, "get_owned", AsyncMock(return_value=task))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/tasks/{task.task_id}")

    assert response.status_code == 200
    assert response.json()["result"]["explanation"] == "Missing dependency"
    assert response.json()["result"]["solution"] == "Install libfoo"


@pytest.mark.asyncio
async def test_delete_running_task_returns_202(mocker):
    task = scheduled_task()
    task.state = AnalysisState.CANCELLING
    mocker.patch.object(
        TaskAnalysis, "request_cancellation", AsyncMock(return_value=task)
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/tasks/{task.task_id}")

    assert response.status_code == 202
    assert response.headers["retry-after"] == "5"
    assert response.json()["status"] == "cancelling"


@pytest.mark.asyncio
async def test_old_koji_url_is_removed():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/analyze/rpmbuild/koji/fedora/123")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_gitlab_expired_source_conflict_returns_409(mocker):
    mocker.patch.dict(
        SERVER_CONFIG.gitlab.instances,
        {
            Forge.gitlab_com.value: GitLabInstanceConfig(name="gitlab.com"),
        },
    )
    mocker.patch.object(
        TaskAnalysis,
        "admit",
        AsyncMock(side_effect=TaskConflictError("Task has expired")),
    )

    request = Request({"type": "http"})
    request.state.api_token_name = None
    response = await receive_gitlab_job_event_webhook(
        job_hook=JobHook(
            object_kind="build",
            build_id=123,
            pipeline_id=456,
            build_name="build_centos_stream_rpm",
            build_status="failed",
            project_id=678,
        ),
        request=request,
        x_gitlab_instance=Forge.gitlab_com.value,
    )

    assert response.status_code == 409
