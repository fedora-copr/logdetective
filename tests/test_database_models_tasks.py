from uuid import uuid4

import pytest
from sqlalchemy import select

from tests.test_helpers import DatabaseFactory

from logdetective import config
from logdetective.database.models.tasks import (
    TaskAnalysis,
    TaskType,
    AnalysisState,
)
from logdetective.database.models.metrics import (
    AnalyzeRequestMetrics,
    EndpointType,
)
from logdetective.database.models.exceptions import (
    AnalysisTaskNotFoundError,
    TaskNotAnalyzedError,
    TaskAnalysisTimeoutError,
)
from logdetective.models import APIResponse, Explanation
from logdetective.compressors import LLMResponseCompressor


@pytest.mark.asyncio
async def test_task_analysis_create():
    """Test creating new analysis task"""
    async with DatabaseFactory().make_new_db() as session_factory:
        task_id = await TaskAnalysis.create(
            task_type=TaskType.GENERIC,
            external_task_id=None,
            metadata=None,
        )

        assert task_id is not None

        # Verify task was created
        query = select(TaskAnalysis).filter(TaskAnalysis.task_id == task_id)
        async with session_factory() as session:
            result = await session.execute(query)
            task = result.scalars().first()

        assert task is not None
        assert task.task_type == TaskType.GENERIC
        assert task.state == AnalysisState.SCHEDULED
        assert task.attempt_count == 0
        assert task.external_task_id is None
        assert task.task_metadata is None


@pytest.mark.asyncio
async def test_task_analysis_create_with_metadata():
    """Test creating task with external_task_id and metadata"""
    async with DatabaseFactory().make_new_db() as session_factory:
        metadata = {
            "koji_instance": "https://koji.fedoraproject.org",
            "log_file": "build.log",
        }
        external_id = "12345"

        task_id = await TaskAnalysis.create(
            task_type=TaskType.KOJI,
            external_task_id=external_id,
            metadata=metadata,
        )

        query = select(TaskAnalysis).filter(TaskAnalysis.task_id == task_id)
        async with session_factory() as session:
            result = await session.execute(query)
            task = result.scalars().first()
        assert task
        assert task.task_type == TaskType.KOJI
        assert task.external_task_id == external_id
        assert task.task_metadata == metadata


@pytest.mark.asyncio
async def test_task_analysis_create_or_restart_new():
    """Test create_or_restart creates new task when none exists"""
    async with DatabaseFactory().make_new_db() as session_factory:
        external_id = "67890"
        metadata = {"instance": "test"}

        task_id = await TaskAnalysis.create_or_restart(
            task_type=TaskType.KOJI,
            external_task_id=external_id,
            metadata=metadata,
        )

        assert task_id is not None

        query = select(TaskAnalysis).filter(
            TaskAnalysis.external_task_id == external_id
        )
        async with session_factory() as session:
            result = await session.execute(query)
            task = result.scalars().first()
        assert task
        assert task.task_id == task_id
        assert task.attempt_count == 0


@pytest.mark.asyncio
async def test_task_analysis_create_or_restart_existing():
    """Test create_or_restart restarts existing task"""
    async with DatabaseFactory().make_new_db() as session_factory:
        external_id = "11111"

        # Create initial task
        task_id_1 = await TaskAnalysis.create_or_restart(
            task_type=TaskType.KOJI,
            external_task_id=external_id,
            metadata={"run": 1},
        )

        # Restart same task
        task_id_2 = await TaskAnalysis.create_or_restart(
            task_type=TaskType.KOJI,
            external_task_id=external_id,
            metadata={"run": 2},
        )

        # Should return same task_id
        assert task_id_1 == task_id_2

        query = select(TaskAnalysis).filter(
            TaskAnalysis.external_task_id == external_id
        )
        async with session_factory() as session:
            result = await session.execute(query)
            task = result.scalars().first()
        assert task
        # Attempt count should increment
        assert task.attempt_count == 1


@pytest.mark.asyncio
async def test_task_analysis_add_response():
    """Test adding response to task"""
    async with DatabaseFactory().make_new_db() as session_factory:
        # Create task
        task_id = await TaskAnalysis.create(
            task_type=TaskType.GENERIC,
        )

        # Create metrics
        metrics_id = await AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE,
        )

        # Create response
        response = APIResponse(
            explanation=Explanation(text="Test explanation"),
        )

        # Add response to task
        await TaskAnalysis.add_response(
            task_id=task_id,
            metric_id=metrics_id,
            response=response,
        )

        # Verify task has response
        query = select(TaskAnalysis).filter(TaskAnalysis.task_id == task_id)
        async with session_factory() as session:
            result = await session.execute(query)
            task = result.scalars().first()
        assert task
        assert task.response is not None
        assert task.state == AnalysisState.DONE
        assert task.response_metrics_id == metrics_id


@pytest.mark.asyncio
async def test_task_analysis_get_task_by_id():
    """Test retrieving task by UUID"""
    async with DatabaseFactory().make_new_db():
        # Create and complete task
        task_id = await TaskAnalysis.create(task_type=TaskType.GENERIC)
        metrics_id = await AnalyzeRequestMetrics.create(endpoint=EndpointType.ANALYZE)
        response = APIResponse(explanation=Explanation(text="Done"))
        await TaskAnalysis.add_response(task_id, metrics_id, response)

        # Retrieve task
        task = await TaskAnalysis.get_task_by_id(task_id)

        assert task.task_id == task_id
        assert task.response is not None
        assert task.response_returned_at is not None


@pytest.mark.asyncio
async def test_task_analysis_get_task_by_external_id():
    """Test retrieving task by external_task_id"""
    async with DatabaseFactory().make_new_db():
        external_id = "external-123"

        # Create and complete task
        task_id = await TaskAnalysis.create(
            task_type=TaskType.KOJI,
            external_task_id=external_id,
        )
        metrics_id = await AnalyzeRequestMetrics.create(
            endpoint=EndpointType.ANALYZE_KOJI_TASK
        )
        response = APIResponse(explanation=Explanation(text="Koji done"))
        await TaskAnalysis.add_response(task_id, metrics_id, response)

        # Retrieve by external ID
        task = await TaskAnalysis.get_task_by_external_id(external_id)

        assert task.task_id == task_id
        assert task.external_task_id == external_id
        assert task.response is not None


@pytest.mark.asyncio
async def test_task_analysis_get_task_not_found():
    """Test exception when task doesn't exist"""
    async with DatabaseFactory().make_new_db():
        with pytest.raises(AnalysisTaskNotFoundError):
            await TaskAnalysis.get_task_by_id(uuid4())


@pytest.mark.asyncio
async def test_task_analysis_get_task_not_analyzed():
    """Test exception when task not yet analyzed"""
    async with DatabaseFactory().make_new_db():
        # Create task without response
        task_id = await TaskAnalysis.create(task_type=TaskType.GENERIC)

        with pytest.raises(TaskNotAnalyzedError):
            await TaskAnalysis.get_task_by_id(task_id)


@pytest.mark.asyncio
async def test_task_analysis_timeout(monkeypatch):
    """Test timeout exception for old tasks"""
    # Patch timeout to 0 seconds

    original_timeout = config.SERVER_CONFIG.general.analysis_timeout
    config.SERVER_CONFIG.general.analysis_timeout = 0

    try:
        async with DatabaseFactory().make_new_db():
            # Create task in the past
            task_id = await TaskAnalysis.create(task_type=TaskType.GENERIC)

            # Should timeout immediately
            with pytest.raises(TaskAnalysisTimeoutError):
                await TaskAnalysis.get_task_by_id(task_id)
    finally:
        config.SERVER_CONFIG.general.analysis_timeout = original_timeout


@pytest.mark.asyncio
async def test_task_analysis_response_decompression():
    """Test response can be compressed and decompressed"""
    async with DatabaseFactory().make_new_db():
        task_id = await TaskAnalysis.create(task_type=TaskType.GENERIC)
        metrics_id = await AnalyzeRequestMetrics.create(endpoint=EndpointType.ANALYZE)

        original_response = APIResponse(
            explanation=Explanation(
                text="Complex explanation with special chars: ñ, é, 中文"
            ),
        )

        await TaskAnalysis.add_response(task_id, metrics_id, original_response)

        task = await TaskAnalysis.get_task_by_id(task_id)
        assert task.response
        decompressed = LLMResponseCompressor.unzip(task.response)

        assert decompressed.explanation.text == original_response.explanation.text
