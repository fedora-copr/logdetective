"""Runtime TF admission limits checked before persistence and retrieval."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import update

from logdetective import artifacts, task_execution
from logdetective.config import SERVER_CONFIG
from logdetective.database.base import transaction
from logdetective.database.models.tasks import TaskAnalysis
from logdetective.models import (
    Config, TestingFarmConfig as TFTConfig, TestingFarmSuiteRequest as TFTRequest,
)
from logdetective.remote_log import RemoteLog
from logdetective.server import app
from tests.test_helpers import DatabaseFactory


def suite(case_logs=1, shared_artifacts=0, cases=1):
    """Distinct scoped entries deliberately reuse one URL."""
    def logs(count):
        return [{"name": f"log-{i}", "url": "https://example.org/log"} for i in range(count)]

    return TFTRequest(
        suite={"plan": "/plan", "state": "error"}, shared_artifacts=logs(shared_artifacts),
        cases=[{"case_id": f"c{i}", "name": "/test", "result": "failed",
                "artifacts": logs(case_logs)} for i in range(cases)],
    )


@pytest.mark.parametrize("limit", [1, 150])
@pytest.mark.parametrize("scope", [
    "max_artifacts_per_case", "max_shared_artifacts", "max_total_artifacts",
])
def test_configured_boundaries(scope, limit):
    """Lower and higher caps apply to entries, even when URLs repeat."""
    config = TFTConfig(
        max_artifacts_per_case=200, max_shared_artifacts=200, max_total_artifacts=400,
    )
    setattr(config, scope, limit)

    def request(count):
        if scope == "max_shared_artifacts":
            return suite(0, count)
        if scope == "max_total_artifacts":
            return suite(1, count - 1)
        return suite(count)

    request(limit).validate_limits(config)
    with pytest.raises(ValueError, match=scope):
        request(limit + 1).validate_limits(config)


def test_total_can_be_lower_and_no_case_count_cap():
    """Scope allowances do not bypass the total; empty cases are not truncated."""
    with pytest.raises(ValueError, match="max_total_artifacts"):
        suite(2, 2, 2).validate_limits(TFTConfig(max_total_artifacts=5))
    request = suite(0, 1, 100)
    request.validate_limits(TFTConfig())
    assert len(request.cases) == 100
    assert Config().testing_farm == TFTConfig()
    assert TFTConfig().model_dump() == {
        "max_artifacts_per_case": 10, "max_shared_artifacts": 10,
        "max_total_artifacts": 120,
    }


@pytest.mark.parametrize("field", [
    "max_artifacts_per_case", "max_shared_artifacts", "max_total_artifacts",
])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_log_limits(field, value):
    """Log allowances must be positive integers."""
    with pytest.raises(ValidationError):
        TFTConfig(**{field: value})


@pytest.mark.asyncio
async def test_excess_artifact_count_rejected_before_enqueue(monkeypatch):
    """A new suite exceeding the configured count limit is not enqueued."""
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_total_artifacts", 1)
    monkeypatch.setattr(TaskAnalysis, "_existing", AsyncMock(return_value=None))
    enqueue = MagicMock()
    monkeypatch.setattr("logdetective.server.analyze_tft_suite", enqueue)
    payload = suite(1, 1).model_dump(mode="json")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/analyze/tft", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "testing_farm.max_total_artifacts exceeded"}
    enqueue.configure.assert_not_called()


@pytest.mark.asyncio
async def test_lowered_cap_preserves_uuid_retry_and_conflict(monkeypatch):
    """Existing UUIDs are resolved before a changed cap, without a second job."""
    public_id = uuid4()
    payload = suite(1, 1).model_dump(mode="json")
    payload["id"] = str(public_id)
    enqueue = AsyncMock(return_value=41)
    job = MagicMock()
    job.configure.return_value.defer_async = enqueue
    monkeypatch.setattr("logdetective.server.analyze_tft_suite", job)
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_total_artifacts", 2)
    async with DatabaseFactory().make_new_db(), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post("/analyze/tft", json=payload)
        assert first.status_code == 202
        monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_total_artifacts", 1)
        retry = await client.post("/analyze/tft", json=payload)
        assert retry.status_code == 202
        assert retry.json() == first.json()
        conflict = await client.post("/analyze/tft", json={
            **payload, "suite": {"plan": "/different", "state": "error"},
        })
        assert conflict.status_code == 409
        async with transaction(commit=True) as session:
            await session.execute(update(TaskAnalysis).where(
                TaskAnalysis.task_id == public_id
            ).values(expires_at=datetime.now(UTC)))
        expired = await client.post("/analyze/tft", json=payload)
        assert expired.status_code == 409
        enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieval_honors_admission_count_and_keeps_url_cache(monkeypatch):
    """Lowering the count cap after validation does not prevent retrieval."""
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_artifacts_per_case", 12)
    payload = suite(12)
    payload.validate_limits(SERVER_CONFIG.testing_farm)
    monkeypatch.setattr(SERVER_CONFIG.testing_farm, "max_artifacts_per_case", 10)
    download = AsyncMock(return_value="failure")
    monkeypatch.setattr(RemoteLog, "get_url_content", download)
    fetched = await artifacts.fetch_suite_artifacts(payload, 100)
    assert len(fetched.artifacts) == 12
    download.assert_awaited_once()
