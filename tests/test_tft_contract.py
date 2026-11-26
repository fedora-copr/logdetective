"""Testing Farm request and agent output contracts."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from logdetective.agents.testing_farm_analysis import analyze_testing_farm_suite
from logdetective.models import (
    TestingFarmAgentOutput as TFTAgentOutput, TestingFarmSuiteRequest as TFTSuiteRequest,
)
from logdetective.server import app


def test_artifact_contract_schema():
    """OpenAPI exposes artifact terminology at every request scope."""
    schemas = app.openapi()["components"]["schemas"]
    for model, current, previous in (
        ("TestingFarmSuiteRequest", "shared_artifacts", "shared_logs"),
        ("TestingFarmCase", "artifacts", "logs"),
        ("TestingFarmCheck", "artifact_refs", "log_refs"),
    ):
        assert current in schemas[model]["properties"]
        assert previous not in schemas[model]["properties"]


@pytest.mark.parametrize("scope", ["suite", "case", "check", "subresult"])
def test_previous_log_fields_rejected(scope):
    """Old names cannot silently discard submitted evidence or references."""
    case = {"case_id": "c1", "name": "/test", "result": "failed"}
    payload = {"suite": {"plan": "/plan"}, "cases": [case]}
    if scope == "suite":
        payload["shared_logs"] = []
    elif scope == "case":
        case["logs"] = []
    else:
        case["checks" if scope == "check" else "subresults"] = [
            {"name": "check", "log_refs": []},
        ]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TFTSuiteRequest.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(("has_cases", "fallback", "error"), [
    (True, {"explanation": "suite cause"}, "unexpected suite fallback"),
    (False, None, "no suite fallback"),
    (False, {}, "no suite fallback"),
    (False, {"explanation": "  "}, "no suite fallback"),
])
async def test_suite_agent_rejects_invalid_fallback(has_cases, fallback, error):
    """A fallback is mandatory for empty suites and forbidden for individual cases."""
    request = TFTSuiteRequest(
        suite={"plan": "/plan", "state": "error"},
        cases=[{"case_id": "c1", "name": "/test", "result": "failed"}] if has_cases else [],
    )
    with (
        patch("logdetective.agents.testing_farm_analysis.RequirementAgent"),
        patch("logdetective.agents.testing_farm_analysis.run_agent", AsyncMock(
            return_value=TFTAgentOutput(suite_fallback=fallback),
        )),
        pytest.raises(ValueError, match=error),
    ):
        await analyze_testing_farm_suite(
            {}, MagicMock(), request, eligible_case_ids=set(),
            deadline=asyncio.get_running_loop().time() + 1200,
        )
