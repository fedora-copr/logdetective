"""Testing Farm artifact field naming contract."""

import pytest
from pydantic import ValidationError

from logdetective.models import TestingFarmSuiteRequest as TFTSuiteRequest
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
