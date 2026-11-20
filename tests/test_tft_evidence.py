"""Shared evidence references and artifact identity across suite scopes."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from logdetective import task_execution
from logdetective.agents.testing_farm_analysis import _collect_suite_evidence
from logdetective.artifacts import fetch_suite_artifacts
from logdetective.models import (
    Snippet,
    TestingFarmAgentOutput as TFTAgentOutput,
    TestingFarmCaseResponse as TFTCaseResponse,
    TestingFarmSuiteRequest as TFTSuiteRequest,
    TestingFarmSuiteResult as TFTSuiteResult,
)
from logdetective.remote_log import RemoteLog


@pytest.mark.parametrize("scope", ["shared", "case"])
def test_duplicate_artifact_names_are_rejected_within_scope(scope):
    """Even identical URLs cannot make duplicate names unambiguous in one scope."""
    log = {"name": "test.log", "url": "https://example.org/log"}
    with pytest.raises(ValidationError, match="Duplicate artifact names in one scope"):
        TFTSuiteRequest(suite={"plan": "/plan"},
                        shared_artifacts=[log, log] if scope == "shared" else [],
                        cases=[{"case_id": "c1", "name": "/test", "result": "failed",
                                "artifacts": [log, log] if scope == "case" else []}])


@pytest.mark.asyncio
async def test_identical_artifact_names_and_content_keep_scope(monkeypatch):
    """One cached URL can populate shared and own logs without identity collisions."""
    download = AsyncMock(return_value="same evidence")
    monkeypatch.setattr(RemoteLog, "get_url_content", download)
    log = {"name": "test.log", "url": "https://example.org/log"}
    request = TFTSuiteRequest(suite={"plan": "/plan"}, shared_artifacts=[log], cases=[
        {"case_id": case_id, "name": "/test", "result": "failed", "artifacts": [log]}
        for case_id in ("c1", "c2")
    ])
    fetched = await fetch_suite_artifacts(request, 100)
    download.assert_awaited_once()
    assert set(fetched.artifacts) == {
        "shared/test.log", "case/c1/test.log", "case/c2/test.log",
    }
    evidence = _collect_suite_evidence({
        f"{source}:1": Snippet(text=content, source_file=source, line_number=1)
        for source, content in fetched.artifacts.items()
    }, [])
    result = task_execution._suite_result(request, fetched.eligible, {}, TFTAgentOutput(
        cases=[{"case_id": case_id, "explanation": "cause"} for case_id in ("c1", "c2")],
    ), evidence)
    assert list(result.shared_snippets) == ["shared/test.log:1"]
    for case_id, outcome in result.cases.items():
        assert [snippet.source_file for snippet in outcome.analysis.snippets] == [
            f"case/{case_id}/test.log",
        ]
        assert outcome.analysis.shared_snippet_refs == ["shared/test.log:1"]


@pytest.mark.parametrize("fallback", [False, True])
def test_suite_rejects_dangling_shared_references(fallback):
    """Case and fallback references must resolve in the serialized suite index."""
    analysis = TFTCaseResponse(explanation="cause", shared_snippet_refs=["shared/log:1"])
    with pytest.raises(ValidationError, match="Unknown shared snippet reference"):
        TFTSuiteResult(
            cases={} if fallback else {"c1": {"status": "done", "analysis": analysis}},
            suite_fallback=analysis if fallback else None,
        )
