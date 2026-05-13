from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from beeai_framework.tools.think import ThinkTool
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend.errors import ChatModelError
from litellm.exceptions import Timeout, RateLimitError

from logdetective.server.agent.agent import analyze_artifacts
from logdetective.server.agent.tools import DrainExtractorTool, CSGrepExtractorTool
from logdetective.server.config import SERVER_CONFIG, get_openai_chat_model
from logdetective.server.exceptions import (
    LogDetectiveInferenceError,
    LogDetectiveInferenceTimeout,
    LogDetectiveInferenceRateLimit,
)
from logdetective.server.models import AgentResponse, Explanation


@pytest.fixture
def mock_agent_setup():
    """Fixture to setup common agent mocks for initialization tests."""
    mock_artifacts = {"build.log": "Error: compilation failed"}
    mock_chat_model = MagicMock(spec=OpenAIChatModel)

    mock_agent_output = MagicMock()
    mock_agent_output.state.answer.text = "Mocked analysis result"
    mock_agent_output.output_structured = AgentResponse(
        explanation=Explanation(text="Mock explanation"))
    return mock_artifacts, mock_chat_model, mock_agent_output


@pytest.mark.asyncio
async def test_analyze_artifacts_init_default(mock_agent_setup):
    """Test default initialization (CSGrep disabled)."""
    mock_artifacts, mock_chat_model, mock_agent_output = mock_agent_setup

    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        mock_run_instance = MagicMock()
        mock_run_instance.middleware = AsyncMock(return_value=mock_agent_output)
        MockAgent.return_value.run.return_value = mock_run_instance

        with patch.object(SERVER_CONFIG.extractor, "csgrep", False):
            await analyze_artifacts(mock_artifacts, mock_chat_model)

            _, kwargs = MockAgent.call_args
            tools = kwargs.get("tools", [])

            assert any(isinstance(t, ThinkTool) for t in tools)
            assert any(isinstance(t, DrainExtractorTool) for t in tools)
            assert not any(isinstance(t, CSGrepExtractorTool) for t in tools)


@pytest.mark.asyncio
async def test_analyze_artifacts_init_with_csgrep(mock_agent_setup):
    """Test initialization when CSGrep is enabled."""
    mock_artifacts, mock_chat_model, mock_agent_output = mock_agent_setup

    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        mock_run_instance = MagicMock()
        mock_run_instance.middleware = AsyncMock(return_value=mock_agent_output)
        MockAgent.return_value.run.return_value = mock_run_instance

        with patch.object(SERVER_CONFIG.extractor, "csgrep", True):
            await analyze_artifacts(mock_artifacts, mock_chat_model)

            _, kwargs = MockAgent.call_args
            tools = kwargs.get("tools", [])

            assert any(isinstance(t, ThinkTool) for t in tools)
            assert any(isinstance(t, DrainExtractorTool) for t in tools)
            assert any(isinstance(t, CSGrepExtractorTool) for t in tools)


@pytest.mark.asyncio
async def test_analyze_artifacts_execution_flow(mock_agent_setup):
    """Test the execution flow and response mapping of analyze_artifacts."""
    mock_artifacts, mock_chat_model, mock_agent_output = mock_agent_setup
    mock_artifacts = {"artifact_1.log": "content 1", "artifact_2.log": "content 2"}

    expected_answer = "The build failed because of a missing dependency."
    mock_agent_output.output_structured.explanation.text = expected_answer

    mock_run_chain = MagicMock()
    mock_run_chain.middleware = AsyncMock(return_value=mock_agent_output)

    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value = mock_run_chain

        response = await analyze_artifacts(mock_artifacts, mock_chat_model)

        assert response.explanation.text == expected_answer

        run_call_args = mock_agent_instance.run.call_args[0][0]
        assert "artifact_1.log" in run_call_args
        assert "artifact_2.log" in run_call_args


@pytest.mark.asyncio
@pytest.mark.parametrize("cause, expected_exc", [
    (Timeout("timed out", "model-mock", "provider-mock"), LogDetectiveInferenceTimeout),
    (RateLimitError("rate limited", "model-mock", "provider-mock"), LogDetectiveInferenceRateLimit),
    (ChatModelError(), LogDetectiveInferenceError),
])
async def test_analyze_artifacts_inference_errors(mock_agent_setup, cause, expected_exc):
    mock_artifacts, mock_chat_model, _ = mock_agent_setup
    mock_error = ChatModelError("model error")
    mock_error.__cause__ = cause

    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        mock_run_instance = MagicMock()
        mock_run_instance.middleware = AsyncMock(
            side_effect=mock_error
        )
        MockAgent.return_value.run.return_value = mock_run_instance

        with pytest.raises(LogDetectiveInferenceError) as exc_info:
            await analyze_artifacts(mock_artifacts, mock_chat_model)

    assert isinstance(exc_info.value, expected_exc)
