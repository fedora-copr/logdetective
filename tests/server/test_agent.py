import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from beeai_framework.tools.think import ThinkTool
from beeai_framework.adapters.openai import OpenAIChatModel

from logdetective.server.agent.agent import analyze_artifacts
from logdetective.server.agent.tools import DrainExtractorTool, CSGrepExtractorTool
from logdetective.server.config import SERVER_CONFIG


@pytest.fixture
def mock_agent_setup():
    """Fixture to setup common agent mocks for initialization tests."""
    mock_artifacts = {"build.log": "Error: compilation failed"}
    mock_chat_model = MagicMock(spec=OpenAIChatModel)

    mock_agent_output = MagicMock()
    mock_agent_output.state.answer.text = "Mocked analysis result"

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
async def test_analyze_artifacts_execution_flow():
    """Test the execution flow and response mapping of analyze_artifacts."""
    mock_artifacts = {"artifact_1.log": "content 1", "artifact_2.log": "content 2"}
    mock_chat_model = MagicMock(spec=OpenAIChatModel)

    expected_answer = "The build failed because of a missing dependency."
    mock_agent_output = MagicMock()
    mock_agent_output.state.answer.text = expected_answer

    mock_run_chain = MagicMock()
    mock_run_chain.middleware = AsyncMock(return_value=mock_agent_output)

    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run.return_value = mock_run_chain

        response = await analyze_artifacts(mock_artifacts, mock_chat_model)

        assert response.explanation.text == expected_answer
        assert response.response_certainty == 0.0

        run_call_args = mock_agent_instance.run.call_args[0][0]
        assert "artifact_1.log" in run_call_args
        assert "artifact_2.log" in run_call_args
