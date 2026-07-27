from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from beeai_framework.tools.think import ThinkTool
from beeai_framework.backend import ChatModel
from beeai_framework.backend.errors import ChatModelError
from litellm.exceptions import Timeout, RateLimitError

from logdetective.server.agent.agent import analyze_artifacts
from logdetective.server.agent.tools import (
    AnnotatedSnippetLookupTool,
    DrainExtractorTool,
    CSGrepExtractorTool,
)
from logdetective.server.config import SERVER_CONFIG, load_embedding_model
from logdetective.server.database.models.annotated_builds import AnnotatedSnippets
from logdetective.server.exceptions import (
    LogDetectiveInferenceError,
    LogDetectiveInferenceTimeout,
    LogDetectiveInferenceRateLimit,
)
from logdetective.server.models import AgentResponse, Explanation, Solution


@pytest.fixture
def mock_agent_setup():
    """Fixture to setup common agent mocks for initialization tests."""
    mock_artifacts = {"build.log": "Error: compilation failed"}
    mock_chat_model = MagicMock(spec=ChatModel)

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
        with patch("asyncio.sleep", new_callable=AsyncMock):
            mock_run_instance = MagicMock()
            mock_run_instance.middleware = AsyncMock(
                side_effect=mock_error
            )
            MockAgent.return_value.run.return_value = mock_run_instance

            with pytest.raises(LogDetectiveInferenceError) as exc_info:
                await analyze_artifacts(mock_artifacts, mock_chat_model)

    assert isinstance(exc_info.value, expected_exc)


@pytest.mark.asyncio
async def test_analyze_artifacts_solution_stripped_when_disabled(mock_agent_setup):
    """Test that solution is None when generate_solution is False."""
    mock_artifacts, mock_chat_model, mock_agent_output = mock_agent_setup
    mock_agent_output.output_structured = AgentResponse(
        explanation=Explanation(text="Mock explanation"),
        solution=Solution(text="Mock solution"),
    )

    mock_run_chain = MagicMock()
    mock_run_chain.middleware = AsyncMock(return_value=mock_agent_output)

    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        MockAgent.return_value.run.return_value = mock_run_chain

        with patch.object(SERVER_CONFIG.general, "generate_solution", False):
            response = await analyze_artifacts(mock_artifacts, mock_chat_model)
            assert response.solution is None


@pytest.mark.asyncio
async def test_analyze_artifacts_solution_kept_when_enabled(mock_agent_setup):
    """Test that solution is preserved when generate_solution is True."""
    mock_artifacts, mock_chat_model, mock_agent_output = mock_agent_setup
    mock_agent_output.output_structured = AgentResponse(
        explanation=Explanation(text="Mock explanation"),
        solution=Solution(text="Mock solution"),
    )

    mock_run_chain = MagicMock()
    mock_run_chain.middleware = AsyncMock(return_value=mock_agent_output)

    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        MockAgent.return_value.run.return_value = mock_run_chain

        with patch.object(SERVER_CONFIG.general, "generate_solution", True):
            response = await analyze_artifacts(mock_artifacts, mock_chat_model)
            assert response.solution is not None
            assert response.solution.text == "Mock solution"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_option", "snippet_count", "expecting_lookup_tool"),
    [
        (True, 10, True),
        (False, 10, False),
        (True, 0, False),
    ],
    ids=("everything-set-up", "config-option-off", "db-not-populated"),
)
async def test_analyze_artifacts_lookup_tool_included(
    mock_agent_setup,
    config_option: bool,
    snippet_count: int,
    expecting_lookup_tool: bool,
):
    """AnnotatedSnippetLookupTool is enabled only if the flag is on and the table has data."""
    mock_artifacts, mock_chat_model, mock_agent_output = mock_agent_setup
    with patch("logdetective.server.agent.agent.RequirementAgent") as MockAgent:
        mock_run_instance = MagicMock()
        mock_run_instance.middleware = AsyncMock(return_value=mock_agent_output)
        MockAgent.return_value.run.return_value = mock_run_instance

        with (
            patch.object(SERVER_CONFIG.general, "annotation_lookup_tool", config_option),
            patch.object(SERVER_CONFIG.general, "max_annotations", 3),
            patch("logdetective.server.config.TextEmbedding", MagicMock()),
            patch.object(
                AnnotatedSnippets, "get_count", new_callable=AsyncMock, return_value=snippet_count
            ),
        ):
            model_instance = load_embedding_model(SERVER_CONFIG)
            with patch(
                "logdetective.server.agent.agent.EMBEDDING_MODEL_INSTANCE", model_instance
            ):
                await analyze_artifacts(mock_artifacts, mock_chat_model)

        _, kwargs = MockAgent.call_args
        tools = kwargs.get("tools", [])
        is_tool_included = any(isinstance(t, AnnotatedSnippetLookupTool) for t in tools)
        assert is_tool_included == expecting_lookup_tool
