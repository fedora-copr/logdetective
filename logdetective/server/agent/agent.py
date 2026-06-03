from typing import Optional

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.think import ThinkTool
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from litellm.exceptions import (
    Timeout as LiteLLMTimeout,
    RateLimitError as LiteLLMRateLimit
)
from pydantic import ValidationError

from logdetective.server.config import PROMPT_CONFIG, SERVER_CONFIG, SKIP_SNIPPETS_CONFIG
from logdetective.server.agent.tools import (
    ExtractorTool,
    DrainExtractorTool,
    CSGrepExtractorTool,
    PythonTracebackExtractorTool,
    SnippetAnalysisTool,
)
from logdetective.server.models import APIResponse, BuildMetadata, AgentResponse
from logdetective.server.exceptions import (
    LogDetectiveAgentResponseFailure,
    LogDetectiveInferenceTimeout,
    LogDetectiveInferenceError,
    LogDetectiveInferenceRateLimit,
)


async def analyze_artifacts(
    artifacts: dict[str, str],
    chat_model: OpenAIChatModel,
    build_metadata: Optional[BuildMetadata] = None
) -> APIResponse:
    """Analyze build artifacts using Log Detective agent.

    :artifacts: dictionary of build artifacts, each referenced by their file name
    :chat_model: OpenAIChatModel providing inference to the agent
    :build_metadata: BuildMetadata structure, containing additional information
                     Full implementation in future release
    """
    # TODO: Handle build_metadata by supplying it to the agent

    drain_extractor = DrainExtractorTool(
        extractor_config=SERVER_CONFIG.extractor,
        available_artifacts=artifacts,
        skip_snippets=SKIP_SNIPPETS_CONFIG,
    )
    csgrep_extractor = None
    python_tb_extractor = None
    tools = [
        ThinkTool(),
        drain_extractor,
    ]

    requirements = [
        # We don't want the agent to think more than twice
        ConditionalRequirement(
            ThinkTool, force_at_step=1, consecutive_allowed=False, max_invocations=2
        ),
        # At most, allow one invocation of extractor per artifact
        ConditionalRequirement(
            DrainExtractorTool,
            consecutive_allowed=True,
            min_invocations=len(artifacts),
            max_invocations=len(artifacts),
        ),
    ]

    # Use CSGrepExtractorTool only if it is enabled
    if SERVER_CONFIG.extractor.csgrep:
        csgrep_extractor = CSGrepExtractorTool(
            extractor_config=SERVER_CONFIG.extractor,
            available_artifacts=artifacts,
            skip_snippets=SKIP_SNIPPETS_CONFIG,
        )
        tools.append(csgrep_extractor)
        requirements.append(
            ConditionalRequirement(
                CSGrepExtractorTool,
                consecutive_allowed=True,
                max_invocations=len(artifacts),
            )
        )

    if SERVER_CONFIG.extractor.python_traceback:
        python_tb_extractor = PythonTracebackExtractorTool(
            extractor_config=SERVER_CONFIG.extractor,
            available_artifacts=artifacts,
            skip_snippets=SKIP_SNIPPETS_CONFIG,
        )
        tools.append(python_tb_extractor)
        requirements.append(
            ConditionalRequirement(
                PythonTracebackExtractorTool,
                consecutive_allowed=True,
                max_invocations=len(artifacts),
            )
        )

    # Add snippet analysis tool, link all extractors and condition it to run after them
    # max_invocations are set at 5. Most snippets are not informative, and annotating them
    # provides no benefit.
    extractors = [tool for tool in tools if isinstance(tool, ExtractorTool)]
    tools.append(SnippetAnalysisTool(extractors=extractors))
    requirements.append(
        ConditionalRequirement(
            SnippetAnalysisTool,
            consecutive_allowed=True,
            only_after=ExtractorTool,
            max_invocations=5,
        )
    )

    agent = RequirementAgent(
        llm=chat_model,
        memory=UnconstrainedMemory(),
        name="log-detective-agent",
        instructions=PROMPT_CONFIG.default_system_prompt,
        role=SERVER_CONFIG.inference.system_role,
        tools=tools,
        requirements=requirements,
    )

    # Middleware for observability and potentially snippet retrieval
    middleware = GlobalTrajectoryMiddleware(pretty=True)

    # Names of build artifacts are inserted into the template.
    if build_metadata:
        agent_input = PROMPT_CONFIG.agent_start_prompt(artifacts=list(artifacts.keys()), commentary=build_metadata.commentary)
    else:
        agent_input = PROMPT_CONFIG.agent_start_prompt(artifacts=list(artifacts.keys()))

    try:
        raw_output = await agent.run(
            agent_input,
            max_retries_per_step=SERVER_CONFIG.inference.max_retries_per_step,
            total_max_retries=SERVER_CONFIG.inference.total_max_retries,
            expected_output=AgentResponse
        ).middleware(middleware)
    except ChatModelError as exc:
        cause = exc.__cause__
        if isinstance(cause, LiteLLMTimeout):
            raise LogDetectiveInferenceTimeout(str(cause)) from exc
        if isinstance(cause, LiteLLMRateLimit):
            raise LogDetectiveInferenceRateLimit(str(cause)) from exc
        raise LogDetectiveInferenceError(exc.message) from exc

    if not raw_output.output_structured:
        raise LogDetectiveAgentResponseFailure
    try:
        structured_output: AgentResponse = AgentResponse.model_validate(raw_output.output_structured)
    except ValidationError as exc:
        raise LogDetectiveInferenceError("Invalid format of the agent response") from exc

    all_snippets = drain_extractor.extracted_snippets

    if csgrep_extractor:
        all_snippets.extend(csgrep_extractor.extracted_snippets)

    if python_tb_extractor:
        all_snippets.extend(python_tb_extractor.extracted_snippets)

    response = APIResponse(
        explanation=structured_output.explanation,
        solution=structured_output.solution,
        no_issue_found=structured_output.no_issue_found,
        snippets=all_snippets,
    )
    return response
