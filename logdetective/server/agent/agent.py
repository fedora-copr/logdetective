from typing import Optional
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.think import ThinkTool
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware

from logdetective.server.config import PROMPT_CONFIG, SERVER_CONFIG
from logdetective.server.agent.tools import DrainExtractorTool, CSGrepExtractorTool
from logdetective.server.models import APIResponse, BuildMetadata, Explanation
from logdetective.server.exceptions import LogDetectiveAgentResponseFailure


async def analyze_artifacts(
    artifacts: dict[str, str], chat_model: OpenAIChatModel, build_metadata: Optional[BuildMetadata] = None
) -> APIResponse:
    """Analyze build artifacts using Log Detective agent.

    :artifacts: dictionary of build artifacts, each referenced by their file name
    :chat_model: OpenAIChatModel providing inference to the agent
    :build_metadata: BuildMetadata structure, containing additional information
                     Full implementation in future release
    """
    # TODO: Handle build_metadata by supplying it to the agent

    drain_extractor = DrainExtractorTool(
        extractor_config=SERVER_CONFIG.extractor, available_artifacts=artifacts
    )
    csgrep_extractor = None
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
            min_invocations=1,
            max_invocations=len(artifacts),
        ),
    ]

    # Use CSGrepExtractorTool only if it is enabled
    if SERVER_CONFIG.extractor.csgrep:
        csgrep_extractor = CSGrepExtractorTool(
            extractor_config=SERVER_CONFIG.extractor, available_artifacts=artifacts
        )
        tools.append(csgrep_extractor)
        requirements.append(
            ConditionalRequirement(
                CSGrepExtractorTool,
                consecutive_allowed=True,
                max_invocations=len(artifacts),
            )
        )

    # TODO: Use AgentResponse as an output
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
    agent_input = PROMPT_CONFIG.agent_start_prompt.format(
        artifacts=",".join(artifacts.keys())
    )

    agent_output = await agent.run(
        agent_input,
        max_retries_per_step=SERVER_CONFIG.inference.max_retries_per_step,
        total_max_retries=SERVER_CONFIG.inference.total_max_retries,
    ).middleware(middleware)

    if not agent_output.state.answer:
        raise LogDetectiveAgentResponseFailure

    all_snippets = drain_extractor.extracted_snippets

    if csgrep_extractor:
        all_snippets.extend(csgrep_extractor.extracted_snippets)

    response = APIResponse(
        explanation=Explanation(text=agent_output.state.answer.text),
        snippets=all_snippets,
    )
    return response
