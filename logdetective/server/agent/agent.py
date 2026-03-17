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
from logdetective.server.models import Response, Explanation
from logdetective.server.exceptions import LogDetectiveAgentResponseFailure


async def analyze_artifacts(
    artifacts: dict[str, str], chat_model: OpenAIChatModel
) -> Response:
    """Analyze build artifacts using Log Detective agent."""

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
    middleware = GlobalTrajectoryMiddleware()

    # Names of build artifacts are inserted into the template.
    agent_input = PROMPT_CONFIG.agent_start_prompt.format(
        artifacts=",".join(artifacts.keys())
    )

    agent_output = await agent.run(agent_input).middleware(middleware)

    if not agent_output.state.answer:
        raise LogDetectiveAgentResponseFailure

    all_snippets = drain_extractor.extracted_snippets

    if csgrep_extractor:
        all_snippets.extend(csgrep_extractor.extracted_snippets)

    response = Response(
        explanation=Explanation(text=agent_output.state.answer.text),
        snippets=all_snippets,
        response_certainty=0.0,
    )
    return response
