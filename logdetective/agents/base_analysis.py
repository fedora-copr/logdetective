"""Build-artifact analysis workflow."""

import re
from typing import Optional

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.think import ThinkTool

from logdetective.agents.common import (
    build_tool_setup, inference_retry, run_agent, to_api_response,
)
from logdetective.config import LOG, PROMPT_CONFIG, SERVER_CONFIG
from logdetective.models import APIResponse, BuildMetadata
from logdetective.remote_log import RemoteLog


@inference_retry
async def analyze_artifacts(
    artifacts: dict[str, str | RemoteLog],
    chat_model: ChatModel,
    build_metadata: Optional[BuildMetadata] = None,
) -> APIResponse:
    """Analyze build artifacts with the build-specific agent and prompt."""
    setup = await build_tool_setup(artifacts)
    setup.tools.insert(0, ThinkTool())
    setup.requirements.insert(0, ConditionalRequirement(
        ThinkTool, force_at_step=1, consecutive_allowed=False, max_invocations=2,
    ))
    agent = RequirementAgent(
        llm=chat_model,
        memory=UnconstrainedMemory(),
        name="log-detective-agent",
        instructions=PROMPT_CONFIG.default_system_prompt,
        role=SERVER_CONFIG.inference.system_role,
        tools=setup.tools,
        requirements=setup.requirements,
    )

    available_tools = {tool.name for tool in setup.tools}
    if build_metadata:
        # Reject an early XML closing tag in supplementary build context.
        xml_closing_tag = re.compile(r"</[^<]+>", re.DOTALL)
        if build_metadata.commentary and re.match(xml_closing_tag, build_metadata.commentary):
            LOG.warning("Suspicious commentary field - contains xml tags - rejecting")
            build_metadata.commentary = None
        if build_metadata.infra_status and re.match(xml_closing_tag, build_metadata.infra_status):
            LOG.warning("Suspicious infra_status field - contains xml tags - rejecting")
            build_metadata.infra_status = None
        agent_input = PROMPT_CONFIG.agent_start_prompt(
            artifacts=list(artifacts),
            commentary=build_metadata.commentary,
            infra_status=build_metadata.infra_status,
            available_tools=available_tools,
        )
    else:
        agent_input = PROMPT_CONFIG.agent_start_prompt(
            artifacts=list(artifacts), available_tools=available_tools,
        )

    output = await run_agent(agent, agent_input)
    return to_api_response(output, setup.extractors)
