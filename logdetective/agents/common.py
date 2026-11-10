"""Tool setup and result utilities shared by independent analysis workflows."""

from dataclasses import dataclass
import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.cache import UnconstrainedCache
from beeai_framework.errors import AbortError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.utils.cancellation import AbortController
from litellm.exceptions import RateLimitError as LiteLLMRateLimit
from litellm.exceptions import Timeout as LiteLLMTimeout
from pydantic import ValidationError
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, stop_any,
    stop_before_delay, wait_exponential_jitter,
)

from logdetective.agents.tools import (
    AnnotatedSnippetLookupTool, CSGrepExtractorTool, DrainExtractorTool,
    ExtractorTool, PythonTracebackExtractorTool, SnippetAnnotationTool,
)
from logdetective.config import (
    SERVER_CONFIG, SKIP_SNIPPETS_CONFIG, get_embedding_model,
)
from logdetective.database.models.annotated_builds import AnnotatedSnippets
from logdetective.exceptions import (
    LogDetectiveAgentResponseFailure, LogDetectiveAgentTimeoutError,
    LogDetectiveInferenceError, LogDetectiveInferenceRateLimit,
    LogDetectiveInferenceTimeout,
)
from logdetective.models import APIResponse, AgentResponse
from logdetective.remote_log import RemoteLog
from logdetective.utils import (
    inference_retry_backoff, inference_retry_giveup, run_blocking,
    blocking_work_scope, wait_for_cleanup,
)


inference_retry = retry(
    stop=stop_any(
        stop_after_attempt(SERVER_CONFIG.inference.retry_max_tries),
        stop_before_delay(SERVER_CONFIG.inference.retry_max_time),
    ),
    wait=wait_exponential_jitter(),
    retry=retry_if_exception_type((
        LogDetectiveInferenceTimeout, LogDetectiveInferenceRateLimit,
    )),
    before_sleep=inference_retry_backoff,
    retry_error_callback=inference_retry_giveup,
)


@dataclass
class ToolSetup:
    """Tools, call requirements, and extractors for one agent run."""

    tools: list[Tool]
    requirements: list[ConditionalRequirement]
    extractors: list[ExtractorTool]

    def add_tool(
        self,
        tool: Tool,
        requirements: tuple[ConditionalRequirement, ...] = (),
    ) -> None:
        """Register a workflow-specific tool and its call constraints.

        Extractors join the list used by build annotation and final response
        assembly. Tool names must remain unique for prompt
        guidance and BeeAI tool calls to be unambiguous.
        """
        if any(existing.name == tool.name for existing in self.tools):
            raise ValueError(f"Duplicate agent tool name: {tool.name}")
        self.tools.append(tool)
        self.requirements.extend(requirements)
        if isinstance(tool, ExtractorTool):
            self.extractors.append(tool)


async def build_tool_setup(
    artifacts: dict[str, str | RemoteLog], *, require_drain: bool = True,
    include_annotation: bool = True, preextracted_evidence: bool = False,
) -> ToolSetup:
    """Configure required and optional extraction tools for either workflow."""
    drain = DrainExtractorTool(
        extractor_config=SERVER_CONFIG.extractor,
        available_artifacts=artifacts,
        skip_snippets=SKIP_SNIPPETS_CONFIG,
        options={"cache": UnconstrainedCache()},
    )
    setup = ToolSetup(tools=[], requirements=[], extractors=[])
    setup.add_tool(drain, (ConditionalRequirement(
        DrainExtractorTool, consecutive_allowed=True,
        min_invocations=len(artifacts) if require_drain else 0,
        max_invocations=len(artifacts),
    ),))

    if SERVER_CONFIG.extractor.csgrep:
        setup.add_tool(CSGrepExtractorTool(
            extractor_config=SERVER_CONFIG.extractor,
            available_artifacts=artifacts,
            skip_snippets=SKIP_SNIPPETS_CONFIG,
            options={"cache": UnconstrainedCache()},
        ), (ConditionalRequirement(
            CSGrepExtractorTool, consecutive_allowed=True,
            max_invocations=len(artifacts),
        ),))

    if SERVER_CONFIG.extractor.python_traceback:
        setup.add_tool(PythonTracebackExtractorTool(
            extractor_config=SERVER_CONFIG.extractor,
            available_artifacts=artifacts,
            skip_snippets=SKIP_SNIPPETS_CONFIG,
            options={"cache": UnconstrainedCache()},
        ), (ConditionalRequirement(
            PythonTracebackExtractorTool, consecutive_allowed=True,
            max_invocations=len(artifacts),
        ),))

    if SERVER_CONFIG.general.annotation_lookup_tool:
        if await AnnotatedSnippets.get_count() > 0:
            embedding_model = await run_blocking(get_embedding_model)
            if embedding_model is not None:
                setup.add_tool(AnnotatedSnippetLookupTool(
                    embedding_model, options={"cache": UnconstrainedCache()},
                ), (ConditionalRequirement(
                    AnnotatedSnippetLookupTool, consecutive_allowed=True,
                    only_after=None if preextracted_evidence else ExtractorTool,
                    max_invocations=5,
                ),))

    if include_annotation:
        setup.add_tool(SnippetAnnotationTool(
            extractors=setup.extractors, options={"cache": UnconstrainedCache()},
        ), (ConditionalRequirement(
            SnippetAnnotationTool, consecutive_allowed=True,
            only_after=ExtractorTool, max_invocations=5,
        ),))
    return setup


async def run_agent(
    agent: RequirementAgent, agent_input: str,
    expected_output: type = AgentResponse,
    timeout_seconds: float | None = None,
) -> AgentResponse:
    """Run one configured agent and validate its structured output."""
    try:
        async with blocking_work_scope():
            controller = AbortController()
            timer = asyncio.get_running_loop().call_later(
                timeout_seconds if timeout_seconds is not None
                else SERVER_CONFIG.general.analysis_timeout,
                controller.abort, "Agent deadline exceeded",
            )
            try:
                operation = asyncio.ensure_future(agent.run(
                    agent_input,
                    max_retries_per_step=SERVER_CONFIG.inference.max_retries_per_step,
                    total_max_retries=SERVER_CONFIG.inference.total_max_retries,
                    expected_output=expected_output,
                    signal=controller.signal,
                ).middleware(GlobalTrajectoryMiddleware(pretty=True)))
                try:
                    raw_output = await asyncio.shield(operation)
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    timed_out = controller.signal.aborted and current.cancelling() == 0
                    # BeeAI must receive its abort signal, not an outer task
                    # cancellation which can leave its nested runners detached.
                    controller.abort("Agent task cancelled")
                    await wait_for_cleanup(asyncio.gather(operation, return_exceptions=True))
                    if timed_out:
                        # Nested BeeAI runners can surface their own cancellation
                        # instead of AbortError when the deadline signal fires.
                        raise LogDetectiveAgentTimeoutError("Agent deadline exceeded") from None
                    raise
            finally:
                timer.cancel()
    except ChatModelError as exc:
        cause = exc.__cause__
        if isinstance(cause, LiteLLMTimeout):
            raise LogDetectiveInferenceTimeout(str(cause)) from exc
        if isinstance(cause, LiteLLMRateLimit):
            raise LogDetectiveInferenceRateLimit(str(cause)) from exc
        raise LogDetectiveInferenceError(exc.message) from exc
    except AbortError as exc:
        raise LogDetectiveAgentTimeoutError(exc.message) from exc

    if not raw_output.output_structured:
        raise LogDetectiveAgentResponseFailure
    try:
        return expected_output.model_validate(raw_output.output_structured)
    except ValidationError as exc:
        raise LogDetectiveInferenceError("Invalid format of the agent response") from exc


def to_api_response(output: AgentResponse, extractors: list[ExtractorTool]) -> APIResponse:
    """Attach extracted evidence and apply the configured solution policy."""
    response = APIResponse(
        explanation=output.explanation.text,
        solution=output.solution.text if output.solution else None,
        no_issue_found=output.no_issue_found,
        snippets=[snippet for tool in extractors for snippet in tool.extracted_snippets],
    )
    if not SERVER_CONFIG.general.generate_solution:
        response.solution = None
    return response
