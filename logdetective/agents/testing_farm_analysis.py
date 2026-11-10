"""One agent invocation (with whole-suite transient retries) per Testing Farm suite."""

from datetime import UTC, datetime

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.cache import UnconstrainedCache

from logdetective.agents.common import build_tool_setup, inference_retry, run_agent
from logdetective.agents.tools import DrainExtractorTool
from logdetective.config import PROMPT_CONFIG, SERVER_CONFIG, SKIP_SNIPPETS_CONFIG
from logdetective.models import Snippet, TestingFarmAgentOutput, TestingFarmSuiteRequest
from logdetective.utils import run_blocking


async def _preextract_suite_evidence(artifacts: dict[str, str]) -> dict[str, Snippet]:
    """Run mandatory Drain extraction once for every available log."""
    evidence: dict[str, Snippet] = {}
    for name, content in artifacts.items():
        extractor = DrainExtractorTool(
            extractor_config=SERVER_CONFIG.extractor,
            available_artifacts={name: content},
            skip_snippets=SKIP_SNIPPETS_CONFIG,
            options={"cache": UnconstrainedCache()},
        )
        for line, snippet_text in await run_blocking(extractor.extractor, content, filename=name):
            evidence[f"{name}:{line}"] = Snippet(
                text=snippet_text, line_number=line, source_file=name
            )
    return evidence


@inference_retry
async def _analyze_suite(
    artifacts: dict[str, str], chat_model: ChatModel,
    request: TestingFarmSuiteRequest, evidence: dict[str, Snippet],
    deadline_at: datetime | None, eligible_case_ids: set[str],
) -> tuple[TestingFarmAgentOutput, dict[str, Snippet]]:
    """Rebuild agent/tool state for a transient retry of the entire suite."""
    setup = await build_tool_setup(
        artifacts, require_drain=False, include_annotation=False,
        preextracted_evidence=bool(evidence),
    )
    drain = next(
        (tool for tool in setup.extractors if isinstance(tool, DrainExtractorTool)), None
    )
    if drain is not None:
        drain.extracted_snippets.extend(evidence.values())
    agent = RequirementAgent(
        llm=chat_model, memory=UnconstrainedMemory(),
        name="testing-farm-suite-agent",
        instructions=PROMPT_CONFIG.testing_farm_system_prompt,
        role=SERVER_CONFIG.inference.system_role,
        tools=setup.tools, requirements=setup.requirements,
    )
    prompt = PROMPT_CONFIG.testing_farm_suite_start_prompt(
        request, [snippet.model_dump() for snippet in evidence.values()],
        {tool.name for tool in setup.tools}, set(artifacts), eligible_case_ids,
    )
    remaining = ((deadline_at - datetime.now(UTC)).total_seconds()
                 if deadline_at is not None else SERVER_CONFIG.general.analysis_timeout)
    if remaining <= 0:
        raise TimeoutError("Suite deadline exceeded")
    output = await run_agent(agent, prompt, TestingFarmAgentOutput,
                             timeout_seconds=remaining)
    ids = [case.case_id for case in output.cases]
    if len(ids) != len(set(ids)) or set(ids) - eligible_case_ids:
        raise ValueError("Agent returned duplicate or ineligible case IDs")
    if request.cases and output.suite_fallback is not None:
        raise ValueError("Agent returned an unexpected suite fallback")
    for tool in setup.extractors:
        for snippet in tool.extracted_snippets:
            evidence.setdefault(f"{snippet.source_file}:{snippet.line_number}", snippet)
    return output, evidence


async def analyze_testing_farm_suite(
    artifacts: dict[str, str], chat_model: ChatModel,
    request: TestingFarmSuiteRequest, deadline_at: datetime | None = None,
    *, eligible_case_ids: set[str],
) -> tuple[TestingFarmAgentOutput, dict[str, Snippet]]:
    """Diagnose eligible IDs in one run with every submitted case as context."""
    evidence = await _preextract_suite_evidence(artifacts)
    return await _analyze_suite(
        artifacts, chat_model, request, evidence, deadline_at, eligible_case_ids
    )
