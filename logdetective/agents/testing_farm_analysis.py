"""One agent invocation (with whole-suite transient retries) per Testing Farm suite."""

from itertools import chain

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.cache import UnconstrainedCache

from logdetective.agents.common import (
    AnalysisAttempt, analysis_attempt, build_tool_setup, inference_retry, run_agent,
)
from logdetective.agents.tools import DrainExtractorTool, ExtractorTool
from logdetective.config import PROMPT_CONFIG, SERVER_CONFIG, SKIP_SNIPPETS_CONFIG
from logdetective.models import (
    Snippet, TestingFarmAgentOutput, TestingFarmEvidenceIndex, TestingFarmSuiteRequest,
)
from logdetective.utils import run_blocking


async def _preextract_suite_evidence(
    artifacts: dict[str, str], attempt: AnalysisAttempt,
) -> dict[str, Snippet]:
    """Extract each log within the attempt budget, joining any active thread."""
    evidence: dict[str, Snippet] = {}
    for name, content in artifacts.items():
        attempt.check_deadline()
        extractor = DrainExtractorTool(
            extractor_config=SERVER_CONFIG.extractor,
            available_artifacts={name: content},
            skip_snippets=SKIP_SNIPPETS_CONFIG,
            options={"cache": UnconstrainedCache()},
        )
        snippets = await run_blocking(extractor.extractor, content, filename=name)
        attempt.check_deadline()
        for line, snippet_text in snippets:
            evidence[f"{name}:{line}"] = Snippet(
                text=snippet_text, line_number=line, source_file=name
            )
    return evidence


def _collect_suite_evidence(
    preextracted: dict[str, Snippet], extractors: list[ExtractorTool],
) -> TestingFarmEvidenceIndex:
    """Retain every distinct variant at a location without changing the retry baseline."""
    evidence: TestingFarmEvidenceIndex = {}
    for snippet in chain(
        preextracted.values(),
        chain.from_iterable(tool.extracted_snippets for tool in extractors),
    ):
        variants = evidence.setdefault(f"{snippet.source_file}:{snippet.line_number}", [])
        if not any(existing.text == snippet.text for existing in variants):
            variants.append(snippet)
    return evidence


async def _run_suite_attempt(
    artifacts: dict[str, str], chat_model: ChatModel,
    request: TestingFarmSuiteRequest, evidence: dict[str, Snippet],
    eligible_case_ids: set[str],
    attempt: AnalysisAttempt,
) -> tuple[TestingFarmAgentOutput, TestingFarmEvidenceIndex]:
    """Rebuild agent/tool state for a transient retry of the entire suite."""
    setup = await build_tool_setup(
        artifacts, require_drain=False, include_annotation=False,
        preextracted_evidence=bool(evidence),
    )
    attempt.check_deadline()
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
    output = await run_agent(agent, prompt, TestingFarmAgentOutput, attempt=attempt)
    ids = [case.case_id for case in output.cases]
    if len(ids) != len(set(ids)) or set(ids) - eligible_case_ids:
        raise ValueError("Agent returned duplicate or ineligible case IDs")
    if request.cases and output.suite_fallback is not None:
        raise ValueError("Agent returned an unexpected suite fallback")
    return output, _collect_suite_evidence(evidence, setup.extractors)


async def analyze_testing_farm_suite(
    artifacts: dict[str, str], chat_model: ChatModel,
    request: TestingFarmSuiteRequest,
    *, eligible_case_ids: set[str],
    deadline: float,
) -> tuple[TestingFarmAgentOutput, TestingFarmEvidenceIndex]:
    """Diagnose eligible IDs in one run with every submitted case as context."""
    # Cache even an empty baseline across retries, but never across suites.
    evidence = None
    async for retry_attempt in inference_retry.copy():
        with retry_attempt:
            async with analysis_attempt(deadline) as attempt:
                if evidence is None:
                    evidence = await _preextract_suite_evidence(artifacts, attempt)
                attempt.check_deadline()
                return await _run_suite_attempt(
                    artifacts, chat_model, request, evidence, eligible_case_ids, attempt,
                )
