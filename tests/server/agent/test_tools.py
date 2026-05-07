from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from beeai_framework.context import RunContext, RunInstance
from beeai_framework.emitter import Emitter
from logdetective.server.agent.tools import (
    DrainExtractorTool,
    ExtractorTool,
    SnippetAnalysisTool,
    SnippetAnalysisToolInput,
    SnippetAnalysisToolOutput,
)
from logdetective.server.models import ExtractorConfig, Snippet, AnalyzedSnippet


class MockRunInstance(RunInstance):
    @property
    def emitter(self) -> Emitter:
        return Emitter()


@pytest.mark.asyncio
async def test_snippet_analysis_tool_init():

    snippet_analysis = "Mock analyis"
    source_file = "build.log"
    line_number = 0
    artifact_content = "extracted text"
    snippet = Snippet(
        text=artifact_content, line_number=line_number, source_file=source_file
    )
    extractors: list[ExtractorTool] = [
        DrainExtractorTool(
            ExtractorConfig(), available_artifacts={source_file: artifact_content}
        )
    ]
    extractors[0].extracted_snippets.append(snippet)
    tool = SnippetAnalysisTool(extractors=extractors)
    analysis_input = SnippetAnalysisToolInput(
        source_file=source_file,
        line_number=line_number,
        snippet_analysis=snippet_analysis,
    )
    result = await tool._run(
        input=analysis_input,
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )

    assert result.snippet_analysis == snippet_analysis
    assert result.snippet == snippet


@pytest.mark.asyncio
async def test_snippet_analysis_nonexistent_source_file():
    """ToolInputValidationError when source_file is not in any extractor's available_artifacts."""
    from beeai_framework.tools.errors import ToolInputValidationError

    source_file = "build.log"
    extractors: list[ExtractorTool] = [
        DrainExtractorTool(
            ExtractorConfig(), available_artifacts={source_file: "content"}
        )
    ]
    tool = SnippetAnalysisTool(extractors=extractors)
    analysis_input = SnippetAnalysisToolInput(
        source_file="nonexistent.log", line_number=0, snippet_analysis="analysis"
    )
    with pytest.raises(ToolInputValidationError):
        await tool._run(
            input=analysis_input,
            context=RunContext(instance=MockRunInstance(), signal=None),
            options=None,
        )


@pytest.mark.asyncio
async def test_snippet_analysis_wrong_line_number():
    """ToolError when source_file exists but no snippet matches the line_number."""
    from beeai_framework.tools.errors import ToolError

    source_file = "build.log"
    artifact_content = "extracted text"
    snippet = Snippet(text=artifact_content, line_number=0, source_file=source_file)
    extractors: list[ExtractorTool] = [
        DrainExtractorTool(
            ExtractorConfig(), available_artifacts={source_file: artifact_content}
        )
    ]
    extractors[0].extracted_snippets.append(snippet)
    tool = SnippetAnalysisTool(extractors=extractors)
    analysis_input = SnippetAnalysisToolInput(
        source_file=source_file, line_number=999, snippet_analysis="analysis"
    )
    with pytest.raises(ToolError):
        await tool._run(
            input=analysis_input,
            context=RunContext(instance=MockRunInstance(), signal=None),
            options=None,
        )


@pytest.mark.asyncio
async def test_snippet_analysis_mutates_extractor_snippets():
    """After analysis, the extractor's snippet list contains AnalyzedSnippet in place of original."""
    source_file = "build.log"
    artifact_content = "extracted text"
    snippet = Snippet(text=artifact_content, line_number=0, source_file=source_file)
    extractors: list[ExtractorTool] = [
        DrainExtractorTool(
            ExtractorConfig(), available_artifacts={source_file: artifact_content}
        )
    ]
    extractors[0].extracted_snippets.append(snippet)
    tool = SnippetAnalysisTool(extractors=extractors)
    analysis_input = SnippetAnalysisToolInput(
        source_file=source_file, line_number=0, snippet_analysis="this is important"
    )
    await tool._run(
        input=analysis_input,
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )
    mutated = extractors[0].extracted_snippets[0]
    assert isinstance(mutated, AnalyzedSnippet)
    assert mutated.snippet_analysis == "this is important"
    assert mutated.text == artifact_content
    assert mutated.line_number == 0
    assert mutated.source_file == source_file


@pytest.mark.asyncio
async def test_snippet_analysis_multiple_extractors():
    """Snippet in the second extractor is found and analyzed correctly."""
    file_a = "a.log"
    file_b = "b.log"
    snippet_b = Snippet(text="error in b", line_number=42, source_file=file_b)

    extractor_a = DrainExtractorTool(
        ExtractorConfig(), available_artifacts={file_a: "content a"}
    )
    extractor_b = DrainExtractorTool(
        ExtractorConfig(), available_artifacts={file_b: "content b"}
    )
    extractor_b.extracted_snippets.append(snippet_b)

    tool = SnippetAnalysisTool(extractors=[extractor_a, extractor_b])
    analysis_input = SnippetAnalysisToolInput(
        source_file=file_b, line_number=42, snippet_analysis="root cause"
    )
    result = await tool._run(
        input=analysis_input,
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )
    assert result.snippet == snippet_b
    assert result.snippet_analysis == "root cause"
    assert isinstance(extractor_b.extracted_snippets[0], AnalyzedSnippet)


@pytest.mark.asyncio
async def test_snippet_analysis_no_extracted_snippets():
    """ToolError when source_file is valid but no snippets have been extracted yet."""
    from beeai_framework.tools.errors import ToolError

    source_file = "build.log"
    extractors: list[ExtractorTool] = [
        DrainExtractorTool(
            ExtractorConfig(), available_artifacts={source_file: "content"}
        )
    ]
    tool = SnippetAnalysisTool(extractors=extractors)
    analysis_input = SnippetAnalysisToolInput(
        source_file=source_file, line_number=0, snippet_analysis="analysis"
    )
    with pytest.raises(ToolError):
        await tool._run(
            input=analysis_input,
            context=RunContext(instance=MockRunInstance(), signal=None),
            options=None,
        )


def test_snippet_analysis_output_get_text_content():
    snippet = Snippet(text="some error", line_number=5, source_file="build.log")
    output = SnippetAnalysisToolOutput(snippet=snippet, snippet_analysis="looks bad")
    text = output.get_text_content()
    assert "some error" in text
    assert "looks bad" in text


def test_snippet_analysis_output_is_empty():
    snippet = Snippet(text="", line_number=0, source_file="build.log")
    output_empty_text = SnippetAnalysisToolOutput(
        snippet=snippet, snippet_analysis="analysis"
    )
    assert output_empty_text.is_empty()

    snippet_with_text = Snippet(text="content", line_number=0, source_file="build.log")
    output_empty_analysis = SnippetAnalysisToolOutput(
        snippet=snippet_with_text, snippet_analysis=""
    )
    assert output_empty_analysis.is_empty()

    output_full = SnippetAnalysisToolOutput(
        snippet=snippet_with_text, snippet_analysis="analysis"
    )
    assert not output_full.is_empty()
