from enum import Enum
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from beeai_framework.context import RunContext, RunInstance
from beeai_framework.emitter import Emitter
from beeai_framework.tools.errors import ToolError
from logdetective.exceptions import (
    RemoteLogAccessError,
    RemoteLogRequestError,
    RemoteLogTooLargeError,
)
from logdetective.remote_log import RemoteLog
from logdetective.server.agent.tools import (
    DrainExtractorTool,
    ExtractorTool,
    ExtractorToolInput,
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
async def test_snippet_analysis_already_analyzed():
    """ToolInputValidationError when attempting to analyze a snippet that was already analyzed."""
    from beeai_framework.tools.errors import ToolInputValidationError

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
        source_file=source_file, line_number=0, snippet_analysis="first analysis"
    )
    await tool._run(
        input=analysis_input,
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )
    with pytest.raises(ToolInputValidationError):
        await tool._run(
            input=analysis_input,
            context=RunContext(instance=MockRunInstance(), signal=None),
            options=None,
        )


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


@pytest.mark.asyncio
async def test_extractor_tool_downloads_remote_log():
    """When an artifact is a RemoteLog, _run fetches its content before extraction."""
    source_file = "remote.log"
    log_content = "ERROR: something went wrong\nTraceback follows"

    mock_remote_log = AsyncMock(spec=RemoteLog)
    mock_remote_log.get_url_content.return_value = log_content

    tool = DrainExtractorTool(
        ExtractorConfig(),
        available_artifacts={source_file: mock_remote_log},
    )

    ArtifactName = Enum("ArtifactName", {source_file: source_file}, type=str)
    input_data = ExtractorToolInput(artifact_name=ArtifactName(source_file))

    result = await tool._run(
        input=input_data,
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )

    mock_remote_log.get_url_content.assert_awaited_once()
    assert result.source_artifact == source_file
    assert source_file not in result.remaining_artifacts


@pytest.mark.asyncio
async def test_extractor_tool_uses_string_artifact_directly():
    """When an artifact is a plain string, _run uses it without calling get_url_content."""
    source_file = "local.log"
    log_content = "INFO: build completed successfully"

    tool = DrainExtractorTool(
        ExtractorConfig(),
        available_artifacts={source_file: log_content},
    )

    ArtifactName = Enum("ArtifactName", {source_file: source_file}, type=str)
    input_data = ExtractorToolInput(artifact_name=ArtifactName(source_file))

    result = await tool._run(
        input=input_data,
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )

    assert result.source_artifact == source_file
    assert source_file not in result.remaining_artifacts


@pytest.mark.asyncio
async def test_extractor_tool_mixed_artifacts():
    """Tool handles a mix of string and RemoteLog artifacts across calls."""
    string_file = "local.log"
    remote_file = "remote.log"
    string_content = "local log content"
    remote_content = "remote log content"

    mock_remote_log = AsyncMock(spec=RemoteLog)
    mock_remote_log.get_url_content.return_value = remote_content

    tool = DrainExtractorTool(
        ExtractorConfig(),
        available_artifacts={
            string_file: string_content,
            remote_file: mock_remote_log,
        },
    )

    ArtifactName = Enum(
        "ArtifactName",
        {string_file: string_file, remote_file: remote_file},
        type=str,
    )

    result_local = await tool._run(
        input=ExtractorToolInput(artifact_name=ArtifactName(string_file)),
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )
    assert result_local.source_artifact == string_file
    mock_remote_log.get_url_content.assert_not_awaited()

    result_remote = await tool._run(
        input=ExtractorToolInput(artifact_name=ArtifactName(remote_file)),
        context=RunContext(instance=MockRunInstance(), signal=None),
        options=None,
    )
    assert result_remote.source_artifact == remote_file
    mock_remote_log.get_url_content.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("remote_error", [
    RemoteLogAccessError("connection refused"),
    RemoteLogRequestError("invalid URL"),
    RemoteLogTooLargeError("exceeds limit"),
])
async def test_extractor_tool_remote_log_failure_raises_tool_error(remote_error):
    """RemoteLogError from get_url_content is wrapped in ToolError."""
    source_file = "failing.log"

    mock_remote_log = AsyncMock(spec=RemoteLog)
    mock_remote_log.get_url_content.side_effect = remote_error

    tool = DrainExtractorTool(
        ExtractorConfig(),
        available_artifacts={source_file: mock_remote_log},
    )

    ArtifactName = Enum("ArtifactName", {source_file: source_file}, type=str)
    input_data = ExtractorToolInput(artifact_name=ArtifactName(source_file))

    with pytest.raises(ToolError, match=source_file) as exc_info:
        await tool._run(
            input=input_data,
            context=RunContext(instance=MockRunInstance(), signal=None),
            options=None,
        )

    assert exc_info.value.__cause__ is remote_error
