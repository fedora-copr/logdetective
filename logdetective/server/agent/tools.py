from typing import Any

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import Tool
from beeai_framework.tools.errors import ToolInputValidationError, ToolError
from beeai_framework.tools.types import ToolOutput, ToolRunOptions
from pydantic import BaseModel, Field
import yaml

from logdetective.extractors import (
    Extractor,
    DrainExtractor,
    CSGrepExtractor,
    PythonTracebackExtractor,
)
from logdetective.models import SkipSnippets
from logdetective.server.models import ExtractorConfig, Snippet, AnalyzedSnippet


class ExtractorToolInput(BaseModel):
    artifact_name: str = Field(
        description="The exact name of the artifact you want to extract information from."
    )


class ExtractorToolOutput(ToolOutput, BaseModel):
    source_artifact: str = Field(
        description="Name of the artifact the snippets were extracted from."
    )
    extracted_snippets: list[Snippet] = Field(
        description="List of snippets extracted from the artifact."
    )
    remaining_artifacts: set[str] = Field(
        description="Set of artifacts that this extractor was not used on yet."
    )

    def get_text_content(self) -> str:
        """Convert the extractor output into YAML"""
        return yaml.safe_dump(
            self.model_dump(
                exclude_defaults=True,
                exclude_none=True,
                exclude_computed_fields=True,
                exclude_unset=True,
                mode="json",
            ),
            default_flow_style=False,
        )

    def is_empty(self) -> bool:
        return not self.extracted_snippets


class ExtractorTool(Tool[ExtractorToolInput]):
    """Base extractor tool class, not intended for direct use with agent."""

    name: str = "base_extractor"
    description: str = (
        "Base extractor tool. Doesn't actually have an extractor. Do not use it."
    )
    description_template: str
    extractor: Extractor
    all_artifacts: dict[str, str]

    extracted_snippets: list[Snippet]
    _remaining_artifacts: set[str]

    def __init__(
        self,
        all_artifacts: dict[str, str],
        schema: type[ExtractorToolInput] = ExtractorToolInput,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options)
        self._input_schema = schema
        self.all_artifacts = all_artifacts
        self.extracted_snippets = []
        self._remaining_artifacts = set(all_artifacts.keys())

    async def _run(
        self,
        input: ExtractorToolInput,
        options: ToolRunOptions | None,
        context: RunContext,
    ) -> ExtractorToolOutput:
        """Extract snippets from selected build artifact."""

        if input.artifact_name not in self.all_artifacts:
            raise ToolInputValidationError(
                f"Requested artifact: {input.artifact_name} does not exist."
            )

        if input.artifact_name not in self._remaining_artifacts:
            raise ToolInputValidationError(
                f"Requested artifact: {input.artifact_name} was already analyzed."
                f"Only following artifacts are available: {self._remaining_artifacts}"
            )
        self._remaining_artifacts.remove(input.artifact_name)

        artifact = self.all_artifacts[input.artifact_name]
        raw_snippets = self.extractor(artifact)
        for line_number, text in raw_snippets:
            self.extracted_snippets.append(
                Snippet(
                    text=text, line_number=line_number, source_file=input.artifact_name
                )
            )
        return ExtractorToolOutput(
            source_artifact=input.artifact_name,
            extracted_snippets=self.extracted_snippets,
            remaining_artifacts=self._remaining_artifacts,
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "extractor"], creator=self)

    @property
    def input_schema(self) -> type[ExtractorToolInput]:
        return self._input_schema


class DrainExtractorTool(ExtractorTool):
    name: str = "drain_extractor"
    description_template: str = (
        "Use this tool at most once per artifact."
        "Extracts up to {max_clusters} snippets from a log file, using clustering Drain algorithm."
        "Maximum length of extracted snippet is {max_snippet_len}."
    )
    extractor: DrainExtractor

    def __init__(
        self,
        extractor_config: ExtractorConfig,
        available_artifacts: dict[str, str],
        skip_snippets: SkipSnippets = SkipSnippets({}),
        options: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(all_artifacts=available_artifacts, options=options)
        self.description = self.description_template.format(
            max_clusters=extractor_config.max_clusters,
            max_snippet_len=extractor_config.max_snippet_len,
        )
        self.extractor = DrainExtractor(
            verbose=extractor_config.verbose,
            skip_snippets=skip_snippets,
            max_snippet_len=extractor_config.max_snippet_len,
            max_clusters=extractor_config.max_clusters,
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "extractor", "drain"], creator=self
        )


class CSGrepExtractorTool(ExtractorTool):
    name: str = "csgrep_extractor"
    description_template: str = (
        "Use this tool at most once per artifact."
        "Extracts up to {max_clusters} snippets from a log file containing GCC traces, using csgrep tool."
        "Do not use on artifacts that don't contain traces from compiler."
        "Maximum length of extracted snippet is {max_snippet_len}."
    )

    extractor: CSGrepExtractor

    def __init__(
        self,
        extractor_config: ExtractorConfig,
        available_artifacts: dict[str, str],
        skip_snippets: SkipSnippets = SkipSnippets({}),
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(all_artifacts=available_artifacts, options=options)
        self.description = self.description_template.format(
            max_clusters=extractor_config.max_clusters,
            max_snippet_len=extractor_config.max_snippet_len,
        )
        self.extractor = CSGrepExtractor(
            verbose=extractor_config.verbose,
            skip_snippets=skip_snippets,
            max_snippet_len=extractor_config.max_snippet_len,
            max_clusters=extractor_config.max_clusters,
            csgrep_timeout=extractor_config.csgrep_timeout,
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "extractor", "csgrep"], creator=self
        )


class PythonTracebackExtractorTool(ExtractorTool):
    name: str = "python_traceback_extractor"
    description_template: str = (
        "Use this tool at most once per artifact. "
        "Extracts Python exception tracebacks from a log file. "
        "Use on artifacts that contain Python output or test results. "
        "Do not use on artifacts that don't contain Python tracebacks. "
        "Maximum length of extracted snippet is {max_snippet_len}."
    )
    extractor: PythonTracebackExtractor

    def __init__(
        self,
        extractor_config: ExtractorConfig,
        available_artifacts: dict[str, str],
        skip_snippets: SkipSnippets = SkipSnippets({}),
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(all_artifacts=available_artifacts, options=options)
        self.description = self.description_template.format(
            max_snippet_len=extractor_config.max_snippet_len,
        )
        self.extractor = PythonTracebackExtractor(
            verbose=extractor_config.verbose,
            skip_snippets=skip_snippets,
            max_snippet_len=extractor_config.max_snippet_len,
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "extractor", "python_traceback"], creator=self
        )


class SnippetAnalysisToolInput(BaseModel):
    source_file: str = Field(
        description="Name of the file the snippet was extracted from."
    )
    line_number: int = Field(
        description="Location of the snippet in the original file."
    )
    snippet_analysis: str = Field(description="Analysis of given snippet")


class SnippetAnalysisToolOutput(ToolOutput, BaseModel):
    snippet: Snippet = Field(description="Snippet that was analyzed")
    snippet_analysis: str = Field(description="Explanation of the analyzed snippet")

    def get_text_content(self) -> str:
        return f"Snippet:\n    {self.snippet.text}\nAnalysis:\n    {self.snippet_analysis}"

    def is_empty(self) -> bool:
        return not self.snippet_analysis or not self.snippet.text


class SnippetAnalysisTool(
    Tool[SnippetAnalysisToolInput, ToolRunOptions, SnippetAnalysisToolOutput]
):
    name: str = "snippet_analysis"
    description: str = (
        "Analyzes single snippet from list of all extracted snippets."
        "Only the most interesting snippets require separate analysis."
    )

    _extractors: list[ExtractorTool]

    def __init__(
        self,
        extractors: list[ExtractorTool],
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options)

        self._extractors = extractors

    async def _run(
        self,
        input: SnippetAnalysisToolInput,
        options: ToolRunOptions | None,
        context: RunContext,
    ) -> SnippetAnalysisToolOutput:
        """Add annotation to one snippet from all snippets extracted. Extractors do not allow
        for creation of duplicate snippets, so we should see at most one snippet for any
        combination of source_file and line_number.

        In the event that no snippet match given criteria, an error is raised.
        """

        # This sanity check does not guarantee that snippet actually exists
        # only that the it could have been extracted at some point.
        if not any(
            input.source_file in extractor.all_artifacts
            for extractor in self._extractors
        ):
            raise ToolInputValidationError(
                f"Given source file: {input.source_file} does not exist."
            )

        for extractor in self._extractors:
            for snippet_index, snippet in enumerate(extractor.extracted_snippets):
                if snippet.source_file == input.source_file and snippet.line_number == input.line_number:
                    if isinstance(snippet, AnalyzedSnippet):
                        raise ToolInputValidationError(
                            f"Selected snippet from file: {input.source_file} line: {input.line_number} has been analyzed already."
                        )
                    extractor.extracted_snippets[snippet_index] = AnalyzedSnippet(
                        text=snippet.text,
                        line_number=snippet.line_number,
                        source_file=snippet.source_file,
                        snippet_analysis=input.snippet_analysis,
                    )
                    return SnippetAnalysisToolOutput(
                        snippet=snippet, snippet_analysis=input.snippet_analysis
                    )

        raise ToolError(
            message=f"Given source file: {input.source_file} and line number: {input.line_number}, "
            "don't correspond to an existing snippet."
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "snippet_analysis"], creator=self
        )

    @property
    def input_schema(self) -> type[SnippetAnalysisToolInput]:
        return SnippetAnalysisToolInput
