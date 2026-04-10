from typing import Any

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import Tool
from beeai_framework.tools.types import ToolOutput, ToolRunOptions
from pydantic import BaseModel, Field

from logdetective.extractors import DrainExtractor, CSGrepExtractor, Extractor
from logdetective.server.models import ExtractorConfig, Snippet


class ExtractorToolInput(BaseModel):
    artifact_name: str = Field(
        description="The exact name of the artifact you want to extract information from."
    )


class ExtractorToolOutput(ToolOutput, BaseModel):
    source_artifact: str = Field(
        description="Name of the artifact the snippets were extracted from."
    )
    extracted_snippets: list[Snippet] = Field(
        description="Snippets extracted from the artifact. Each element is a tuple of original line number, and the extracted text."
    )
    remaining_artifacts: set[str] = Field(
        description="Set of artifacts that this extractor was not used on yet."
    )

    def get_text_content(self) -> str:
        return f"source_artifact: {self.source_artifact}, extracted_snippets: {self.extracted_snippets}, remaining_artifacts: {self.remaining_artifacts}"

    def is_empty(self) -> bool:
        return False


class ExtractorTool(Tool[ExtractorToolInput]):
    """Base extractor tool class, not intended for direct use with agent."""

    name: str = "base_extractor"
    description: str = (
        "Base extractor tool. Doesn't actually have an extractor. Do not use it."
    )
    description_template: str
    extractor: Extractor
    available_artifacts: dict[str, str]

    extracted_snippets: list[Snippet]
    _remaining_artifacts: set[str]

    def __init__(
        self,
        available_artifacts: dict[str, str],
        schema: type[ExtractorToolInput] = ExtractorToolInput,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options)
        self._input_schema = schema
        self.available_artifacts = available_artifacts
        self.extracted_snippets = []
        self._remaining_artifacts = set(available_artifacts.keys())

    async def _run(
        self,
        input: ExtractorToolInput,
        options: ToolRunOptions | None,
        context: RunContext,
    ) -> ExtractorToolOutput:
        """Extract snippets from selected build artifact."""

        self._remaining_artifacts.remove(input.artifact_name)

        artifact = self.available_artifacts[input.artifact_name]
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
        options: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(available_artifacts=available_artifacts, options=options)
        self.description = self.description_template.format(
            max_clusters=extractor_config.max_clusters,
            max_snippet_len=extractor_config.max_snippet_len,
        )
        self.extractor = DrainExtractor(
            verbose=extractor_config.verbose,
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
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(available_artifacts=available_artifacts, options=options)
        self.description = self.description_template.format(
            max_clusters=extractor_config.max_clusters,
            max_snippet_len=extractor_config.max_snippet_len,
        )
        self.extractor = CSGrepExtractor(
            verbose=extractor_config.verbose,
            max_snippet_len=extractor_config.max_snippet_len,
            max_clusters=extractor_config.max_clusters,
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "extractor", "csgrep"], creator=self
        )
