import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, HttpUrl


class PromptReference(BaseModel):
    """Reference to some web-source, passed to the system prompt"""

    name: str
    link: HttpUrl


class PromptConfig(BaseModel):
    """Model for prompt configuration of log detective."""

    references: list[PromptReference] = Field(default_factory=list)

    @field_validator("references", mode="before")
    @classmethod
    def make_references_optional(cls, v: Any) -> Any:
        """Convert undefined references to an empty list."""
        return [] if v is None else v


class SkipPattern(BaseModel):
    """One loaded skip pattern. If `files` is None, pattern is skipped from every file."""

    pattern: re.Pattern
    files: set[str] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("pattern", mode="after")
    @classmethod
    def apply_dotall_flag(cls, pattern: re.Pattern) -> re.Pattern:
        """Default re.Pattern pydantic model does not apply re.DOTALL flag"""
        return re.compile(pattern.pattern, pattern.flags | re.DOTALL)


class SkipSnippets(BaseModel):
    """Regular expressions defining snippets we should not analyze/pass to LLM."""

    snippet_patterns: dict[str, SkipPattern] = {}


class CSGrepEvent(BaseModel):
    """`csgrep` splits error and warning messages into individual events."""

    file_name: str  # references the source file which compilation error talks about
    line: int  # line number in the source file
    event: str
    message: str
    input_file: str  # references the name of the compilation log (absolute path)
    input_line: int  # line number in the compilation log
    verbosity_level: int


class CSGrepDefect(BaseModel):
    """Defects detected by `csgrep`"""

    checker: str
    language: str
    tool: str
    key_event_idx: int
    events: list[CSGrepEvent]


class CSGrepOutput(BaseModel):
    """Parsed output of `gsgrep`"""

    defects: list[CSGrepDefect]
