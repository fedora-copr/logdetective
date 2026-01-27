import re
from typing import Optional
from pydantic import BaseModel, model_validator

from logdetective.constants import (
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_STAGED,
    SNIPPET_PROMPT_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
)


class PromptConfig(BaseModel):
    """Configuration for basic log detective prompts."""

    prompt_template: str = PROMPT_TEMPLATE
    snippet_prompt_template: str = SNIPPET_PROMPT_TEMPLATE
    prompt_template_staged: str = PROMPT_TEMPLATE_STAGED

    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    snippet_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    staged_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    references: Optional[list[dict[str, str]]] = None


class SkipSnippets(BaseModel):
    """Regular expressions defining snippets we should not analyze"""

    snippet_patterns: dict[str, re.Pattern] = {}

    def __init__(self, data: Optional[dict] = None):
        super().__init__(data=data)
        if data is None:
            return
        self.snippet_patterns = {
            key: re.compile(pattern) for key, pattern in data.items()
        }

    @model_validator(mode="before")
    @classmethod
    def check_patterns(cls, data: dict):
        """Check if all supplied patterns are valid regular expressions.
        Techically replicating what is done in __init__ but with nicer error message."""
        patterns = data["data"]
        for key, pattern in patterns.items():
            try:
                re.compile(pattern=pattern)
            except (TypeError, re.error) as ex:
                raise ValueError(
                    f"Invalid pattern `{pattern}` with name `{key}` supplied for skipping in logs."
                ) from ex

        return data


class CSGrepEvent(BaseModel):
    """`csgrep` splits error and warning messages into individual events."""

    file_name: str
    line: int
    event: str
    message: str
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
