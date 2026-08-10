import re
from typing import Optional
from pydantic import BaseModel, model_validator


class PromptConfig(BaseModel):
    """Configuration for basic log detective prompts."""

    references: Optional[list[dict[str, str]]] = None


class SkipSnippets(BaseModel):
    """Regular expressions defining snippets we should not analyze.

    Each entry in the source dict must have the form:
        name:
            pattern: "regex"
            files:          # optional; if absent the pattern applies to every file
                - "exact_filename.log"
    """

    # maps name -> (compiled pattern, set of exact filenames or None for global)
    snippet_patterns: dict[
        str,
        tuple[
            re.Pattern,
            set[str] | None,
        ],
    ] = {}

    def __init__(self, data: Optional[dict] = None):
        super().__init__(data=data)
        if data is None:
            return
        result = {}
        for key, value in data.items():
            compiled = re.compile(value["pattern"], re.DOTALL)
            files = set(value["files"]) if "files" in value else None
            result[key] = (compiled, files)
        self.snippet_patterns = result

    def get_patterns_for_file(self, filename: str | None) -> dict[str, re.Pattern]:
        """Return compiled patterns applicable to the given filename.
        If unspecified, return globally applied patterns.

        Patterns without a files list are always included. Patterns with a
        files list are included only when filename is an exact match.
        """
        result = {}
        for name, (pattern, files) in self.snippet_patterns.items():
            if files is None or (filename and filename in files):
                result[name] = pattern
        return result

    @model_validator(mode="before")
    @classmethod
    def check_patterns(cls, data: dict):
        """Validate that all supplied patterns are valid regular expressions."""
        patterns = data["data"]
        for key, value in patterns.items():
            if not isinstance(value, dict) or "pattern" not in value:
                raise ValueError(
                    f"Pattern `{key}` must be a mapping with a 'pattern' key."
                )
            try:
                re.compile(pattern=value["pattern"])
            except (TypeError, re.error) as ex:
                raise ValueError(
                    (
                        f"Invalid pattern `{value['pattern']}` "
                        f"with name `{key}` supplied for skipping in logs."
                    )
                ) from ex

        return data


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
