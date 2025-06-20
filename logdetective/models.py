from typing import Optional
from pydantic import BaseModel

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

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return
        self.prompt_template = data.get("prompt_template", PROMPT_TEMPLATE)
        self.snippet_prompt_template = data.get(
            "snippet_prompt_template", SNIPPET_PROMPT_TEMPLATE
        )
        self.prompt_template_staged = data.get(
            "prompt_template_staged", PROMPT_TEMPLATE_STAGED
        )
        self.default_system_prompt = data.get(
            "default_system_prompt", DEFAULT_SYSTEM_PROMPT
        )
        self.snippet_system_prompt = data.get(
            "snippet_system_prompt", DEFAULT_SYSTEM_PROMPT
        )
        self.staged_system_prompt = data.get(
            "staged_system_prompt", DEFAULT_SYSTEM_PROMPT
        )
