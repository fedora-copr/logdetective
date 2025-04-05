from typing import Optional
from pydantic import BaseModel

from logdetective.constants import (
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_STAGED,
    SUMMARIZATION_PROMPT_TEMPLATE,
    SNIPPET_PROMPT_TEMPLATE,
)


class PromptConfig(BaseModel):
    """Configuration for basic log detective prompts."""

    prompt_template: str = PROMPT_TEMPLATE
    summarization_prompt_template: str = SUMMARIZATION_PROMPT_TEMPLATE
    snippet_prompt_template: str = SNIPPET_PROMPT_TEMPLATE
    prompt_template_staged: str = PROMPT_TEMPLATE_STAGED

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return
        self.prompt_template = data.get("prompt_template", PROMPT_TEMPLATE)
        self.summarization_prompt_template = data.get(
            "summarization_prompt_template", SUMMARIZATION_PROMPT_TEMPLATE
        )
        self.snippet_prompt_template = data.get(
            "snippet_prompt_template", SNIPPET_PROMPT_TEMPLATE
        )
        self.prompt_template_staged = data.get(
            "prompt_template_staged", PROMPT_TEMPLATE_STAGED
        )
