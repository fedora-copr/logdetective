from datetime import datetime, timezone
from typing import Optional
from jinja2 import Environment, FileSystemLoader, Template

from logdetective.models import PromptConfig
from logdetective.constants import AGENT_START_PROMPT


class PromptManager:  # pylint: disable=too-many-instance-attributes
    """Manages prompts defined as jinja templates"""
    _tmp_env: Environment

    # Templates for system prompts
    _default_system_prompt_template: Template

    agent_start_prompt: str = AGENT_START_PROMPT

    _references: Optional[list[dict[str, str]]] = None

    def __init__(
        self, prompts_path: str, prompts_configuration: Optional[PromptConfig] = None
    ) -> None:
        self._tmp_env = Environment(loader=FileSystemLoader(prompts_path))

        self._default_system_prompt_template = self._tmp_env.get_template(
            "system_prompt.j2"
        )
        self.default_message_template = self._tmp_env.get_template(
            "message_template.j2"
        )
        if prompts_configuration:
            self._references = prompts_configuration.references

    # To maintain backward compatibility with `logdetective.models.PromptConfig`
    @property
    def default_system_prompt(self) -> str:
        """Render system prompt from a template"""
        return self._default_system_prompt_template.render(
            system_time=datetime.now(timezone.utc), references=self._references
        )

    @property
    def prompt_template(self) -> str:
        """Render message prompt from the template"""
        return self.default_message_template.render()
