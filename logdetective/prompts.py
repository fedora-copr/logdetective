from datetime import datetime, timezone
from typing import Optional
from jinja2 import Environment, FileSystemLoader, Template

from logdetective.models import PromptConfig


class PromptManager:  # pylint: disable=too-many-instance-attributes
    """Manages prompts defined as jinja templates"""
    _tmp_env: Environment

    # Templates for system prompts
    _default_system_prompt_template: Template
    _snippet_system_prompt_template: Template
    _staged_system_prompt_template: Template

    # Templates for messages
    default_message_template: Template
    snippet_message_template: Template
    staged_message_template: Template

    _references: Optional[list[dict[str, str]]] = None

    def __init__(
        self, prompts_path: str, prompts_configuration: Optional[PromptConfig] = None
    ) -> None:
        self._tmp_env = Environment(loader=FileSystemLoader(prompts_path))

        self._default_system_prompt_template = self._tmp_env.get_template(
            "system_prompt.j2"
        )
        self._snippet_system_prompt_template = self._tmp_env.get_template(
            "snippet_system_prompt.j2"
        )
        self._staged_system_prompt_template = self._tmp_env.get_template(
            "staged_system_prompt.j2"
        )

        self.default_message_template = self._tmp_env.get_template(
            "message_template.j2"
        )
        self.snippet_message_template = self._tmp_env.get_template(
            "snippet_message_template.j2"
        )
        self.staged_message_template = self._tmp_env.get_template(
            "staged_message_template.j2"
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
    def snippet_system_prompt(self) -> str:
        """Render system prompt from a template"""
        return self._snippet_system_prompt_template.render(
            system_time=datetime.now(timezone.utc), references=self._references
        )

    @property
    def staged_system_prompt(self) -> str:
        """Render system prompt from a template"""
        return self._staged_system_prompt_template.render(
            system_time=datetime.now(timezone.utc), references=self._references
        )

    @property
    def prompt_template(self) -> str:
        """Render message prompt from the template"""
        return self.default_message_template.render()

    @property
    def snippet_prompt_template(self) -> str:
        """Render message prompt from the template"""
        return self.snippet_message_template.render()

    @property
    def prompt_template_staged(self) -> str:
        """Render message prompt from the template"""
        return self.staged_message_template.render()
