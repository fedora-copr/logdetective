from datetime import datetime, timezone
from typing import Optional
from jinja2 import Environment, FileSystemLoader, Template

from logdetective.models import PromptReference, PromptConfig


class PromptManager:  # pylint: disable=too-many-instance-attributes
    """Manages prompts defined as jinja templates"""

    _tmp_env: Environment

    # Templates for system prompts
    _default_system_prompt_template: Template
    _default_agent_start_prompt_template: Template
    _references: list[PromptReference] = []

    def __init__(self, prompts_path: str, prompt_config: PromptConfig | None = None) -> None:
        self._tmp_env = Environment(loader=FileSystemLoader(prompts_path))

        self._default_agent_start_prompt_template = self._tmp_env.get_template(
            "agent_start_prompt.j2"
        )
        self._default_system_prompt_template = self._tmp_env.get_template(
            "system_prompt.j2"
        )
        self._references = prompt_config.references if prompt_config else []

    @property
    def default_system_prompt(self) -> str:
        """Render system prompt from a template"""
        return self._default_system_prompt_template.render(
            system_time=datetime.now(timezone.utc), references=self._references
        )

    def agent_start_prompt(
        self,
        artifacts: list[str],
        commentary: Optional[str] = None,
        infra_status: Optional[str] = None,
    ) -> str:
        """Render agent start prompt"""

        return self._default_agent_start_prompt_template.render(
            artifacts=artifacts,
            commentary=commentary,
            infra_status=infra_status,
        )
