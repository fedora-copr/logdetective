from datetime import datetime, timezone
from typing import Optional
from jinja2 import Environment, FileSystemLoader, Template

from logdetective.models import (
    PromptReference, Config, TestingFarmSuiteRequest,
)


class PromptManager:
    """Manages prompts defined as jinja templates"""

    _tmp_env: Environment

    # Templates for system prompts
    _default_system_prompt_template: Template
    _default_agent_start_prompt_template: Template
    _references: list[PromptReference] = []

    def __init__(self, prompts_path: str, server_config: Config) -> None:
        self._tmp_env = Environment(loader=FileSystemLoader(prompts_path))

        self._default_agent_start_prompt_template = self._tmp_env.get_template(
            "agent_start_prompt.j2"
        )
        self._default_system_prompt_template = self._tmp_env.get_template(
            "system_prompt.j2"
        )
        self._testing_farm_system_template = self._tmp_env.get_template(
            "testing_farm_system_prompt.j2"
        )
        self._testing_farm_suite_start_template = self._tmp_env.get_template(
            "testing_farm_suite_start_prompt.j2"
        )
        self._references = server_config.prompts.references

    @property
    def default_system_prompt(self) -> str:
        """Render system prompt from a template"""
        return self._default_system_prompt_template.render(
            system_time=datetime.now(timezone.utc), references=self._references
        )

    @property
    def testing_farm_system_prompt(self) -> str:
        """Render the Testing Farm-specific system instructions."""
        return self._testing_farm_system_template.render(
            system_time=datetime.now(timezone.utc)
        )

    def testing_farm_suite_start_prompt(
        self, request: TestingFarmSuiteRequest, snippets: list[dict],
        available_tools: set[str], available_artifacts: set[str],
        eligible_case_ids: set[str],
    ) -> str:
        """Render all case context separately from evidence-eligible diagnosis IDs."""
        # Use available artifact names rather than all submitted references:
        # failed downloads must not be offered to the agent as inspectable logs.
        log_manifest = {
            "shared": [
                f"shared/{log.name}" for log in request.shared_logs
                if f"shared/{log.name}" in available_artifacts
            ],
            "cases": {
                case.case_id: [
                    f"case/{case.case_id}/{log.name}" for log in case.logs
                    if f"case/{case.case_id}/{log.name}" in available_artifacts
                ]
                for case in request.cases
            },
        }
        return self._testing_farm_suite_start_template.render(
            suite=request.suite.model_dump(exclude_none=True),
            guests={name: guest.model_dump(exclude_none=True)
                    for name, guest in request.guests.items()},
            cases=[case.model_dump(exclude={"logs"}) for case in request.cases],
            eligible_case_ids=sorted(eligible_case_ids),
            snippets=snippets, log_manifest=log_manifest,
            available_tools=available_tools,
        )

    def agent_start_prompt(
        self,
        artifacts: list[str],
        commentary: Optional[str] = None,
        infra_status: Optional[str] = None,
        available_tools: Optional[set[str]] = None,
    ) -> str:
        """Render agent start prompt"""

        return self._default_agent_start_prompt_template.render(
            artifacts=artifacts,
            commentary=commentary,
            infra_status=infra_status,
            available_tools=available_tools or set(),
        )
