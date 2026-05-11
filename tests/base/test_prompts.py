import os

import pytest

from jinja2 import exceptions
from logdetective.prompts import PromptManager
from logdetective.models import PromptConfig
import logdetective


def test_prompt_manager():
    """Test that PromptManager can be properly initilized with built-in prompts."""
    manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    )

    assert isinstance(manager.default_system_prompt, str)
    rendered_snippet_message = manager.render_message_template(snippets="Snippet")
    assert isinstance(rendered_snippet_message, str)
    assert "Snippet" in rendered_snippet_message


def test_prompt_manager_with_config():
    """Test that PromptManager can be properly initilized with built-in prompts and PromptConfig"""
    config = PromptConfig()
    config.references = [{"name": "Reference 1", "link": "https://valid_link.url"}]
    manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        prompts_configuration=config,
    )

    rendered_system_prompt = manager.default_system_prompt
    assert isinstance(rendered_system_prompt, str)
    assert manager._references
    assert manager._references[0]["name"] in rendered_system_prompt
    assert manager._references[0]["link"] in rendered_system_prompt

    rendered_snippet_message = manager.render_message_template(snippets="Snippet")
    assert isinstance(rendered_snippet_message, str)
    assert "Snippet" in rendered_snippet_message
    assert manager._references == config.references


def test_prompt_manager_no_templates():
    """Test that manager raises exception when encountering empty or invalid template dir."""

    with pytest.raises(exceptions.TemplateNotFound):
        PromptManager("/no/prompts/here")


def test_prompt_manager_agent_start_prompt_render_artifacts():
    """Test that agent_start_prompt renders artifacts properly"""

    artifacts = [
        "build.log",
        "builder-live.log",
    ]
    prompt_manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    )

    agent_start_prompt = prompt_manager.agent_start_prompt(artifacts=artifacts)

    for artifact in artifacts:
        assert artifact in agent_start_prompt
