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
    assert isinstance(manager.staged_system_prompt, str)
    assert isinstance(manager.snippet_system_prompt, str)

    assert isinstance(manager.prompt_template, str)
    assert isinstance(manager.snippet_prompt_template, str)
    assert isinstance(manager.prompt_template_staged, str)


def test_prompt_manager_with_config():
    """Test that PromptManager can be properly initilized with built-in prompts and PromptConfig"""
    config = PromptConfig()
    config.references = [{"name": "Reference 1", "link": "https://valid_link.url"}]
    manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        prompts_configuration=config,
    )

    assert isinstance(manager.default_system_prompt, str)
    assert isinstance(manager.staged_system_prompt, str)
    assert isinstance(manager.snippet_system_prompt, str)

    assert isinstance(manager.prompt_template, str)
    assert isinstance(manager.snippet_prompt_template, str)
    assert isinstance(manager.prompt_template_staged, str)

    assert manager._references == config.references


def test_prompt_manager_no_templates():
    """Test that manager raises exception when encountering empty or invalid template dir."""

    with pytest.raises(exceptions.TemplateNotFound):
        PromptManager("/no/prompts/here")
