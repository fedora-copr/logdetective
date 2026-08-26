import os

import pytest

from jinja2.exceptions import TemplateNotFound
from logdetective.constants import PROMPT_PATH
from logdetective.prompts import PromptManager
from logdetective.models import PromptConfig, PromptReference
import logdetective


def test_prompt_manager():
    """Test that PromptManager can be properly initilized with built-in prompts."""

    manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    )
    assert isinstance(manager.default_system_prompt, str)

    assert isinstance(PromptManager(PROMPT_PATH), PromptManager)


def test_prompt_manager_wrong_path():
    """Test behavior for case when the path doesn't lead to a any file."""

    with pytest.raises(TemplateNotFound):
        PromptManager("/there/is/nothing/to/read.yml")

    with pytest.raises(TemplateNotFound):
        PromptManager("/no/prompts/here")


def test_prompt_manager_with_config():
    """Test that PromptManager can be properly initilized
    with built-in prompts, PromptConfig, and PromptReferences"""

    mock_config = PromptConfig(
        references=[PromptReference(name="Reference 1", link="https://valid_link.url")]
    )
    prompts_dir = os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    manager = PromptManager(prompts_dir, prompt_config=mock_config)

    rendered_system_prompt = manager.default_system_prompt
    assert isinstance(rendered_system_prompt, str)
    assert manager._references
    assert manager._references[0].name in rendered_system_prompt
    assert str(manager._references[0].link) in rendered_system_prompt


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


def test_prompt_manager_agent_start_prompt_render_commentary():
    """Test that prompt renders commentary if supplied"""

    commentary = "COMMENT ABOUT BUILD"
    artifacts = [
        "build.log",
    ]
    prompt_manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    )

    agent_start_prompt = prompt_manager.agent_start_prompt(
        artifacts=artifacts, commentary=commentary
    )

    for artifact in artifacts:
        assert artifact in agent_start_prompt

    assert commentary in agent_start_prompt


def test_prompt_manager_agent_start_prompt_render_infra_status():
    """Test that prompt renders infra_status if supplied"""

    infra_status = "INFRASTRUCTURE STATUS"
    artifacts = [
        "build.log",
    ]
    prompt_manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    )

    agent_start_prompt = prompt_manager.agent_start_prompt(
        artifacts=artifacts, infra_status=infra_status
    )

    for artifact in artifacts:
        assert artifact in agent_start_prompt

    assert infra_status in agent_start_prompt


def test_prompt_manager_agent_start_prompt_render_supplementary():
    """Test that prompt renders supplementary information if supplied"""

    infra_status = "INFRASTRUCTURE STATUS"
    commentary = "COMMENT ABOUT BUILD"
    artifacts = [
        "build.log",
    ]
    prompt_manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    )

    agent_start_prompt = prompt_manager.agent_start_prompt(
        artifacts=artifacts, infra_status=infra_status, commentary=commentary
    )

    for artifact in artifacts:
        assert artifact in agent_start_prompt

    assert infra_status in agent_start_prompt
    assert commentary in agent_start_prompt
