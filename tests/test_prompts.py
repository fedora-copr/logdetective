import os

import pytest

from jinja2.exceptions import TemplateNotFound
from logdetective.constants import PROMPT_PATH
from logdetective.prompts import PromptManager
from logdetective.models import Config, PromptConfig, PromptReference
import logdetective


def test_prompt_manager():
    """Test that PromptManager can be properly initilized with built-in prompts."""

    manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        server_config=Config(),
    )
    assert isinstance(manager.default_system_prompt, str)

    assert isinstance(PromptManager(PROMPT_PATH, server_config=Config()), PromptManager)


def test_prompt_manager_wrong_path():
    """Test behavior for case when the path doesn't lead to a any file."""

    with pytest.raises(TemplateNotFound):
        PromptManager("/there/is/nothing/to/read.yml", server_config=Config())

    with pytest.raises(TemplateNotFound):
        PromptManager("/no/prompts/here", server_config=Config())


def test_prompt_manager_with_config():
    """Test that PromptManager can be properly initilized
    with built-in prompts, PromptConfig, and PromptReferences"""

    mock_config = Config(
        prompts=PromptConfig(
            references=[
                PromptReference(name="Reference 1", link="https://valid_link.url")
            ]
        )
    )
    prompts_dir = os.path.join(os.path.dirname(logdetective.__file__), "prompts")
    manager = PromptManager(prompts_dir, server_config=mock_config)

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
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        server_config=Config(),
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
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        server_config=Config(),
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
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        server_config=Config(),
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
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        server_config=Config(),
    )

    agent_start_prompt = prompt_manager.agent_start_prompt(
        artifacts=artifacts, infra_status=infra_status, commentary=commentary
    )

    for artifact in artifacts:
        assert artifact in agent_start_prompt

    assert infra_status in agent_start_prompt
    assert commentary in agent_start_prompt


def test_prompt_manager_agent_start_prompt_uses_available_tools():
    """Tool guidance follows tools offered to the current agent run."""
    prompt_manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        server_config=Config(),
    )

    agent_start_prompt = prompt_manager.agent_start_prompt(
        artifacts=["build.log"],
        available_tools={
            "csgrep_extractor", "python_traceback_extractor",
            "annotated_snippet_lookup", "snippet_annotation",
        },
    )

    assert "use `csgrep_extractor`" in agent_start_prompt
    assert "use `python_traceback_extractor`" in agent_start_prompt
    assert "call `annotated_snippet_lookup`" in agent_start_prompt
    assert "specific extracted error" in agent_start_prompt
    assert "similar past examples, not proof" in agent_start_prompt
    assert "Ignore empty or irrelevant results" in agent_start_prompt
    assert "call `snippet_annotation`" in agent_start_prompt
    assert "`snippet_analysis` argument" in agent_start_prompt
    assert "call `snippet_analysis`" not in agent_start_prompt


def test_prompt_manager_agent_start_prompt_omits_unavailable_tools():
    """Unavailable tools are not recommended in the agent prompt."""
    prompt_manager = PromptManager(
        os.path.join(os.path.dirname(logdetective.__file__), "prompts"),
        server_config=Config(),
    )

    agent_start_prompt = prompt_manager.agent_start_prompt(artifacts=["build.log"])

    assert "use `csgrep_extractor`" not in agent_start_prompt
    assert "use `python_traceback_extractor`" not in agent_start_prompt
    assert "annotated_snippet_lookup" not in agent_start_prompt
    assert "snippet_annotation" not in agent_start_prompt


@pytest.mark.parametrize(
    "tool_name",
    ["csgrep_extractor", "python_traceback_extractor",
     "annotated_snippet_lookup", "snippet_annotation"],
)
def test_prompt_manager_only_shows_supplied_optional_tool(tool_name):
    """Each optional tool section is independent of the others."""
    prompt_manager = PromptManager(PROMPT_PATH, server_config=Config())
    rendered = prompt_manager.agent_start_prompt(
        artifacts=["build.log"], available_tools={tool_name}
    )
    assert f"`{tool_name}`" in rendered
    for other_tool in {
        "csgrep_extractor", "python_traceback_extractor",
        "annotated_snippet_lookup", "snippet_annotation",
    } - {tool_name}:
        assert f"`{other_tool}`" not in rendered
