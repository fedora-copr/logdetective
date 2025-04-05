import os
import yaml
from functools import lru_cache

PROMPTS_YAML_PATH = os.path.join(os.path.dirname(__file__), "prompts.yaml")

@lru_cache()
def _load_yaml(path: str = PROMPTS_YAML_PATH) -> dict:
    """Load YAML content from the given file path."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_yaml_data = _load_yaml()

DEFAULT_ADVISOR = _yaml_data.get("default_advisor", "fedora-copr/Mistral-7B-Instruct-v0.2-GGUF")
SNIPPET_DELIMITER = _yaml_data.get("constants", {}).get("snippet_delimiter", "================")

PROMPT_TEMPLATE = _yaml_data["prompts"]["analyze_snippets"]["template"]
PROMPT_TEMPLATE_STAGED = _yaml_data["prompts"]["staged_analysis"]["template"]
SUMMARIZE_PROMPT_TEMPLATE = _yaml_data["prompts"]["summarize_log"]["template"]
SNIPPET_PROMPT_TEMPLATE = _yaml_data["prompts"]["analyze_snippet_only"]["template"]
