import os
import logging
import yaml
from beeai_framework.backend import ChatModel
from beeai_framework.backend.chat import ChatModelParameters

from logdetective.utils import load_prompts, load_skip_snippet_patterns
from logdetective.server.models import Config, InferenceConfig
from logdetective.constants import PROMPT_PATH, PROMPT_CONF_PATH
import logdetective


def load_server_config(path: str | None) -> Config:
    """Load configuration file for logdetective server.
    If no path was provided, or if the file doesn't exist, return defaults.
    """
    if path is not None:
        try:
            with open(path, "r") as config_file:
                return Config.model_validate(yaml.safe_load(config_file))
        except FileNotFoundError:
            # This is not an error, we will fall back to default
            print("Unable to find server config file, using default then.")
    return Config()


def get_log(config: Config):
    """
    Initialize a logger for this server
    """
    log = logging.getLogger(config.log.name)
    if getattr(log, "initialized", False):
        return log

    log.setLevel("DEBUG")

    # Drop the default handler, we will create it ourselves
    log.handlers = []

    # STDOUT
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(config.log.format))
    stream_handler.setLevel(config.log.level_stream)
    log.addHandler(stream_handler)

    # Log to file
    if config.log.path:
        file_handler = logging.FileHandler(config.log.path)
        file_handler.setFormatter(logging.Formatter(config.log.format))
        file_handler.setLevel(config.log.level_file)
        log.addHandler(file_handler)

    log.initialized = True
    return log


def get_chat_model(inference_config: InferenceConfig) -> ChatModel:
    """Set up chat model for Log Detective agent"""
    # Accept bare model names (e.g. "granite-4.0-h-tiny") as shorthand for "openai:<name>"
    model_name = inference_config.model
    if ":" not in model_name:
        model_name = f"openai:{model_name}"
    return ChatModel.from_name(
        model_name,
        ChatModelParameters(
            temperature=inference_config.temperature,
            max_tokens=inference_config.max_tokens,
        ),
        tool_choice_support={"auto"},
        settings={"timeout": inference_config.api_timeout},
    )


SERVER_CONFIG_PATH = os.environ.get("LOGDETECTIVE_SERVER_CONF", None)

# The default location for skip patterns is in the same directory
# as logdetective __init__.py file.
SERVER_SKIP_PATTERNS_PATH = os.environ.get(
    "LOGDETECIVE_SKIP_PATTERNS",
    f"{os.path.dirname(logdetective.__file__)}/skip_snippets.yml",
)

SERVER_CONFIG = load_server_config(SERVER_CONFIG_PATH)
PROMPT_CONFIG = load_prompts(
    template_path=PROMPT_PATH, config_path=PROMPT_CONF_PATH
)
SKIP_SNIPPETS_CONFIG = load_skip_snippet_patterns(SERVER_SKIP_PATTERNS_PATH)

LOG = get_log(SERVER_CONFIG)
