import os
import logging
import yaml
import httpx
from openai import AsyncOpenAI
from beeai_framework.adapters.openai import OpenAIChatModel

from logdetective.utils import load_prompts, load_skip_snippet_patterns
from logdetective.server.models import Config, InferenceConfig
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


def get_openai_chat_model(inference_config: InferenceConfig) -> OpenAIChatModel:
    """Set up OpenAI chat model for Log Detective agent"""
    return OpenAIChatModel(
        model_id=inference_config.model,
        api_key=inference_config.api_token,
        base_url=inference_config.url,
        tool_choice_support={"auto"},
    )


def get_openai_api_client(inference_config: InferenceConfig):
    """Set up AsyncOpenAI client with default configuration."""
    limits = httpx.Limits(
        max_connections=inference_config.max_concurrent_requests,
        max_keepalive_connections=inference_config.max_keep_alive_connections,
    )
    return AsyncOpenAI(
        api_key=inference_config.api_token, base_url=inference_config.url,
        timeout=inference_config.llm_api_timeout,
        http_client=httpx.AsyncClient(limits=limits)  # Defaults are too restrictive
    )


SERVER_CONFIG_PATH = os.environ.get("LOGDETECTIVE_SERVER_CONF", None)
SERVER_PROMPT_CONF_PATH = os.environ.get("LOGDETECTIVE_PROMPTS", None)
SERVER_PROMPT_PATH = os.environ.get("LOGDETECTIVE_PROMPT_TEMPLATES", None)
# The default location for skip patterns is in the same directory
# as logdetective __init__.py file.
SERVER_SKIP_PATTERNS_PATH = os.environ.get(
    "LOGDETECIVE_SKIP_PATTERNS",
    f"{os.path.dirname(logdetective.__file__)}/skip_snippets.yml",
)

SERVER_CONFIG = load_server_config(SERVER_CONFIG_PATH)
PROMPT_CONFIG = load_prompts(SERVER_PROMPT_CONF_PATH, SERVER_PROMPT_PATH)
SKIP_SNIPPETS_CONFIG = load_skip_snippet_patterns(SERVER_SKIP_PATTERNS_PATH)

LOG = get_log(SERVER_CONFIG)

CLIENT = get_openai_api_client(SERVER_CONFIG.inference)
