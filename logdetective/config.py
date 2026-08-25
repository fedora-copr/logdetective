import os
import logging
import tomllib
import yaml
from beeai_framework.backend import ChatModel
from beeai_framework.backend.types import ChatModelParameters
from fastembed import TextEmbedding

from logdetective.constants import PROMPT_PATH, EMBEDDING_MODEL
from logdetective.models import Config, InferenceConfig, SkipSnippets
import logdetective
from logdetective.prompts import PromptManager


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
    return ChatModel.from_name(
        inference_config.model,
        ChatModelParameters(
            temperature=inference_config.temperature,
            max_tokens=inference_config.max_tokens,
        ),
        tool_choice_support={"auto"},
        settings={
            **inference_config.provider_settings,
            "timeout": inference_config.api_timeout,
        },
    )


def load_embedding_model(config: Config) -> TextEmbedding | None:
    """Load embedding model, if DB lookup is configured."""
    if config.general.annotation_lookup_tool:
        try:
            return TextEmbedding(EMBEDDING_MODEL)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.exception("Embedding model load failed: %s", str(exc))
            return None
    return None


def load_skip_snippet_patterns(path: str | None) -> SkipSnippets | None:
    """Load dictionary of snippet patterns we want to skip."""
    if not path:
        return None
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
        if not data:
            LOG.warning("Skip pattern file `%s` is empty, no skip patterns loaded", path)
            return None
        return SkipSnippets(snippet_patterns=data)
    except FileNotFoundError:
        LOG.error(
            "Couldn't open file with snippet skip patterns `%s`",
            path,
            stack_info=True,
        )
    return None


SERVER_CONFIG_PATH = os.environ.get("LOGDETECTIVE_SERVER_CONF", None)

# The default location for skip patterns is in the same directory
# as logdetective __init__.py file.
SERVER_SKIP_PATTERNS_PATH = os.environ.get(
    "LOGDETECTIVE_SKIP_PATTERNS",
    f"{os.path.dirname(logdetective.__file__)}/skip_snippets.toml",
)

SERVER_CONFIG = load_server_config(SERVER_CONFIG_PATH)
PROMPT_CONFIG = PromptManager(PROMPT_PATH, prompt_config=SERVER_CONFIG.prompts)
SKIP_SNIPPETS_CONFIG = load_skip_snippet_patterns(SERVER_SKIP_PATTERNS_PATH)

LOG = get_log(SERVER_CONFIG)


EMBEDDING_MODEL_INSTANCE = load_embedding_model(SERVER_CONFIG)
