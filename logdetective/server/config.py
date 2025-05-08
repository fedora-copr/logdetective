import os
import logging
import yaml
from logdetective.utils import load_prompts
from logdetective.server.models import Config


def load_server_config(path: str | None) -> Config:
    """Load configuration file for logdetective server.
    If no path was provided, or if the file doesn't exist, return defaults.
    """
    if path is not None:
        try:
            with open(path, "r") as config_file:
                return Config(yaml.safe_load(config_file))
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


SERVER_CONFIG_PATH = os.environ.get("LOGDETECTIVE_SERVER_CONF", None)
SERVER_PROMPT_PATH = os.environ.get("LOGDETECTIVE_PROMPTS", None)

SERVER_CONFIG = load_server_config(SERVER_CONFIG_PATH)
PROMPT_CONFIG = load_prompts(SERVER_PROMPT_PATH)

LOG = get_log(SERVER_CONFIG)
