import os
import re
import yaml
from logdetective.reactor.models import Config

DEFAULT_REACTOR_CONFIG_PATH = "/etc/logdetective/reactor.yaml"
MR_REGEX = re.compile(r"refs/merge-requests/(\d+)/.*$")

config = None


def load_config(path: str) -> Config:
    """Load configuration file for logdetective server."""
    global config

    try:
        with open(path, "r") as config_file:
            config = Config(yaml.safe_load(config_file))
    except FileNotFoundError:
        # This is not an error, we will fall back to default
        print("Unable to find server config file")

    return config


def get_config() -> Config:
    global config

    if config:
        return config

    server_config_path = os.environ.get(
        "LOGDETECTIVE_REACTOR_CONF", DEFAULT_REACTOR_CONFIG_PATH
    )
    config = load_config(server_config_path)
    return config
