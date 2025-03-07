import yaml
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
            pass
    return Config()
