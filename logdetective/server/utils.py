import yaml
from logdetective.server.models import Config


def load_server_config(path: str | None) -> Config:
    """Load configuration file for logdetective server."""
    if path is not None:
        with open(path, "r") as config_file:
            config = Config(yaml.safe_load(config_file))
            print(config)
            return config
    return Config()
