from pathlib import Path

from logdetective.server.utils import load_server_config


def test_loading_config():
    """ Load the actual config we have in this repo """
    # this file - this dir (tests/) - repo root
    repo_root = Path(__file__).parent.parent
    config_file = repo_root / "server" / "config.yml"
    assert load_server_config(str(config_file))
