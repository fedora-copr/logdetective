import logging

from .models import Config
from .config import get_config

log = None


def init_log(config: Config):
    """
    Initialize a logger for this server
    """
    global log
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


def get_log():
    """
    Retrieve the logger for this server, possibly initializing it in the
    process
    """
    global log
    if log and getattr(log, "initialized", False):
        return log
    cfg = get_config()
    log = init_log(cfg)
    return log
