"""This file contains various constants to be used as a fallback
in case other values are not specified.
"""

import os
import logdetective

# pylint: disable=line-too-long
DEFAULT_ADVISOR = "fedora-copr/granite-3.2-8b-instruct-GGUF"

SNIPPET_DELIMITER = "================"

DEFAULT_TEMPERATURE = 0.0

# Role for chat API
SYSTEM_ROLE_DEFAULT = "developer"

# Other constants

# Default maximum artifact size is 50 MiB,
# for server it can be overwritten in config as max_artifact_size (in MiB)
DEFAULT_MAXIMUM_ARTIFACT_MIB = 50

PROMPT_CONF_PATH = os.environ.get("LOGDETECTIVE_PROMPTS", None)
PROMPT_PATH = os.environ.get(
    "LOGDETECTIVE_PROMPT_TEMPLATES",
    f"{os.path.dirname(logdetective.__file__)}/prompts/",
)
