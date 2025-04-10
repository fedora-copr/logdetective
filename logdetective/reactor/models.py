import aiohttp
import gitlab

from logging import BASIC_FORMAT
from typing import List, Optional
from pydantic import BaseModel, Field


class GeneralConfig(BaseModel):
    """General config options for Log Detective"""

    packages: List[str] = None

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.packages = data.get("packages", [])


class GitLabConfig(BaseModel):
    """Model for GitLab configuration of logdetective server."""

    url: str = None
    api_root_path: str = None
    api_token: str = None

    # Maximum size of artifacts.zip in MiB. (default: 300 MiB)
    max_artifact_size: int = 300

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.url = data.get("url", "https://gitlab.com")
        self.api_root_path = f"/api/v4"
        self.api_token = data.get("api_token", None)
        self.max_artifact_size = int(data.get("max_artifact_size")) * 1024 * 1024


class LogConfig(BaseModel):
    """Logging configuration"""

    name: str = "logdetective"
    level_stream: str | int = "INFO"
    level_file: str | int = "INFO"
    path: str | None = None
    format: str = BASIC_FORMAT

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.name = data.get("name", "logdetective")
        self.level_stream = data.get("level_stream", "INFO").upper()
        self.level_file = data.get("level_file", "INFO").upper()
        self.path = data.get("path")
        self.format = data.get("format", BASIC_FORMAT)


class LogDetectiveConfig(BaseModel):
    """Configuration for talking to Log Detective Analysis Server"""

    url: str = None

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.url = data.get("url")


class Config(BaseModel):
    """Model for configuration of logdetective server."""

    general: GeneralConfig = GeneralConfig()
    gitlab: GitLabConfig = GitLabConfig()
    log: LogConfig = LogConfig()
    logdetective: LogDetectiveConfig = LogDetectiveConfig()

    def __init__(self, data: Optional[dict] = None):
        super().__init__()

        if data is None:
            return

        self.gitlab = GitLabConfig(data.get("gitlab"))
        self.general = GeneralConfig(data.get("general"))
        self.log = LogConfig(data.get("log"))
        self.logdetective = LogDetectiveConfig(data.get("logdetective"))


class JobHook(BaseModel):
    """Model of Job Hook events sent from GitLab.
    Full details of the specification are available at
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    This model implements only the fields that we care about. The webhook
    sends many more fields that we will ignore."""

    # The unique job ID on this GitLab instance.
    build_id: int

    # The identifier of the job. We only care about 'build_rpm' and
    # 'build_centos_stream_rpm' jobs.
    build_name: str = Field(pattern=r"^build.*rpm$")

    # A string representing the job status. We only care about 'failed' jobs.
    build_status: str = Field(pattern=r"^failed$")

    # The kind of webhook message. We are only interested in 'build' messages
    # which represents job tasks in a pipeline.
    object_kind: str = Field(pattern=r"^build$")

    # The unique ID of the enclosing pipeline on this GitLab instance.
    pipeline_id: int

    # The unique ID of the project triggering this event
    project_id: int
