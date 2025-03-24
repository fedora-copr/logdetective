from logging import BASIC_FORMAT
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class BuildLog(BaseModel):
    """Model of data submitted to API."""

    url: str


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
    build_name: str = Field(pattern=r"^build(_.*)?_rpm$")

    # A string representing the job status. We only care about 'failed' jobs.
    build_status: str = Field(pattern=r"^failed$")

    # The kind of webhook message. We are only interested in 'build' messages
    # which represents job tasks in a pipeline.
    object_kind: str = Field(pattern=r"^build$")

    # The unique ID of the enclosing pipeline on this GitLab instance.
    pipeline_id: int

    # The unique ID of the project triggering this event
    project_id: int


class Response(BaseModel):
    """Model of data returned by Log Detective API

    explanation: CreateCompletionResponse
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama_types.CreateCompletionResponse
    response_certainty: float
    """

    explanation: Dict
    response_certainty: float


class StagedResponse(Response):
    """Model of data returned by Log Detective API when called when staged response
    is requested. Contains list of reponses to prompts for individual snippets.

    explanation: CreateCompletionResponse
        https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama_types.CreateCompletionResponse
    response_certainty: float
    snippets:
        list of dictionaries {
        'snippet' : '<original_text>,
        'comment': CreateCompletionResponse,
        'line_number': '<location_in_log>' }
    """

    snippets: List[Dict[str, str | Dict | int]]


class InferenceConfig(BaseModel):
    """Model for inference configuration of logdetective server."""

    max_tokens: int = -1
    log_probs: int = 1

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.max_tokens = data.get("max_tokens", -1)
        self.log_probs = data.get("log_probs", 1)


class ExtractorConfig(BaseModel):
    """Model for extractor configuration of logdetective server."""

    context: bool = True
    max_clusters: int = 8
    verbose: bool = False

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.context = data.get("context", True)
        self.max_clusters = data.get("max_clusters", 8)
        self.verbose = data.get("verbose", False)


class GitLabConfig(BaseModel):
    """Model for GitLab configuration of logdetective server."""

    url: str = None
    api_url: str = None
    api_token: str = None

    # Maximum size of artifacts.zip in MiB. (default: 300 MiB)
    max_artifact_size: int = 300

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.url = data.get("url", "https://gitlab.com")
        self.api_url = f"{self.url}/api/v4"
        self.api_token = data.get("api_token", None)
        self.max_artifact_size = int(data.get("max_artifact_size")) * 1024 * 1024


class LogConfig(BaseModel):
    """Logging configuration"""

    name: str = "logdetective"
    level: str | int = "INFO"
    path: str | None = None
    format: str = BASIC_FORMAT

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.name = data.get("name", "logdetective")
        self.level = data.get("level", "INFO").upper()
        self.path = data.get("path")
        self.format = data.get("format", BASIC_FORMAT)


class GeneralConfig(BaseModel):
    """General config options for Log Detective"""

    packages: List[str] = None

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.packages = data.get("packages", [])


class Config(BaseModel):
    """Model for configuration of logdetective server."""

    log: LogConfig = LogConfig()
    inference: InferenceConfig = InferenceConfig()
    extractor: ExtractorConfig = ExtractorConfig()
    gitlab: GitLabConfig = GitLabConfig()
    general: GeneralConfig = GeneralConfig()

    def __init__(self, data: Optional[dict] = None):
        super().__init__()

        if data is None:
            return

        self.log = LogConfig(data.get("log"))
        self.inference = InferenceConfig(data.get("inference"))
        self.extractor = ExtractorConfig(data.get("extractor"))
        self.gitlab = GitLabConfig(data.get("gitlab"))
        self.general = GeneralConfig(data.get("general"))
