import datetime
from logging import BASIC_FORMAT
from typing import List, Dict, Optional, Literal

from pydantic import BaseModel, Field, model_validator, field_validator


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


class Explanation(BaseModel):
    """Model of snippet or general log explanation from Log Detective"""

    text: str
    logprobs: Optional[List[Dict]] = None

    def __str__(self):
        return self.text


class AnalyzedSnippet(BaseModel):
    """Model for snippets already processed by Log Detective.

    explanation: LLM output in form of plain text and logprobs dictionary
    text: original snippet text
    line_number: location of snippet in original log
    """

    explanation: Explanation
    text: str
    line_number: int


class Response(BaseModel):
    """Model of data returned by Log Detective API

    explanation: Explanation
    response_certainty: float
    """

    explanation: Explanation
    response_certainty: float


class StagedResponse(Response):
    """Model of data returned by Log Detective API when called when staged response
    is requested. Contains list of reponses to prompts for individual snippets.

    explanation: Explanation
    response_certainty: float
    snippets: list of AnalyzedSnippet objects
    """

    snippets: List[AnalyzedSnippet]


class InferenceConfig(BaseModel):
    """Model for inference configuration of logdetective server."""

    max_tokens: int = -1
    log_probs: int = 1
    api_endpoint: Optional[Literal["/chat/completions", "/completions"]] = (
        "/chat/completions"
    )
    url: str = ""
    api_token: str = ""

    def __init__(self, data: Optional[dict] = None):
        super().__init__()
        if data is None:
            return

        self.max_tokens = data.get("max_tokens", -1)
        self.log_probs = data.get("log_probs", 1)
        self.api_endpoint = data.get("api_endpoint", "/chat/completions")
        self.url = data.get("url", "")
        self.api_token = data.get("api_token", "")


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


class TimePeriod(BaseModel):
    """Specification for a period of time.

    If no indication is given
    it falls back to a 2 days period of time.

    Can't be smaller than a hour"""

    weeks: Optional[int] = None
    days: Optional[int] = None
    hours: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def check_exclusive_fields(cls, data):
        """ Check that only one key between weeks, days and hours is defined"""
        if isinstance(data, dict):
            how_many_fields = sum(
                1
                for field in ["weeks", "days", "hours"]
                if field in data and data[field] is not None
            )

            if how_many_fields == 0:
                data["days"] = 2  # by default fallback to a 2 days period

            if how_many_fields > 1:
                raise ValueError("Only one of months, weeks, days, or hours can be set")

        return data

    @field_validator("weeks", "days", "hours")
    @classmethod
    def check_positive(cls, v):
        """Check that the given value is positive"""
        if v is not None and v <= 0:
            raise ValueError("Time period must be positive")
        return v

    def get_time_period(self) -> datetime.timedelta:
        """Get the period of time represented by this input model.

        Returns:
            datetime.timedelta: The time period as a timedelta object.
        """
        delta = None
        if self.weeks:
            delta = datetime.timedelta(weeks=self.weeks)
        elif self.days:
            delta = datetime.timedelta(days=self.days)
        elif self.hours:
            delta = datetime.timedelta(hours=self.hours)
        return delta

    def get_period_start_time(
        self, end_time: datetime.datetime = None
    ) -> datetime.datetime:
        """Calculate the start time of this period based on the end time.

        Args:
            end_time (datetime.datetime, optional): The end time of the period.
                Defaults to current UTC time if not provided.

        Returns:
            datetime.datetime: The start time of the period.
        """
        time = end_time or datetime.datetime.now(datetime.timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=datetime.timezone.utc)
        return time - self.get_time_period()
