import datetime
from typing import List, Dict, Optional, Any
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    NonNegativeFloat,
    HttpUrl,
)


from logdetective.constants import (
    DEFAULT_TEMPERATURE,
    LLM_DEFAULT_MAX_QUEUE_SIZE,
    LLM_DEFAULT_REQUESTS_PER_MINUTE,
    SYSTEM_ROLE_DEFAULT,
    USER_ROLE_DEFAULT,
)
from logdetective.utils import check_csgrep


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


class EmojiMergeRequest(BaseModel):
    """Model of the 'merge_request' subsection of Emoji webhook messages.
    This model implements only the fields that we care about. The webhook
    sends many more fields that we will ignore."""

    # The identifier of the target project
    target_project_id: int

    # The internal identifier (relative to the target project)
    iid: int


class EmojiHook(BaseModel):
    """Model of Job Hook events sent from GitLab.
    Full details of the specification are available at
    https://docs.gitlab.com/user/project/integrations/webhook_events/#job-events
    This model implements only the fields that we care about. The webhook
    sends many more fields that we will ignore."""

    # The kind of webhook message. We are only interested in 'emoji' messages
    # which represents awarding or revoking emoji reactions on notes.
    object_kind: str = Field(pattern=r"^emoji$")

    # Information about the merge request this emoji applies to, if any.
    merge_request: EmojiMergeRequest = Field(default=None)


class SnippetAnalysis(BaseModel):
    """Model of snippet analysis from LLM."""

    text: str = Field(description="Analysis of log snippet contents.")


class RatedSnippetAnalysis(SnippetAnalysis):
    """Model for rated snippet analysis. This model is used to generate
    json schema for inference with structured output."""

    relevance: int = Field(
        ge=0,
        le=100,
        description="Estimate of likelyhood that snippet contains an error, "
        "with 0 standing for completely unlikely, 100 for absolutely certain.",
    )


class Explanation(BaseModel):
    """Model of snippet or general log explanation from Log Detective"""

    text: str
    logprobs: Optional[List[Dict]] = None

    def __str__(self):
        """Return text of the Explanation"""
        return self.text


class AnalyzedSnippet(BaseModel):
    """Model for snippets already processed by Log Detective.

    explanation: LLM output in form of plain text and logprobs dictionary
    text: original snippet text
    line_number: location of snippet in original log
    """

    explanation: SnippetAnalysis | RatedSnippetAnalysis
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


class KojiStagedResponse(BaseModel):
    """Model of data returned by Log Detective API when called when a Koji build
    analysis is requested. Contains list of reponses to prompts for individual
    snippets.
    """

    task_id: int
    log_file_name: str
    response: StagedResponse


class InferenceConfig(BaseModel):  # pylint: disable=too-many-instance-attributes
    """Model for inference configuration of logdetective server."""

    max_tokens: int = -1
    log_probs: bool = True
    url: str = ""
    # OpenAI client library requires a string to be specified for API token
    # even if it is not checked on the server side
    api_token: str = "None"
    model: str = "default-model"
    temperature: NonNegativeFloat = DEFAULT_TEMPERATURE
    max_queue_size: int = LLM_DEFAULT_MAX_QUEUE_SIZE
    http_timeout: float = 5.0
    user_role: str = USER_ROLE_DEFAULT
    system_role: str = SYSTEM_ROLE_DEFAULT
    llm_api_timeout: float = 15.0
    requests_per_minute: int = LLM_DEFAULT_REQUESTS_PER_MINUTE


class ExtractorConfig(BaseModel):
    """Model for extractor configuration of logdetective server."""

    max_clusters: int = 8
    verbose: bool = False
    max_snippet_len: int = 2000
    csgrep: bool = False

    @field_validator("csgrep", mode="before")
    @classmethod
    def verify_csgrep(cls, v: bool):
        """Verify presence of csgrep binary if csgrep extractor is requested."""
        if v and not check_csgrep():
            raise ValueError(
                "Requested csgrep extractor but `csgrep` binary is not in the PATH"
            )
        return v


class GitLabInstanceConfig(BaseModel):  # pylint: disable=too-many-instance-attributes
    """Model for GitLab configuration of logdetective server."""

    name: str
    url: str = "https://gitlab.com"
    # Path to API of the gitlab instance, assuming `url` as prefix.
    api_path: str = "/api/v4"
    api_token: Optional[str] = None

    # This is a list to support key rotation.
    # When the key is being changed, we will add the new key as a new entry in
    # the configuration and then remove the old key once all of the client
    # webhook configurations have been updated.
    # If this option is left empty or unspecified, all requests will be
    # considered authorized.
    webhook_secrets: Optional[List[str]] = None

    timeout: float = 5.0

    # Maximum size of artifacts.zip in MiB. (default: 300 MiB)
    max_artifact_size: int = 300 * 1024 * 1024

    @field_validator("max_artifact_size", mode="before")
    @classmethod
    def megabytes_to_bytes(cls, v: Any):
        """Convert max_artifact_size from megabytes to bytes."""
        if isinstance(v, int):
            return v * 1024 * 1024
        return 300 * 1024 * 1024


class GitLabConfig(BaseModel):
    """Model for GitLab configuration of logdetective server."""

    instances: Dict[str, GitLabInstanceConfig] = {}

    @model_validator(mode="before")
    @classmethod
    def set_gitlab_instance_configs(cls, data: Any):
        """Initialize configuration for each GitLab instance"""
        if not isinstance(data, dict):
            return data

        instances = {}
        for instance_name, instance_data in data.items():
            instance = GitLabInstanceConfig(name=instance_name, **instance_data)
            instances[instance.url] = instance

        return {"instances": instances}


class KojiInstanceConfig(BaseModel):
    """Model for Koji configuration of logdetective server."""

    name: str = ""
    xmlrpc_url: str = "https://koji.fedoraproject.org/kojihub"
    tokens: List[str] = []


class KojiConfig(BaseModel):
    """Model for Koji configuration of logdetective server."""

    instances: Dict[str, KojiInstanceConfig] = {}
    analysis_timeout: int = 15
    max_artifact_size: int = 300 * 1024 * 1024

    @field_validator("max_artifact_size", mode="before")
    @classmethod
    def megabytes_to_bytes(cls, v: Any):
        """Convert max_artifact_size from megabytes to bytes."""
        if isinstance(v, int):
            return v * 1024 * 1024
        return 300 * 1024 * 1024

    @model_validator(mode="before")
    @classmethod
    def set_koji_instance_configs(cls, data: Any):
        """Initialize configuration for each Koji instance."""
        if isinstance(data, dict):
            instances = {}
            for instance_name, instance_data in data.get("instances", {}).items():
                instances[instance_name] = KojiInstanceConfig(
                    name=instance_name, **instance_data
                )
            data["instances"] = instances
        return data


class LogConfig(BaseModel):
    """Logging configuration"""

    name: str = "logdetective"
    level_stream: str | int = "INFO"
    level_file: str | int = "INFO"
    path: str | None = None
    format: str = "%(levelname)s:%(name)s:%(asctime)s:%(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


class GeneralConfig(BaseModel):
    """General config options for Log Detective"""

    packages: List[str] = []
    excluded_packages: List[str] = []
    devmode: bool = False
    sentry_dsn: HttpUrl | None = None
    collect_emojis_interval: int = 60 * 60  # seconds
    top_k_snippets: int = 0
    report_certainty: bool = False


class Config(BaseModel):
    """Model for configuration of logdetective server."""

    log: LogConfig = Field(default_factory=LogConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    snippet_inference: InferenceConfig = Field(default_factory=InferenceConfig)
    # TODO(jpodivin): Extend to work with multiple extractor configs
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    gitlab: GitLabConfig = Field(default_factory=GitLabConfig)
    koji: KojiConfig = Field(default_factory=KojiConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)

    @model_validator(mode="before")
    @classmethod
    def default_snippet_inference(cls, data: Any):
        """Use base inference configuration, if specific snippet configuration isn't provided."""
        if isinstance(data, dict):
            if "snippet_inference" not in data and "inference" in data:
                data["snippet_inference"] = data["inference"]
        return data


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
        """Check that only one key between weeks, days and hours is defined,
        if no period is specified, fall back to 2 days."""
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
        Will default to 2 days, if no period is set.

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
        else:
            delta = datetime.timedelta(days=2)
        return delta

    def get_period_start_time(
        self, end_time: Optional[datetime.datetime] = None
    ) -> datetime.datetime:
        """Calculate the start time of this period based on the end time.

        Args:
            end_time (datetime.datetime, optional): The end time of the period.
                Defaults to current UTC time if not provided.

        Returns:
            datetime.datetime: The start time of the period.
        """
        time = end_time or datetime.datetime.now(datetime.timezone.utc)
        if time.tzinfo is None:
            time = time.replace(tzinfo=datetime.timezone.utc)
        return time - self.get_time_period()


class MetricTimeSeries(BaseModel):
    """Recorded values of given metric"""
    metric: str
    timestamps: List[datetime.datetime]
    values: List[float]


class MetricResponse(BaseModel):
    """Requested metrics"""
    time_series: List[MetricTimeSeries]
